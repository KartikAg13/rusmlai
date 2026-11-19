//! [`RandomForestClassifier`] and [`RandomForestRegressor`]: many trees, each grown on its own
//! bootstrap sample of the rows, with a new draw of features at every node; and
//! [`ExtraTreesClassifier`] and [`ExtraTreesRegressor`], the same with random thresholds.
//!
//! The features are put into buckets once (`models/hist.rs`) and every tree reads the same
//! buckets. A bootstrap sample is a count per row, not a copy of the rows. The trees grow as
//! tasks on the Polars pool, and the large nodes of each tree as tasks inside them.

use std::io::{self, Write};

use polars::prelude::{DataType, NamedFrom, Series};
use polars_core::runtime::THREAD_POOL;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dataset::{Dataset, split::Rng},
    metadata::models::CachedRun,
    models::{
        Classifier, Model, ModelError, Regressor, check,
        design::{CHUNK, column},
        hist::{Bins, Grown, Target, feature, grow},
        tree::{
            self, ClassWeight, Learned, MaxFeatures, Tree, add_shares, class_criterion,
            class_target, load, row_weights,
            number_criterion, number_target, pick_classes, share_columns, sum_leaves,
            tree_settings,
        },
    },
};

/// The settings of a forest, with scikit-learn's names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Params {
    n_estimators: usize,
    #[serde(flatten)]
    tree: tree::Params,
    bootstrap: bool,
    oob_score: bool,
    max_samples: Option<f64>,
}

impl Params {
    fn new(criterion: &str, max_features: MaxFeatures) -> Self {
        Self {
            n_estimators: 100,
            tree: tree::Params::new(criterion, max_features),
            bootstrap: true,
            oob_score: false,
            max_samples: None,
        }
    }

    /// Extra Trees: every tree has every row, and each node splits at random thresholds.
    fn extra(criterion: &str, max_features: MaxFeatures) -> Self {
        let mut params: Self = Self::new(criterion, max_features);
        params.bootstrap = false;
        params.tree.splitter = "random".into();
        params
    }

    /// The rows a tree draws from `rows` rows of total weight `weight`; an error when a
    /// setting needs a bootstrap that is off, like scikit-learn.
    fn draws(&self, rows: usize, weight: f64) -> Result<usize, ModelError> {
        let needs_bootstrap = |setting: &'static str, reason: &'static str| {
            (!self.bootstrap).then_some(ModelError::MissingSetting { setting, reason })
        };
        if self.oob_score
            && let Some(e) = needs_bootstrap(
                "bootstrap(true)",
                "oob_score scores each row with the trees that did not draw it, and without a \
                 bootstrap every tree has every row",
            )
        {
            return Err(e);
        }
        let Some(share) = self.max_samples else {
            return Ok(rows);
        };
        match needs_bootstrap(
            "bootstrap(true)",
            "max_samples is how many rows each tree draws, and without a bootstrap no tree draws",
        ) {
            Some(e) => Err(e),
            None => Ok(((share * weight) as usize).max(1)),
        }
    }

    /// The seed of tree `t`'s rows and of its nodes' feature draws.
    fn seeds(&self, t: usize) -> (u64, u64) {
        let mut rng: Rng = Rng::new(self.tree.random_state ^ ((t as u64 + 1) << 32));
        (rng.next_u64(), rng.next_u64())
    }

    /// How often tree `t` drew each row: its bootstrap sample. `None` without a bootstrap,
    /// when every tree has every row once.
    fn in_bag(&self, t: usize, rows: usize, bag: &Bag) -> Option<Vec<f32>> {
        self.bootstrap.then(|| {
            let mut rng: Rng = Rng::new(self.seeds(t).0);
            let mut counts: Vec<f32> = vec![0.0; rows];
            let total: f64 = bag.mass.last().copied().unwrap_or(0.0);
            for _ in 0..bag.draws {
                let row: usize = if bag.mass.is_empty() {
                    (rng.next_u64() % rows as u64) as usize
                } else {
                    // A class by its weight, then a row of it.
                    let at: f64 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
                    let class: &[u32] = &bag.classes[bag.mass.partition_point(|m| *m <= at)];
                    class[(rng.next_u64() % class.len() as u64) as usize] as usize
                };
                counts[row] += 1.0;
            }
            counts
        })
    }
}

/// How the trees of a forest draw their rows. Each row alike, or, with a bootstrap and a
/// `class_weight` of `"balanced"` or set weights, each row as often as its weight says: the
/// trees then get the draws alone, as scikit-learn draws by `sample_weight`.
struct Bag {
    /// The rows each tree draws.
    draws: usize,
    /// The rows of each class, and the weight of the classes up to each one; empty when rows
    /// are drawn alike.
    classes: Vec<Vec<u32>>,
    mass: Vec<f64>,
}

impl Bag {
    fn new(p: &Params, target: &Target, classes: &[String]) -> Result<Self, ModelError> {
        let rows: usize = target.rows();
        let (mut of_class, mut mass): (Vec<Vec<u32>>, Vec<f64>) = (Vec::new(), Vec::new());
        if let (Some(weight), Target::Classes { y, .. }) = (&p.tree.class_weight, target)
            && p.bootstrap
            && !weight.per_tree()
        {
            of_class = vec![Vec::new(); classes.len()];
            for (row, class) in y.iter().enumerate() {
                of_class[*class as usize].push(row as u32);
            }
            let mut total: f64 = 0.0;
            for (rows, w) in of_class.iter().zip(weight.of_classes(classes, y, None)?) {
                total += f64::from(w) * rows.len() as f64;
                mass.push(total);
            }
        }
        let weight: f64 = mass.last().copied().unwrap_or(rows as f64);
        Ok(Self {
            draws: p.draws(rows, weight)?,
            classes: of_class,
            mass,
        })
    }
}

/// The trees of a forest and scikit-learn's `feature_importances_`: the mean of the trees'
/// own (a tree that is one leaf has none), as shares that sum to 1.
/// `classes` are the classes of a classifier's target, for its `class_weight`.
fn grow_forest(
    bins: &Bins,
    target: &Target,
    p: &Params,
    classes: &[String],
    bag: &Bag,
) -> Result<(Vec<Tree>, Vec<f64>), ModelError> {
    let (rows, features): (usize, usize) = (target.rows(), bins.names.len());
    // "balanced_subsample" weighs the classes by the rows each tree drew, times how often it
    // drew them. Other class weights changed the draws (`Bag`), or without a bootstrap are
    // the weights of the rows.
    let class_weight: Option<(&ClassWeight, &[u32])> = match (&p.tree.class_weight, target) {
        (Some(class_weight), Target::Classes { y, .. }) => Some((class_weight, y)),
        _ => None,
    };
    let of_all_rows: Vec<f32> = match class_weight {
        Some((class_weight, y)) => class_weight.of_classes(classes, y, None)?,
        None => Vec::new(),
    };
    let grown: Vec<Grown> = THREAD_POOL.install(|| {
        (0..p.n_estimators)
            .into_par_iter()
            .map(|t| {
                let drawn: Option<Vec<f32>> = p.in_bag(t, rows, bag);
                let weights: Option<Vec<f32>> = match class_weight {
                    Some((class_weight, y)) if class_weight.per_tree() => {
                        let of_drawn = class_weight.of_classes(classes, y, drawn.as_deref())?;
                        Some(row_weights(&of_drawn, y, drawn.as_deref()))
                    }
                    Some((_, y)) if drawn.is_none() => Some(row_weights(&of_all_rows, y, None)),
                    _ => drawn,
                };
                let growth = p.tree.growth(features, p.seeds(t).1);
                Ok(grow(bins, target, &growth, weights.as_deref(), None))
            })
            .collect::<Result<_, ModelError>>()
    })?;
    let mut importances: Vec<f64> = vec![0.0; features];
    let mut trees: Vec<Tree> = Vec::with_capacity(grown.len());
    for g in grown {
        for (sum, share) in importances.iter_mut().zip(&g.importances) {
            *sum += share;
        }
        trees.push(g.tree);
    }
    let sum: f64 = importances.iter().sum();
    if sum > 0.0 {
        importances.iter_mut().for_each(|share| *share /= sum);
    }
    Ok((trees, importances))
}

/// For each row, what the trees that did not draw it say: `width` sums per row (`add` adds a
/// leaf's value), and how many such trees it had. Each tree's sample is drawn again from its
/// seed, so no tree keeps its sample.
#[allow(clippy::too_many_arguments)]
fn out_of_bag(
    trees: &[Tree],
    dataset: &Dataset,
    names: &[String],
    p: &Params,
    bag: &Bag,
    width: usize,
    add: impl Fn(&[f64], &mut [f64]) + Sync,
) -> Result<(Vec<f64>, Vec<u32>), ModelError> {
    let rows: usize = dataset.height();
    let series: Vec<Series> = names
        .iter()
        .map(|name| feature(dataset, name))
        .collect::<Result<_, _>>()?;
    let columns: Vec<&[f64]> = series.iter().map(column).collect();
    let (mut sums, mut votes): (Vec<f64>, Vec<u32>) = (vec![0.0; rows * width], vec![0; rows]);
    for (t, tree) in trees.iter().enumerate() {
        let in_bag: Vec<f32> = p.in_bag(t, rows, bag).unwrap_or_default();
        let outputs: usize = tree.outputs();
        THREAD_POOL.install(|| {
            sums.par_chunks_mut(CHUNK * width)
                .zip(votes.par_chunks_mut(CHUNK))
                .enumerate()
                .for_each_init(
                    || vec![0u32; CHUNK],
                    |leaves, (k, (sums, votes))| {
                        let leaves: &mut [u32] = &mut leaves[..votes.len()];
                        tree.walk(&columns, k * CHUNK, leaves);
                        let drawn: &[f32] = &in_bag[k * CHUNK..][..votes.len()];
                        let cells = sums.chunks_mut(width).zip(votes.iter_mut());
                        for ((leaf, drawn), (cell, votes)) in leaves.iter().zip(drawn).zip(cells) {
                            if *drawn == 0.0 {
                                add(&tree.value()[*leaf as usize * outputs..][..outputs], cell);
                                *votes += 1;
                            }
                        }
                    },
                )
        });
    }
    Ok((sums, votes))
}

/// The settings both forests share, as setters on `$model`.
macro_rules! forest_settings {
    ($model:ident) => {
        tree_settings!($model, params.tree);

        impl $model {
            /// The number of trees. Default 100.
            #[track_caller]
            pub fn n_estimators(mut self, n_estimators: usize) -> Self {
                check(n_estimators > 0, "n_estimators", n_estimators, "1 or more");
                self.params.n_estimators = n_estimators;
                self
            }

            /// `true` (the default) grows each tree on its own sample of the rows, drawn
            /// with repeats; `false` gives every tree every row, so the trees differ only
            /// by the features their nodes draw.
            pub fn bootstrap(mut self, bootstrap: bool) -> Self {
                self.params.bootstrap = bootstrap;
                self
            }

            /// `true` scores each training row with the trees that did not draw it, into
            /// `oob_score_`: a test score without a test set. Default `false`. It needs
            /// `bootstrap(true)`.
            pub fn oob_score(mut self, oob_score: bool) -> Self {
                self.params.oob_score = oob_score;
                self
            }

            /// The rows each tree draws, as a share of the rows above 0 and at most 1.
            /// Default: as many as there are rows. A count (scikit-learn's int) is not taken.
            #[track_caller]
            pub fn max_samples(mut self, max_samples: f64) -> Self {
                let ok: bool = max_samples > 0.0 && max_samples <= 1.0;
                check(ok, "max_samples", max_samples, "a share above 0.0 up to 1.0");
                self.params.max_samples = Some(max_samples);
                self
            }
        }
    };
}

macro_rules! forest_classifier {
    ($(#[$meta:meta])* pub struct $name:ident($default:expr, $new_doc:literal);) => {
$(#[$meta])*
#[derive(Debug, Clone)]
pub struct $name {
    params: Params,
    /// The classes `fit` saw, sorted: the order of `predict_proba`'s columns.
    pub classes_: Vec<String>,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// The mean of the trees' importances, as shares that sum to 1.
    pub feature_importances_: Vec<f64>,
    /// The fitted trees.
    pub estimators_: Vec<Tree>,
    /// With `oob_score(true)`: the accuracy on the training rows, each one predicted by the
    /// trees that did not draw it.
    pub oob_score_: Option<f64>,
    dtype: Option<DataType>,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl Default for $name {
    fn default() -> Self {
        Self {
            params: $default,
            classes_: Vec::new(),
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            feature_importances_: Vec::new(),
            estimators_: Vec::new(),
            oob_score_: None,
            dtype: None,
            target: None,
            run: None,
        }
    }
}

forest_settings!($name);

impl $name {
    #[doc = $new_doc]
    pub fn new() -> Self {
        Self::default()
    }

    /// What a split makes smaller: `"gini"` (the default), or `"entropy"` / `"log_loss"`,
    /// two names of the same measure.
    #[track_caller]
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.params.tree.criterion = class_criterion(criterion);
        self
    }

    /// How much each class counts: `"balanced"`, so that a rare class counts as much as a
    /// frequent one, `"balanced_subsample"`, the same from the rows each tree drew, or your
    /// own weights, `[("yes", 5.0), ("no", 1.0)]`. Default: every row counts once. With a
    /// bootstrap, `"balanced"` and your own weights set how often each row is drawn, as in
    /// scikit-learn; `"balanced_subsample"` weighs the rows each tree drew.
    #[track_caller]
    pub fn class_weight(mut self, class_weight: impl Into<ClassWeight>) -> Self {
        self.params.tree.class_weight = Some(class_weight.into().checked(true));
        self
    }

    /// The sum of the trees' class shares for each row, or the error of a model that is not
    /// fitted.
    fn shares(&self, dataset: &Dataset) -> Result<Vec<f64>, ModelError> {
        if self.target.is_none() {
            return Err(ModelError::NotFitted {
                model: stringify!($name),
            });
        }
        let names: &[String] = &self.feature_names_in_;
        sum_leaves(&self.estimators_, dataset, names, self.classes_.len(), add_shares)
    }
}

impl Model for $name {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let p: &Params = &self.params;
        let (classes, task): (Vec<String>, Target) =
            class_target(dataset, target, &p.tree.criterion)?;
        let bins: Bins = Bins::try_new(dataset, target, p.tree.max_bins)?;
        let bag: Bag = Bag::new(p, &task, &classes)?;
        let (trees, importances): (Vec<Tree>, Vec<f64>) = grow_forest(&bins, &task, p, &classes, &bag)?;
        self.oob_score_ = None;
        if let (true, Target::Classes { y, .. }) = (p.oob_score, &task) {
            let k: usize = classes.len();
            let add = |v: &[f64], sum: &mut [f64]| add_shares(0, v, sum);
            let (sums, votes) = out_of_bag(&trees, dataset, &bins.names, p, &bag, k, add)?;
            // The class with the largest sum, among the rows that some tree did not draw.
            let voted = sums.chunks(k).zip(y).zip(&votes).filter(|(_, votes)| **votes > 0);
            let (hits, rows): (usize, usize) = voted.fold((0, 0), |(hits, rows), ((sums, y), _)| {
                let pick: usize = (1..k).fold(0, |b, c| if sums[c] > sums[b] { c } else { b });
                (hits + usize::from(pick == *y as usize), rows + 1)
            });
            self.oob_score_ = Some(hits as f64 / rows as f64);
        }
        self.dtype = Some(dataset.column(target)?.dtype().clone());
        self.n_features_in_ = bins.names.len();
        (self.feature_names_in_, self.classes_) = (bins.names, classes);
        (self.estimators_, self.feature_importances_) = (trees, importances);
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: self.classes_.clone(),
            feature_names_in_: self.feature_names_in_.clone(),
            feature_importances_: self.feature_importances_.clone(),
            dtype: self.dtype.clone().unwrap_or(DataType::String),
            oob_score_: self.oob_score_,
        })
        .unwrap_or_default()
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        self.estimators_.iter().try_for_each(|tree| tree.write(out))
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        self.params = serde_json::from_value(params)?;
        Ok(())
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        crate::models::evaluate_classifier(self, dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        crate::models::predict_classifier(self, dataset, method)
    }

    fn is_classifier(&self) -> bool {
        true
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let (l, trees): (Learned, Vec<Tree>) = load(learned, weights, true)?;
        self.n_features_in_ = l.feature_names_in_.len();
        (self.feature_names_in_, self.feature_importances_) =
            (l.feature_names_in_, l.feature_importances_);
        (self.classes_, self.dtype, self.oob_score_) = (l.classes_, Some(l.dtype), l.oob_score_);
        self.estimators_ = trees;
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }
}

impl Classifier for $name {
    /// The class with the largest mean share over the trees; the first one on a tie.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        pick_classes(
            &self.shares(dataset)?,
            &self.classes_,
            self.target.as_deref().unwrap_or_default(),
            self.dtype.as_ref().unwrap_or(&DataType::String),
        )
    }

    /// The mean over the trees of the share of each class in the row's leaf.
    fn try_predict_proba(&self, dataset: &Dataset) -> Result<Dataset, ModelError> {
        let trees: usize = self.estimators_.len();
        share_columns(dataset, &self.shares(dataset)?, &self.classes_, trees)
    }
}
    };
}

forest_classifier! {
    /// Predicts a class as the vote of many decision trees, like scikit-learn's
    /// `RandomForestClassifier`.
    ///
    /// Each of the `n_estimators` trees grows on a bootstrap sample of the rows (as many rows,
    /// drawn with repeats) and looks at `max_features` (default `"sqrt"`) randomly drawn features
    /// at each node. `predict_proba` is the mean of the trees' class shares, and `predict` the
    /// class with the largest mean, like scikit-learn's soft vote. One tree learns its rows by
    /// heart; the mean of many different ones does not.
    ///
    /// The trees are those of [`DecisionTreeClassifier`](crate::models::DecisionTreeClassifier):
    /// features in at most `max_bins` buckets, missing values as they are, and the same settings.
    /// The buckets are made once for all trees. `random_state` (default 42) seeds every draw, so
    /// a fit gives the same forest on every run and on any number of threads. The draws are not
    /// scikit-learn's, so the trees differ from its trees, and the scores agree as two of its
    /// own seeds agree.
    ///
    /// The cache keeps a forest's trees up to 256 MB. A larger forest still trains and predicts;
    /// it is only not stored next to the CSV, with a warning.
    ///
    /// scikit-learn settings with no setter: `n_jobs` (the trees run on the Polars pool,
    /// `POLARS_MAX_THREADS`), `verbose`. Not yet: `warm_start`, `min_weight_fraction_leaf`,
    /// `ccp_alpha`, `monotonic_cst`, and `max_samples` as a count.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "hours" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    ///         "sleep" => [4.0, 5.0, 4.5, 5.5, 4.0, 5.0, 7.0, 8.0, 7.5, 8.5, 7.0, 8.0],
    ///         "passed" => ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// let mut model = RandomForestClassifier::new().n_estimators(20).random_state(42);
    /// model.fit(&ds, "passed");
    /// assert_eq!(model.estimators_.len(), 20);
    /// assert_eq!(model.classes_, ["no", "yes"]);
    /// assert_eq!(model.score(&ds, "passed"), 1.0);
    /// let importance: f64 = model.feature_importances_.iter().sum();
    /// assert!((importance - 1.0).abs() < 1e-12);
    ///
    /// let proba = model.predict_proba(&ds); // the mean of the 20 trees' shares
    /// assert_eq!(proba.columns(), ["no", "yes"]);
    /// assert_eq!(proba.to_list("yes")[11], 1.0);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.ensemble import RandomForestClassifier
    /// ds = pd.DataFrame({
    ///     "hours": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
    ///     "sleep": [4.0, 5.0, 4.5, 5.5, 4.0, 5.0, 7.0, 8.0, 7.5, 8.5, 7.0, 8.0],
    ///     "passed": ["no", "no", "no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    /// })
    /// x, y = ds[["hours", "sleep"]], ds["passed"]
    /// model = RandomForestClassifier(n_estimators=20, random_state=42).fit(x, y)
    /// assert len(model.estimators_) == 20
    /// assert model.classes_.tolist() == ["no", "yes"]
    /// assert model.score(x, y) == 1.0
    /// assert abs(model.feature_importances_.sum() - 1.0) < 1e-12
    ///
    /// proba = model.predict_proba(x)
    /// assert proba.shape == (12, 2)
    /// assert proba[11, 1] == 1.0
    /// ```
    pub struct RandomForestClassifier(Params::new("gini", "sqrt".into()), "100 full trees: gini, a bootstrap sample each, `\"sqrt\"` of the features per node.");
}

macro_rules! forest_regressor {
    ($(#[$meta:meta])* pub struct $name:ident($default:expr, $new_doc:literal);) => {
$(#[$meta])*
#[derive(Debug, Clone)]
pub struct $name {
    params: Params,
    /// The feature columns `fit` saw.
    pub feature_names_in_: Vec<String>,
    /// How many features `fit` saw.
    pub n_features_in_: usize,
    /// The mean of the trees' importances, as shares that sum to 1.
    pub feature_importances_: Vec<f64>,
    /// The fitted trees.
    pub estimators_: Vec<Tree>,
    /// With `oob_score(true)`: R² on the training rows, each one predicted by the trees that
    /// did not draw it.
    pub oob_score_: Option<f64>,
    target: Option<String>,
    run: Option<CachedRun>,
}

impl Default for $name {
    fn default() -> Self {
        Self {
            params: $default,
            feature_names_in_: Vec::new(),
            n_features_in_: 0,
            feature_importances_: Vec::new(),
            estimators_: Vec::new(),
            oob_score_: None,
            target: None,
            run: None,
        }
    }
}

forest_settings!($name);

impl $name {
    #[doc = $new_doc]
    pub fn new() -> Self {
        Self::default()
    }

    /// What a split makes smaller: `"squared_error"`, the only one so far.
    #[track_caller]
    pub fn criterion(mut self, criterion: &str) -> Self {
        self.params.tree.criterion = number_criterion(criterion);
        self
    }
}

impl Model for $name {
    fn try_train(&mut self, dataset: &Dataset, target: &str) -> Result<(), ModelError> {
        let p: &Params = &self.params;
        let task: Target = number_target(dataset, target)?;
        let bins: Bins = Bins::try_new(dataset, target, p.tree.max_bins)?;
        let bag: Bag = Bag::new(p, &task, &[])?;
        let (trees, importances): (Vec<Tree>, Vec<f64>) = grow_forest(&bins, &task, p, &[], &bag)?;
        self.oob_score_ = None;
        if let (true, Target::Regression { y }) = (p.oob_score, &task) {
            let add = |mean: &[f64], sum: &mut [f64]| sum[0] += mean[0];
            let (sums, votes) = out_of_bag(&trees, dataset, &bins.names, p, &bag, 1, add)?;
            // R² over the rows that some tree did not draw.
            let voted = || sums.iter().zip(y).zip(&votes).filter(|(_, votes)| **votes > 0);
            let (total, rows): (f64, f64) =
                voted().fold((0.0, 0.0), |(total, rows), ((_, y), _)| (total + y, rows + 1.0));
            let (residual, spread): (f64, f64) =
                voted().fold((0.0, 0.0), |(residual, spread), ((sum, y), votes)| {
                    let error: f64 = y - sum / f64::from(*votes);
                    let off: f64 = y - total / rows;
                    (residual + error * error, spread + off * off)
                });
            self.oob_score_ = Some(1.0 - residual / spread);
        }
        self.n_features_in_ = bins.names.len();
        self.feature_names_in_ = bins.names;
        (self.estimators_, self.feature_importances_) = (trees, importances);
        self.target = Some(target.into());
        Ok(())
    }

    fn params(&self) -> Value {
        serde_json::to_value(&self.params).unwrap_or_default()
    }

    fn learned(&self) -> Value {
        serde_json::to_value(Learned {
            classes_: Vec::new(),
            feature_names_in_: self.feature_names_in_.clone(),
            feature_importances_: self.feature_importances_.clone(),
            dtype: DataType::Float64,
            oob_score_: self.oob_score_,
        })
        .unwrap_or_default()
    }

    fn write_weights(&self, out: &mut dyn Write) -> io::Result<()> {
        self.estimators_.iter().try_for_each(|tree| tree.write(out))
    }

    fn try_set_params(&mut self, params: Value) -> Result<(), ModelError> {
        self.params = serde_json::from_value(params)?;
        Ok(())
    }

    fn try_evaluate(&self, dataset: &Dataset, target: &str, scoring: &str) -> Result<f64, ModelError> {
        crate::models::evaluate_regressor(self, dataset, target, scoring)
    }

    fn try_predict_method(&self, dataset: &Dataset, method: &str) -> Result<Dataset, ModelError> {
        crate::models::predict_regressor(self, dataset, method)
    }

    fn try_load(&mut self, learned: Value, weights: &[u8], target: &str) -> Result<(), ModelError> {
        let (l, trees): (Learned, Vec<Tree>) = load(learned, weights, false)?;
        self.n_features_in_ = l.feature_names_in_.len();
        (self.feature_names_in_, self.feature_importances_) =
            (l.feature_names_in_, l.feature_importances_);
        (self.estimators_, self.oob_score_) = (trees, l.oob_score_);
        self.target = Some(target.into());
        Ok(())
    }

    fn run(&self) -> Option<&CachedRun> {
        self.run.as_ref()
    }

    fn set_run(&mut self, run: Option<CachedRun>) {
        self.run = run;
    }
}

impl Regressor for $name {
    /// The mean over the trees of the mean target of each row's leaf.
    fn try_predict(&self, dataset: &Dataset) -> Result<Series, ModelError> {
        let target: &String = self.target.as_ref().ok_or(ModelError::NotFitted {
            model: stringify!($name),
        })?;
        let (trees, names) = (&self.estimators_, &self.feature_names_in_);
        let mut predicted: Vec<f64> =
            sum_leaves(trees, dataset, names, 1, |_, mean, sum| sum[0] += mean[0])?;
        let count: f64 = trees.len() as f64;
        predicted.iter_mut().for_each(|sum| *sum /= count);
        Ok(Series::new(target.into(), predicted))
    }
}
    };
}

forest_regressor! {
    /// Predicts a number as the mean of many decision trees, like scikit-learn's
    /// `RandomForestRegressor`.
    ///
    /// Each of the `n_estimators` trees grows on a bootstrap sample of the rows (as many rows,
    /// drawn with repeats), and `predict` is the mean of what the trees predict. `max_features`
    /// is `1.0` by default, every feature at every node, like scikit-learn: the trees then
    /// differ by their rows only. A share such as `0.33` makes them differ more.
    ///
    /// The trees, the buckets, the seed and the cache are those of [`RandomForestClassifier`].
    ///
    /// scikit-learn settings with no setter: `n_jobs` (the trees run on the Polars pool,
    /// `POLARS_MAX_THREADS`), `verbose`. Not yet: the criteria `"friedman_mse"`,
    /// `"absolute_error"` and `"poisson"`, `warm_start`, `min_weight_fraction_leaf`, `ccp_alpha`,
    /// `monotonic_cst`, and `max_samples` as a count.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let ds = Dataset::new(
    ///     df!(
    ///         "area" => [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///         "rooms" => [1, 2, 2, 3, 3, 3, 4, 4],
    ///         "price" => [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    ///     )
    ///     .unwrap(),
    ///     "",
    /// );
    /// let mut model = RandomForestRegressor::new().n_estimators(50).random_state(42);
    /// model.fit(&ds, "price");
    /// assert_eq!(model.estimators_.len(), 50);
    /// assert!(model.score(&ds, "price") > 0.9); // the mean of 50 trees, not one tree's 1.0
    ///
    /// // A forest never predicts outside the targets it saw.
    /// let predicted = model.predict(&ds).f64().unwrap().to_vec();
    /// assert!(predicted.iter().all(|p| (150.0..=420.0).contains(&p.unwrap())));
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import pandas as pd
    /// from sklearn.ensemble import RandomForestRegressor
    /// ds = pd.DataFrame({
    ///     "area": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    ///     "rooms": [1, 2, 2, 3, 3, 3, 4, 4],
    ///     "price": [150.0, 180.0, 200.0, 260.0, 270.0, 310.0, 400.0, 420.0],
    /// })
    /// x, y = ds[["area", "rooms"]], ds["price"]
    /// model = RandomForestRegressor(n_estimators=50, random_state=42).fit(x, y)
    /// assert len(model.estimators_) == 50
    /// assert model.score(x, y) > 0.9
    ///
    /// predicted = model.predict(x)
    /// assert ((predicted >= 150.0) & (predicted <= 420.0)).all()
    /// ```
    pub struct RandomForestRegressor(Params::new("squared_error", MaxFeatures::Share(1.0)), "100 full trees: squared error, a bootstrap sample each, every feature at every node.");
}

forest_classifier! {
    /// Predicts a class as the vote of many extremely randomized trees, like scikit-learn's
    /// `ExtraTreesClassifier`.
    ///
    /// A random forest whose trees split at random: at each node, each of the `max_features`
    /// (default `"sqrt"`) features drawn offers one split, at a threshold drawn uniformly
    /// between its smallest and largest value among the node's rows, and the best of those
    /// splits wins. And every tree sees every row (`bootstrap` is `false` by default). The
    /// trees vary more and each one fits less, so the vote smooths more than a random
    /// forest's, and a tree costs less: no scan for the best threshold.
    ///
    /// The thresholds come from rusmlai's own generator, so the trees are not scikit-learn's
    /// tree for tree; the scores are alike. The settings are [`RandomForestClassifier`]'s.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let x: Vec<f64> = (0..200).map(|i| (i * 37 % 101) as f64).collect();
    /// let z: Vec<f64> = (0..200).map(|i| (i * 53 % 17) as f64).collect();
    /// let y: Vec<&str> = x.iter().zip(&z).map(|(x, z)| if x + 3.0 * z > 80.0 { "hi" } else { "lo" }).collect();
    /// let ds = Dataset::new(df!("x" => x, "z" => z, "y" => y).unwrap(), "");
    /// let mut model = ExtraTreesClassifier::new().n_estimators(50);
    /// model.fit(&ds, "y");
    /// assert_eq!(model.estimators_.len(), 50);
    /// assert!(model.score(&ds, "y") > 0.99); // every row, full trees
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import numpy as np
    /// import pandas as pd
    /// from sklearn.ensemble import ExtraTreesClassifier
    /// i = np.arange(200)
    /// ds = pd.DataFrame({"x": (i * 37 % 101) * 1.0, "z": (i * 53 % 17) * 1.0})
    /// ds["y"] = np.where(ds["x"] + 3 * ds["z"] > 80, "hi", "lo")
    /// model = ExtraTreesClassifier(n_estimators=50, random_state=42).fit(ds[["x", "z"]], ds["y"])
    /// assert len(model.estimators_) == 50
    /// assert model.score(ds[["x", "z"]], ds["y"]) > 0.99
    /// ```
    pub struct ExtraTreesClassifier(Params::extra("gini", "sqrt".into()), "100 full trees: gini, every row, `\"sqrt\"` of the features per node, random thresholds.");
}

forest_regressor! {
    /// Predicts a number as the mean of many extremely randomized trees, like scikit-learn's
    /// `ExtraTreesRegressor`.
    ///
    /// A random forest whose trees split at random: at each node, every feature (the default
    /// `max_features`, 1.0) offers one split, at a threshold drawn uniformly between its
    /// smallest and largest value among the node's rows, and the best of those splits wins.
    /// Every tree sees every row (`bootstrap` is `false` by default). The thresholds come from
    /// rusmlai's own generator, so the trees are not scikit-learn's tree for tree. The
    /// settings are [`RandomForestRegressor`]'s.
    ///
    /// ```
    /// use rusmlai::prelude::*;
    /// let x: Vec<f64> = (0..200).map(|i| (i * 37 % 101) as f64).collect();
    /// let y: Vec<f64> = x.iter().map(|x| (x / 10.0).sin() * 10.0).collect();
    /// let ds = Dataset::new(df!("x" => x, "y" => y).unwrap(), "");
    /// let mut model = ExtraTreesRegressor::new().n_estimators(30);
    /// model.fit(&ds, "y");
    /// assert!(model.score(&ds, "y") > 0.99);
    /// ```
    ///
    /// Python:
    ///
    /// ```python
    /// import numpy as np
    /// import pandas as pd
    /// from sklearn.ensemble import ExtraTreesRegressor
    /// x = (np.arange(200) * 37 % 101) * 1.0
    /// ds = pd.DataFrame({"x": x, "y": np.sin(x / 10) * 10})
    /// model = ExtraTreesRegressor(n_estimators=30, random_state=42).fit(ds[["x"]], ds["y"])
    /// assert model.score(ds[["x"]], ds["y"]) > 0.99
    /// ```
    pub struct ExtraTreesRegressor(Params::extra("squared_error", MaxFeatures::Share(1.0)), "100 full trees: squared error, every row, every feature at every node, random thresholds.");
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::models::{DecisionTreeClassifier, DecisionTreeRegressor};

    /// 600 rows: two features that decide the class and the number, one that is noise.
    fn table() -> Dataset {
        let mut rng: Rng = Rng::new(5);
        let mut draw = |levels: u64| -> Vec<f64> {
            (0..600).map(|_| (rng.next_u64() % levels) as f64).collect()
        };
        let (a, b, noise) = (draw(50), draw(20), draw(30));
        let (wobble, flip) = (draw(15), draw(10));
        let score: Vec<f64> = (0..600).map(|i| a[i] + 2.0 * b[i] + wobble[i]).collect();
        // One row in ten has the other class: a full tree learns that noise, a forest less.
        let class: Vec<&str> = (0..600)
            .map(|i| if (score[i] > 50.0) != (flip[i] == 0.0) { "high" } else { "low" })
            .collect();
        Dataset::new(
            df!("a" => a, "b" => b, "noise" => noise, "class" => class, "y" => score).unwrap(),
            "",
        )
    }

    fn halves(ds: &Dataset, drop: &str) -> (Dataset, Dataset) {
        let frame = ds.drop(&[drop]).into_dataframe();
        (Dataset::new(frame.slice(0, 400), ""), Dataset::new(frame.slice(400, 200), ""))
    }

    #[test]
    fn without_a_bootstrap_and_with_every_feature_each_tree_is_the_decision_tree() {
        let (train, test) = halves(&table(), "y");
        let mut forest = RandomForestClassifier::new().n_estimators(3).bootstrap(false).max_features(1.0);
        forest.fit(&train, "class");
        let mut one: DecisionTreeClassifier = DecisionTreeClassifier::new();
        one.fit(&train, "class");
        assert!(forest.estimators_.iter().all(|tree| *tree == one.tree_));
        assert_eq!(forest.predict(&test), one.predict(&test));
        assert_eq!(forest.predict_proba(&test).to_list("high"), one.predict_proba(&test).to_list("high"));
        for (got, want) in forest.feature_importances_.iter().zip(&one.feature_importances_) {
            assert!((got - want).abs() < 1e-12);
        }

        let (train, test) = halves(&table(), "class");
        let mut forest = RandomForestRegressor::new().n_estimators(2).bootstrap(false);
        forest.fit(&train, "y");
        let mut one: DecisionTreeRegressor = DecisionTreeRegressor::new();
        one.fit(&train, "y");
        assert_eq!(forest.estimators_[1], one.tree_);
        assert_eq!(forest.predict(&test), one.predict(&test));
    }

    #[test]
    fn a_forest_scores_better_on_new_rows_than_one_tree_and_the_seed_decides() {
        let (train, test) = halves(&table(), "y");
        let fit = |model: RandomForestClassifier| {
            let mut model: RandomForestClassifier = model;
            model.fit(&train, "class");
            model
        };
        let forest = fit(RandomForestClassifier::new().oob_score(true));
        let mut one: DecisionTreeClassifier = DecisionTreeClassifier::new();
        one.fit(&train, "class");
        let (forest_test, tree_test) = (forest.score(&test, "class"), one.score(&test, "class"));
        assert!(forest_test > tree_test, "forest {forest_test}, tree {tree_test}");
        // The out-of-bag score is a test score made from the training rows.
        let oob: f64 = forest.oob_score_.unwrap();
        assert!((oob - forest_test).abs() < 0.06, "oob {oob}, test {forest_test}");
        // The noise column decides nothing, and the importances say so.
        let importances: &[f64] = &forest.feature_importances_;
        assert!(importances[2] < importances[0] && importances[2] < importances[1]);
        assert!((importances.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let small = |seed: u64| fit(RandomForestClassifier::new().n_estimators(5).random_state(seed));
        let (a, again, b) = (small(1), small(1), small(2));
        assert_eq!(a.estimators_, again.estimators_, "the same seed, the same forest");
        assert_ne!(a.estimators_, b.estimators_);
        assert_ne!(a.estimators_[0], a.estimators_[1], "each tree has its own rows");
        // A tree counts the rows it drew once each: about 63% of them.
        let drawn: u32 = a.estimators_[0].n_node_samples()[0];
        assert!((230..=275).contains(&drawn), "{drawn} of 400 rows");
        let half = fit(RandomForestClassifier::new().n_estimators(5).max_samples(0.5));
        assert!(half.estimators_[0].n_node_samples()[0] < 180);
    }

    #[test]
    fn the_regressor_averages_its_trees_and_scores_its_out_of_bag_rows() {
        let (train, test) = halves(&table(), "class");
        let mut forest = RandomForestRegressor::new().n_estimators(60).oob_score(true);
        forest.fit(&train, "y");
        let mut one: DecisionTreeRegressor = DecisionTreeRegressor::new();
        one.fit(&train, "y");
        let (forest_test, tree_test) = (forest.score(&test, "y"), one.score(&test, "y"));
        assert!(forest_test > tree_test, "forest {forest_test}, tree {tree_test}");
        let oob: f64 = forest.oob_score_.unwrap();
        assert!((oob - forest_test).abs() < 0.05, "oob {oob}, test {forest_test}");
        // The mean of the trees, by hand, for the first test row.
        let row: Dataset = Dataset::new(test.dataframe().slice(0, 1), "");
        let by_hand: f64 = forest
            .estimators_
            .iter()
            .map(|tree| {
                let mut one: DecisionTreeRegressor = DecisionTreeRegressor::new();
                one.fit(&train, "y"); // only to get a fitted model to put the tree in
                one.tree_ = tree.clone();
                one.predict(&row).f64().unwrap().get(0).unwrap()
            })
            .sum::<f64>()
            / 60.0;
        let predicted: f64 = forest.predict(&row).f64().unwrap().get(0).unwrap();
        assert!((predicted - by_hand).abs() < 1e-9);
    }

    #[test]
    fn learned_values_round_trip_and_settings_that_need_a_bootstrap_say_so() {
        let (train, test) = halves(&table(), "y");
        let mut trained = RandomForestClassifier::new().n_estimators(4).oob_score(true);
        trained.fit(&train, "class");
        let mut weights: Vec<u8> = Vec::new();
        trained.write_weights(&mut weights).unwrap();
        let mut loaded: RandomForestClassifier = RandomForestClassifier::new();
        loaded.try_load(trained.learned(), &weights, "class").unwrap();
        assert_eq!(loaded.estimators_, trained.estimators_);
        assert_eq!(loaded.oob_score_, trained.oob_score_);
        assert_eq!(loaded.predict(&test), trained.predict(&test));
        let mut again: RandomForestClassifier = RandomForestClassifier::new();
        again.try_set_params(trained.params()).unwrap();
        assert_eq!(again.params(), trained.params());
        assert_eq!(trained.params()["max_features"], "sqrt");
        assert_eq!(trained.params()["n_estimators"], 4);

        let fit = |model: RandomForestClassifier| {
            let mut model: RandomForestClassifier = model;
            model.try_train(&train, "class").unwrap_err().to_string()
        };
        let no_bootstrap = || RandomForestClassifier::new().bootstrap(false);
        assert!(fit(no_bootstrap().oob_score(true)).starts_with("bootstrap(true) must be set: oob_score"));
        assert!(fit(no_bootstrap().max_samples(0.5)).starts_with("bootstrap(true) must be set: max_samples"));
        assert!(matches!(
            RandomForestRegressor::new().try_predict(&test),
            Err(ModelError::NotFitted { .. })
        ));
    }

    #[test]
    fn balanced_subsample_weighs_the_classes_by_the_rows_each_tree_drew() {
        let (train, _) = halves(&table(), "y");
        let fit = |model: RandomForestClassifier| {
            let mut model: RandomForestClassifier = model.n_estimators(3);
            model.fit(&train, "class");
            model
        };
        let plain = fit(RandomForestClassifier::new());
        let balanced = fit(RandomForestClassifier::new().class_weight("balanced"));
        let subsample = fit(RandomForestClassifier::new().class_weight("balanced_subsample"));
        assert_ne!(plain.estimators_, balanced.estimators_);
        assert_ne!(balanced.estimators_, subsample.estimators_);
        // Each class of the rows a tree drew weighs the same in all.
        for tree in &subsample.estimators_ {
            let root: &[f64] = &tree.value()[..2];
            assert!((root[0] - root[1]).abs() < 1e-3, "{root:?}");
        }
        let mine = fit(RandomForestClassifier::new().class_weight([("high", 1.0), ("low", 4.0)]));
        assert_eq!(mine.params()["class_weight"]["low"], 4.0);
        // Set weights change the draws, as scikit-learn's: each tree has 400 whole rows, a
        // "low" one drawn 4 times as often as a "high" one.
        let classes = train.dataframe().column("class").unwrap().str().unwrap().clone();
        let low: f64 = classes.iter().filter(|c| *c == Some("low")).count() as f64;
        let want: f64 = 4.0 * low / (400.0 - low + 4.0 * low);
        for tree in &mine.estimators_ {
            let root: &[f64] = &tree.value()[..2];
            assert_eq!(root[0] + root[1], 400.0);
            assert!((root[1] / 400.0 - want).abs() < 0.06, "{root:?} against {want}");
        }
    }

    #[test]
    #[should_panic(expected = "max_features = cube is not valid")]
    fn a_bad_setting_panics_at_the_setter() {
        let _ = RandomForestClassifier::new().max_features("cube");
    }
    /// scikit-learn 1.9's range over 20 seeds on this split
    /// (analysis/2026-11-18/new-features/extra-trees-parity.py): the test score and the mean
    /// node count of 50 trees.
    #[test]
    fn extra_trees_score_and_grow_like_scikit_learn() {
        let n: i64 = 2000;
        let i: Vec<i64> = (0..n).collect();
        let a: Vec<f64> = i.iter().map(|i| (i * 37 % 101) as f64).collect();
        let b: Vec<f64> = i.iter().map(|i| (i * 53 % 17) as f64).collect();
        let c: Vec<f64> = i.iter().map(|i| (i * 11 % 7) as f64).collect();
        let score: Vec<f64> = (0..n as usize).map(|k| a[k] + 3.0 * b[k] - 5.0 * c[k] + (i[k] * 29 % 13) as f64).collect();
        let class: Vec<i64> = score.iter().map(|s| ((s / 40.0).floor() as i64).rem_euclid(3)).collect();
        let y: Vec<f64> = score.iter().map(|s| s / 4.0).collect();
        let ds: Dataset = Dataset::new(df!("a" => a, "b" => b, "c" => c, "class" => class, "y" => y).unwrap(), "");
        let train: Dataset = Dataset::new(ds.dataframe().slice(0, 1500), "");
        let test: Dataset = Dataset::new(ds.dataframe().slice(1500, 500), "");
        let nodes = |trees: &[Tree]| trees.iter().map(|t| t.node_count() as f64).sum::<f64>() / trees.len() as f64;
        let within = |got: f64, (low, high): (f64, f64), slack: f64| {
            assert!(got >= low - slack && got <= high + slack, "{got} not in [{low}, {high}]");
        };
        for (depth, scores, counts) in [(None, (0.89, 0.912), (1179.3, 1207.6)), (Some(6), (0.824, 0.86), (105.0, 111.6))] {
            let mut model = ExtraTreesClassifier::new().n_estimators(50);
            if let Some(d) = depth {
                model = model.max_depth(d);
            }
            model.try_train(&train.drop(&["y"]), "class").unwrap();
            within(model.try_evaluate(&test.drop(&["y"]), "class", "").unwrap(), scores, 0.01);
            within(nodes(&model.estimators_), counts, 0.03 * counts.1);
        }
        for (depth, scores, counts) in [(None, (0.9784, 0.9796), (2997.1, 2997.6)), (Some(6), (0.961, 0.9657), (124.0, 125.6))] {
            let mut model = ExtraTreesRegressor::new().n_estimators(50);
            if let Some(d) = depth {
                model = model.max_depth(d);
            }
            model.try_train(&train.drop(&["class"]), "y").unwrap();
            within(model.try_evaluate(&test.drop(&["class"]), "y", "").unwrap(), scores, 0.003);
            within(nodes(&model.estimators_), counts, 0.03 * counts.1);
        }
    }

}
