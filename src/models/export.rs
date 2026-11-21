//! Boosted models in their library's own file formats, which Python reads: XGBoost's JSON
//! model (`xgboost.Booster().load_model`, `XGBRegressor().load_model`) and LightGBM's text
//! model (`lightgbm.Booster(model_file=...)`). Each tree is written as it is, its leaves
//! already scaled by the learning rate, so the library predicts what rusmlai predicts.

use std::{fmt::Write as _, fs, path::Path};

use serde_json::{Value, json};

use crate::{
    metadata::utils::print_saved,
    models::{ModelError, tree::Tree},
};

/// The error of a model the file format can't hold.
fn unsupported(format: &str, reason: &'static str) -> ModelError {
    ModelError::InvalidSetting { setting: "save_model", value: format.into(), expected: reason }
}

/// Every tree splits numbers only: a categorical split has no counterpart written here yet.
fn numbers_only(trees: &[Tree], format: &str) -> Result<(), ModelError> {
    if trees.iter().any(|t| t.is_categorical().contains(&1)) {
        return Err(unsupported(format, "a model without categorical features: their splits are not written yet"));
    }
    Ok(())
}

/// What XGBoost needs to know of a fitted model.
pub(crate) struct XgbModel<'a> {
    pub trees: &'a [Tree],
    /// The start of every score, in margin space: one per class for more than two classes.
    pub baseline: &'a [f64],
    pub names: &'a [String],
    pub objective: &'a str,
}

/// XGBoost's JSON model (format of xgboost 3.x). `base_score` is written in XGBoost's output
/// space: a probability for `binary:logistic`, exp of the margin for the log links.
pub(crate) fn xgboost_json(m: &XgbModel, path: &Path) -> Result<(), ModelError> {
    numbers_only(m.trees, "XGBoost JSON")?;
    let k: usize = m.baseline.len();
    let objective: &str = match (m.objective, k) {
        ("multi:softmax" | "multi:softprob", _) | (_, 3..) => "multi:softprob",
        (name, _) => name,
    };
    let output = |margin: f64| -> f64 {
        match objective {
            "binary:logistic" => 1.0 / (1.0 + (-margin).exp()),
            "count:poisson" | "reg:gamma" | "reg:tweedie" => margin.exp(),
            _ => margin,
        }
    };
    let base: Vec<String> = m.baseline.iter().map(|b| format!("{:E}", output(*b) as f32)).collect();
    let trees: Vec<Value> = m.trees.iter().enumerate().map(|(id, tree)| xgb_tree(tree, id, m.names.len())).collect();
    let objective_json: Value = match objective {
        "multi:softprob" => json!({"name": objective, "softmax_multiclass_param": {"num_class": k.to_string()}}),
        "count:poisson" => json!({"name": objective, "poisson_regression_param": {"max_delta_step": "0.7"}}),
        "reg:tweedie" => json!({"name": objective, "tweedie_regression_param": {"tweedie_variance_power": "1.5"}}),
        "reg:pseudohubererror" => json!({"name": objective, "pseudo_huber_param": {"huber_slope": "1"}}),
        "reg:quantileerror" => json!({"name": objective, "quantile_loss_param": {"quantile_alpha": "[0.5]"}}),
        _ => json!({"name": objective, "reg_loss_param": {"scale_pos_weight": "1"}}),
    };
    let model: Value = json!({
        "learner": {
            "attributes": {},
            "feature_names": m.names,
            "feature_types": vec!["float"; m.names.len()],
            "gradient_booster": {
                "model": {
                    "cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                    "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": m.trees.len().to_string()},
                    "iteration_indptr": (0..=m.trees.len() / k).map(|r| r * k).collect::<Vec<usize>>(),
                    "tree_info": (0..m.trees.len()).map(|t| if k > 1 { t % k } else { 0 }).collect::<Vec<usize>>(),
                    "trees": trees,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": format!("[{}]", base.join(",")),
                "boost_from_average": "1",
                "num_class": if k > 1 { k.to_string() } else { "0".into() },
                "num_feature": m.names.len().to_string(),
                "num_target": "1",
            },
            "objective": objective_json,
        },
        "version": [3, 4, 1],
    });
    fs::write(path, serde_json::to_vec(&model)?)?;
    print_saved(path);
    Ok(())
}

/// One tree as XGBoost's parallel arrays, in rusmlai's node order (XGBoost reads any order
/// whose parents and children agree). A row goes left in XGBoost when its value is below
/// `split_conditions`: rusmlai's "at most the threshold" is "below the next float up".
fn xgb_tree(tree: &Tree, id: usize, features: usize) -> Value {
    let n: usize = tree.node_count();
    let (feature, threshold, right, missing_left) = (tree.feature(), tree.threshold(), tree.children_right(), tree.missing_go_to_left());
    let mut left_children: Vec<i64> = vec![-1; n];
    let mut right_children: Vec<i64> = vec![-1; n];
    let mut parents: Vec<i64> = vec![2_147_483_647; n];
    let mut conditions: Vec<f32> = vec![0.0; n];
    let mut weights: Vec<f32> = vec![0.0; n];
    let mut split_indices: Vec<i64> = vec![0; n];
    let mut default_left: Vec<u8> = vec![0; n];
    for i in 0..n {
        if feature[i] < 0 {
            conditions[i] = tree.value()[i] as f32;
            weights[i] = tree.value()[i] as f32;
            continue;
        }
        let (l, r): (usize, usize) = (i + 1, right[i] as usize);
        (left_children[i], right_children[i]) = (l as i64, r as i64);
        (parents[l], parents[r]) = (i as i64, i as i64);
        // A split that sends every value left (threshold f64::MAX) would be "below +inf",
        // which JSON can't hold: below f32::MAX routes every value the same way.
        conditions[i] = (threshold[i].next_up() as f32).clamp(f32::MIN, f32::MAX);
        split_indices[i] = i64::from(feature[i]);
        default_left[i] = missing_left[i];
    }
    json!({
        "base_weights": weights,
        "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
        "default_left": default_left,
        "id": id,
        "left_children": left_children,
        "loss_changes": vec![0.0; n],
        "parents": parents,
        "right_children": right_children,
        "split_conditions": conditions,
        "split_indices": split_indices,
        "split_type": vec![0; n],
        "sum_hessian": tree.n_node_samples().iter().map(|c| f64::from(*c)).collect::<Vec<f64>>(),
        "tree_param": {"num_deleted": "0", "num_feature": features.to_string(), "num_nodes": n.to_string(), "size_leaf_vector": "1"},
    })
}

/// What LightGBM needs to know of a fitted model.
pub(crate) struct LgbmModel<'a> {
    pub trees: &'a [Tree],
    pub baseline: &'a [f64],
    pub names: &'a [String],
    /// LightGBM's objective line: `regression`, `binary sigmoid:1`, `multiclass num_class:3`, ...
    pub objective: String,
}

/// LightGBM's text model (version v4). LightGBM keeps no separate start: it is added to the
/// leaves of each class's first tree, as LightGBM itself does with `boost_from_average`.
pub(crate) fn lightgbm_text(m: &LgbmModel, path: &Path) -> Result<(), ModelError> {
    numbers_only(m.trees, "LightGBM text")?;
    let k: usize = m.baseline.len();
    let mut out: String = String::new();
    let names: Vec<String> = m.names.iter().map(|n| n.replace(' ', "_")).collect();
    let _ = write!(
        out,
        "tree\nversion=v4\nnum_class={k}\nnum_tree_per_iteration={k}\nlabel_index=0\nmax_feature_idx={}\nobjective={}\nfeature_names={}\nfeature_infos={}\n\n",
        m.names.len().saturating_sub(1),
        m.objective,
        names.join(" "),
        vec!["none"; m.names.len()].join(" "),
    );
    for (t, tree) in m.trees.iter().enumerate() {
        let start: f64 = if t < k { m.baseline[t % k] } else { 0.0 };
        lgbm_tree(&mut out, t, tree, start);
    }
    out.push_str("end of trees\n\npandas_categorical:null\n");
    fs::write(path, out)?;
    print_saved(path);
    Ok(())
}

/// One tree in LightGBM's layout: split nodes numbered 0.. and leaves numbered 0.., both in
/// rusmlai's (depth-first) order, a child written as its split number or as `~leaf`. A
/// missing value goes where rusmlai sends it (`decision_type` 8: NaN is missing, + 2: left).
fn lgbm_tree(out: &mut String, t: usize, tree: &Tree, start: f64) {
    let n: usize = tree.node_count();
    let (feature, threshold, right, missing_left) = (tree.feature(), tree.threshold(), tree.children_right(), tree.missing_go_to_left());
    let mut place: Vec<i64> = vec![0; n];
    let (mut splits, mut leaves): (i64, i64) = (0, 0);
    for i in 0..n {
        if feature[i] < 0 {
            place[i] = !leaves;
            leaves += 1;
        } else {
            place[i] = splits;
            splits += 1;
        }
    }
    let join = |v: Vec<String>| v.join(" ");
    let split_nodes: Vec<usize> = (0..n).filter(|i| feature[*i] >= 0).collect();
    let leaf_nodes: Vec<usize> = (0..n).filter(|i| feature[*i] < 0).collect();
    let counts = |nodes: &[usize]| join(nodes.iter().map(|i| tree.n_node_samples()[*i].to_string()).collect());
    if split_nodes.is_empty() {
        // A tree of one leaf: LightGBM writes it with num_leaves=1 and its value.
        let _ = write!(out, "Tree={t}\nnum_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\ndecision_type=\nleft_child=\nright_child=\nleaf_value={:?}\nleaf_weight=\nleaf_count=\ninternal_value=\ninternal_weight=\ninternal_count=\nis_linear=0\nshrinkage=1\n\n\n", tree.value()[0] + start);
        return;
    }
    let _ = write!(
        out,
        "Tree={t}\nnum_leaves={}\nnum_cat=0\nsplit_feature={}\nsplit_gain={}\nthreshold={}\ndecision_type={}\nleft_child={}\nright_child={}\nleaf_value={}\nleaf_weight={}\nleaf_count={}\ninternal_value={}\ninternal_weight={}\ninternal_count={}\nis_linear=0\nshrinkage=1\n\n\n",
        leaf_nodes.len(),
        join(split_nodes.iter().map(|i| feature[*i].to_string()).collect()),
        join(split_nodes.iter().map(|_| "0".to_string()).collect()),
        join(split_nodes.iter().map(|i| format!("{:?}", threshold[*i])).collect()),
        join(split_nodes.iter().map(|i| (8 + 2 * u32::from(missing_left[*i])).to_string()).collect()),
        join(split_nodes.iter().map(|i| place[i + 1].to_string()).collect()),
        join(split_nodes.iter().map(|i| place[right[*i] as usize].to_string()).collect()),
        join(leaf_nodes.iter().map(|i| format!("{:?}", tree.value()[*i] + start)).collect()),
        counts(&leaf_nodes),
        counts(&leaf_nodes),
        join(split_nodes.iter().map(|_| "0".to_string()).collect()),
        counts(&split_nodes),
        counts(&split_nodes),
    );
}

/// What CatBoost needs to know of a fitted model.
pub(crate) struct CatBoostModel<'a> {
    /// One tree per class and round, the classes' trees of a round alike but for values.
    pub trees: &'a [Tree],
    pub baseline: &'a [f64],
    pub names: &'a [String],
    /// `"RMSE"`, `"Logloss"` or `"MultiClass"`.
    pub loss: &'a str,
    pub classes: &'a [String],
    /// `nan_mode="Max"`: a missing value counts as above every border.
    pub nan_max: bool,
}

/// CatBoost's JSON model, which `CatBoostRegressor().load_model(path, format="json")` (or
/// `CatBoostClassifier`) reads. Each round's trees, alike in their splits, become one
/// symmetric tree with a value per class in each leaf.
pub(crate) fn catboost_json(m: &CatBoostModel, path: &Path) -> Result<(), ModelError> {
    numbers_only(m.trees, "CatBoost JSON")?;
    let k: usize = m.baseline.len();
    // Every round's levels: (feature, border), from the root down the left side.
    let levels = |tree: &Tree| -> Vec<(usize, f32)> {
        let (feature, threshold) = (tree.feature(), tree.threshold());
        let mut out: Vec<(usize, f32)> = Vec::new();
        let mut at: usize = 0;
        while feature[at] >= 0 {
            // A threshold of ±inf parts the missing values from the rest: a border beyond
            // every value does the same.
            out.push((feature[at] as usize, (threshold[at] as f32).clamp(f32::MIN, f32::MAX)));
            at += 1;
        }
        out
    };
    let mut borders: Vec<Vec<f32>> = vec![Vec::new(); m.names.len()];
    let rounds: Vec<Vec<(usize, f32)>> = m.trees.chunks(k).map(|round| levels(&round[0])).collect();
    for (f, border) in rounds.iter().flatten() {
        if !borders[*f].contains(border) {
            borders[*f].push(*border);
        }
    }
    borders.iter_mut().for_each(|b| b.sort_by(f32::total_cmp));
    let offsets: Vec<usize> = borders.iter().scan(0, |at, b| { let start = *at; *at += b.len(); Some(start) }).collect();
    let trees: Vec<Value> = m.trees.chunks(k).zip(&rounds).map(|(round, splits)| {
        let depth: usize = splits.len();
        let mut values: Vec<f64> = vec![0.0; (1 << depth) * k];
        let mut weights: Vec<u32> = vec![0; 1 << depth];
        for (c, tree) in round.iter().enumerate() {
            // Depth first, left first: a leaf's CatBoost index has bit `level` set when the
            // row went right at that level.
            let mut stack: Vec<(usize, usize, usize)> = vec![(0, 0, 0)];
            while let Some((node, level, index)) = stack.pop() {
                if tree.feature()[node] < 0 {
                    values[index * k + c] = tree.value()[node];
                    weights[index] = tree.n_node_samples()[node];
                    continue;
                }
                stack.push((tree.children_right()[node] as usize, level + 1, index | 1 << level));
                stack.push((node + 1, level + 1, index));
            }
        }
        let splits: Vec<Value> = splits.iter().map(|(f, border)| {
            let place: usize = borders[*f].iter().position(|b| b == border).unwrap_or(0);
            json!({"border": border, "float_feature_index": f, "split_index": offsets[*f] + place, "split_type": "FloatFeature"})
        }).collect();
        json!({"leaf_values": values, "leaf_weights": weights, "splits": splits})
    }).collect();
    let features: Vec<Value> = (0..m.names.len()).map(|f| json!({
        "borders": borders[f], "feature_id": m.names[f], "feature_index": f, "flat_feature_index": f,
        "has_nans": true, "nan_value_treatment": if m.nan_max { "AsTrue" } else { "AsFalse" },
    })).collect();
    let mut info: Value = json!({"params": {"loss_function": {"params": {}, "type": m.loss}}});
    if !m.classes.is_empty() {
        info["class_params"] = json!({
            "class_label_type": "String", "class_names": m.classes,
            "class_to_label": (0..m.classes.len()).collect::<Vec<usize>>(), "classes_count": 0,
        });
    }
    let model: Value = json!({
        "features_info": {"float_features": features},
        "model_info": info,
        "oblivious_trees": trees,
        "scale_and_bias": [1, m.baseline],
    });
    fs::write(path, serde_json::to_vec(&model)?)?;
    print_saved(path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use polars::prelude::*;

    use crate::{
        dataset::Dataset,
        models::{
            CatBoostClassifier, CatBoostRegressor, Classifier, LGBMClassifier, LGBMRegressor, Model, Regressor, XGBClassifier,
            XGBRegressor,
        },
    };

    /// 300 rows: three features (one with missing values), a number, a positive number and
    /// classes.
    fn data() -> Dataset {
        let n: usize = 300;
        let a: Vec<Option<f64>> = (0..n).map(|i| if i % 11 == 0 { None } else { Some((i * 37 % 101) as f64) }).collect();
        let b: Vec<f64> = (0..n).map(|i| (i * 53 % 17) as f64 / 3.0).collect();
        let c: Vec<f64> = (0..n).map(|i| ((i * 7) as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| a[i].unwrap_or(50.0) * 0.2 + b[i] - 3.0 * c[i]).collect();
        let pos: Vec<f64> = y.iter().map(|v| v.abs() + 1.0).collect();
        let class: Vec<i64> = y.iter().map(|v| (*v / 8.0).floor().rem_euclid(3.0) as i64).collect();
        let hit: Vec<i64> = class.iter().map(|c| i64::from(*c == 1)).collect();
        Dataset::new(df!("a" => a, "b" => b, "c" => c, "y" => y, "pos" => pos, "class" => class, "hit" => hit).unwrap(), "")
    }

    /// Writes `x` and rusmlai's predictions (one column per output) next to the model file,
    /// for the Python check.
    fn expect(dir: &std::path::Path, name: &str, x: &Dataset, outputs: Vec<Vec<f64>>) {
        let mut frame: DataFrame = x.dataframe().clone();
        for (k, values) in outputs.into_iter().enumerate() {
            frame.with_column(Column::new(format!("__want_{k}").into(), values)).unwrap();
        }
        CsvWriter::new(std::fs::File::create(dir.join(format!("{name}.csv"))).unwrap()).finish(&mut frame).unwrap();
    }

    const CHECK: &str = r#"
import glob, os, sys
import numpy as np, pandas as pd
d = sys.argv[1]
bad = []
for csv in sorted(glob.glob(os.path.join(d, "*.csv"))):
    name = os.path.basename(csv)[:-4]
    t = pd.read_csv(csv)
    cols = [c for c in t if c.startswith("__want_")]
    want = t[cols].to_numpy()
    t = t.drop(columns=cols)
    if name.startswith("cb-reg"):
        import catboost as cb
        m = cb.CatBoostRegressor(); m.load_model(os.path.join(d, name + ".json"), format="json")
        got = m.predict(t)
    elif name.startswith("cb"):
        import catboost as cb
        m = cb.CatBoostClassifier(); m.load_model(os.path.join(d, name + ".json"), format="json")
        got = m.predict_proba(t)
        got = got[:, 1] if got.shape[1] == 2 else got
    elif name.startswith("xgb"):
        import xgboost as xgb
        b = xgb.Booster(model_file=os.path.join(d, name + ".json"))
        got = b.predict(xgb.DMatrix(t, missing=np.nan))
    else:
        import lightgbm as lgb
        b = lgb.Booster(model_file=os.path.join(d, name + ".txt"))
        got = b.predict(t)
    got = np.asarray(got).reshape(len(t), -1)
    err = np.abs(got - want).max() / max(1.0, np.abs(want).max())
    print(f"{name}: largest gap {err:.2e}")
    if err > 1e-5:
        bad.append(name)
sys.exit(1 if bad else 0)
"#;

    #[test]
    fn python_loads_the_files_and_predicts_the_same() {
        let has = |module: &str| Command::new("python3").args(["-c", &format!("import {module}")]).output().is_ok_and(|o| o.status.success());
        if !has("xgboost") || !has("lightgbm") || !has("catboost") || !has("pandas") {
            eprintln!("skipped: python3 with xgboost, lightgbm, catboost and pandas is not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("rusmlai-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ds: Dataset = data();
        let x = |target: &str| ds.select(&["a", "b", "c", target]);
        let features: Dataset = ds.select(&["a", "b", "c"]);
        for (name, objective, target) in [
            ("xgb-squared", "reg:squarederror", "y"),
            ("xgb-poisson", "count:poisson", "pos"),
            ("xgb-gamma", "reg:gamma", "pos"),
        ] {
            let mut m = XGBRegressor::new().objective(objective).n_estimators(20);
            m.try_train(&x(target), target).unwrap();
            m.try_save_model(dir.join(format!("{name}.json"))).unwrap();
            expect(&dir, name, &features, vec![m.predict(&features).f64().unwrap().into_no_null_iter().collect()]);
        }
        for (name, target) in [("xgb-binary", "hit"), ("xgb-classes", "class")] {
            let mut m = XGBClassifier::new().n_estimators(20);
            m.try_train(&x(target), target).unwrap();
            m.try_save_model(dir.join(format!("{name}.json"))).unwrap();
            let proba: Dataset = m.predict_proba(&features);
            let want: Vec<Vec<f64>> = match m.classes_.len() {
                2 => vec![proba.to_list(&m.classes_[1])],
                _ => m.classes_.iter().map(|c| proba.to_list(c)).collect(),
            };
            expect(&dir, name, &features, want);
        }
        for (name, objective, target) in [
            ("lgbm-regression", "regression", "y"),
            ("lgbm-l1", "regression_l1", "y"),
            ("lgbm-poisson", "poisson", "pos"),
            ("lgbm-tweedie", "tweedie", "pos"),
        ] {
            let mut m = LGBMRegressor::new().objective(objective).n_estimators(20).min_child_samples(5);
            m.try_train(&x(target), target).unwrap();
            m.try_save_model(dir.join(format!("{name}.txt"))).unwrap();
            expect(&dir, name, &features, vec![m.predict(&features).f64().unwrap().into_no_null_iter().collect()]);
        }
        let mut m = LGBMClassifier::new().n_estimators(20).min_child_samples(5);
        m.try_train(&x("hit"), "hit").unwrap();
        m.try_save_model(dir.join("lgbm-binary.txt")).unwrap();
        expect(&dir, "lgbm-binary", &features, vec![m.predict_proba(&features).to_list("1")]);
        let mut m = LGBMClassifier::new().n_estimators(20).min_child_samples(5);
        m.try_train(&x("class"), "class").unwrap();
        m.try_save_model(dir.join("lgbm-classes.txt")).unwrap();
        let proba: Dataset = m.predict_proba(&features);
        expect(&dir, "lgbm-classes", &features, m.classes_.iter().map(|c| proba.to_list(c)).collect());

        for (name, nan_mode) in [("cb-reg-min", "Min"), ("cb-reg-max", "Max")] {
            let mut m = CatBoostRegressor::new().iterations(30).nan_mode(nan_mode);
            m.try_train(&x("y"), "y").unwrap();
            m.try_save_model(dir.join(format!("{name}.json"))).unwrap();
            expect(&dir, name, &features, vec![m.predict(&features).f64().unwrap().into_no_null_iter().collect()]);
        }
        for (name, target) in [("cb-binary", "hit"), ("cb-classes", "class")] {
            let mut m = CatBoostClassifier::new().iterations(30);
            m.try_train(&x(target), target).unwrap();
            m.try_save_model(dir.join(format!("{name}.json"))).unwrap();
            let proba: Dataset = m.predict_proba(&features);
            let want: Vec<Vec<f64>> = match m.classes_.len() {
                2 => vec![proba.to_list(&m.classes_[1])],
                _ => m.classes_.iter().map(|c| proba.to_list(c)).collect(),
            };
            expect(&dir, name, &features, want);
        }

        let out = Command::new("python3").args(["-c", CHECK, dir.to_str().unwrap()]).output().unwrap();
        let report: String = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
        if out.status.success() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        assert!(out.status.success(), "{report} (files kept in {})", dir.display());
        eprintln!("{report}");
    }
}
