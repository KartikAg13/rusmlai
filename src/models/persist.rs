//! `Model::save` and `Model::load`: a fitted model (a `Pipeline` with its preprocessing) in
//! one file, read back as the same model on any machine.
//!
//! The file is the magic `RUSMLAI\0`, a little-endian `u32` format version, a `u64` length,
//! that many bytes of JSON (`{"rusmlai", "model", "params", "learned", "target", "prep"}`),
//! then the bytes `Model::write_weights` wrote (a tree model's nodes; nothing for most
//! models). It holds numbers and names only, never code, so loading a file runs nothing from
//! it, unlike a pickle; a file that does not fit the model is refused.

use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

use serde_json::{Value, json};

use crate::{
    metadata::utils::print_saved,
    models::{Model, ModelError},
};

const MAGIC: &[u8; 8] = b"RUSMLAI\0";
const FORMAT: u32 = 1;

/// The error of a file that is not a saved model of this kind.
fn bad(path: &Path, reason: impl Into<String>) -> ModelError {
    ModelError::RunMismatch { run: path.display().to_string(), reason: reason.into() }
}

pub(crate) fn try_save<M: Model + ?Sized>(model: &M, path: &Path) -> Result<(), ModelError> {
    let target: &str = model.target().ok_or(ModelError::NotFitted { model: model.name() })?;
    let head: Value = json!({
        "rusmlai": env!("CARGO_PKG_VERSION"),
        "model": model.name(),
        "params": model.params(),
        "learned": model.learned(),
        "target": target,
        "prep": model.prep_state()?,
    });
    let head: Vec<u8> = serde_json::to_vec(&head)?;
    let mut weights: Vec<u8> = Vec::new();
    model.write_weights(&mut weights)?;
    let mut file = fs::File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&FORMAT.to_le_bytes())?;
    file.write_all(&(head.len() as u64).to_le_bytes())?;
    file.write_all(&head)?;
    file.write_all(&weights)?;
    print_saved(path);
    Ok(())
}

pub(crate) fn try_load<M: Model + Default>(path: &Path) -> Result<M, ModelError> {
    let mut bytes: Vec<u8> = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 20 || &bytes[..8] != MAGIC {
        return Err(bad(path, "it is not a model saved by rusmlai's save"));
    }
    let format: u32 = u32::from_le_bytes(bytes[8..12].try_into().unwrap_or_default());
    if format != FORMAT {
        return Err(bad(path, format!("it has file format {format}; this rusmlai reads format {FORMAT}")));
    }
    let len: usize = u64::from_le_bytes(bytes[12..20].try_into().unwrap_or_default()) as usize;
    let Some(head) = bytes.get(20..20 + len) else {
        return Err(bad(path, "the file is cut short"));
    };
    let head: Value = serde_json::from_slice(head)?;
    let mut model: M = M::default();
    let saved: &str = head["model"].as_str().unwrap_or_default();
    if saved != model.name() {
        return Err(bad(path, format!("it holds a {saved}, not a {}", model.name())));
    }
    let target: &str = head["target"].as_str().unwrap_or_default();
    model.try_set_params(head["params"].clone())?;
    if !head["prep"].is_null() {
        model.try_restore_prep(head["prep"].clone())?;
    }
    model.try_load(head["learned"].clone(), &bytes[20 + len..], target)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use polars::df;

    use super::*;
    use crate::{
        dataset::Dataset,
        models::{
            AdaBoostClassifier, CatBoostClassifier, CatBoostRegressor, Classifier, DecisionTreeClassifier, DecisionTreeRegressor,
            DummyRegressor, ElasticNet, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
            HistGradientBoostingRegressor, LGBMClassifier, LGBMRegressor, Lasso, LinearRegression, LogisticRegression,
            LogisticRegressionCV,
            RandomForestClassifier, RandomForestRegressor, Regressor, Ridge, RidgeCV, RidgeClassifier, XGBClassifier,
            XGBRegressor,
        },
        pipeline::Pipeline,
        preprocess::{ColumnTransformer, OneHotEncoder, SimpleImputer, StandardScaler, TargetEncoder},
        steps,
    };

    fn data() -> Dataset {
        let n: usize = 120;
        let x: Vec<f64> = (0..n).map(|i| (i * 37 % 101) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i * 53 % 17) as f64).collect();
        let y: Vec<f64> = x.iter().zip(&z).map(|(x, z)| x * 0.3 + z - 5.0).collect();
        let class: Vec<&str> = y.iter().map(|v| if *v > 12.0 { "hi" } else { "lo" }).collect();
        Dataset::new(df!("x" => x, "z" => z, "y" => y, "class" => class).unwrap(), "")
    }

    fn path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rusmlai-test-{}-{name}.model", std::process::id()))
    }

    fn regressor<M: Regressor + Default>(mut model: M, ds: &Dataset) {
        let x: Dataset = ds.drop(&["class"]);
        model.try_train(&x, "y").unwrap();
        let file = path(model.name());
        model.try_save(&file).unwrap();
        let loaded: M = M::try_load_from(&file).unwrap();
        assert_eq!(loaded.predict(&x), model.predict(&x), "{}", model.name());
        assert_eq!(loaded.params(), model.params(), "{}", model.name());
        std::fs::remove_file(file).unwrap();
    }

    fn classifier<M: Classifier + Default>(mut model: M, ds: &Dataset) {
        let x: Dataset = ds.drop(&["y"]);
        model.try_train(&x, "class").unwrap();
        let file = path(model.name());
        model.try_save(&file).unwrap();
        let loaded: M = M::try_load_from(&file).unwrap();
        assert_eq!(loaded.predict_proba(&x).dataframe(), model.predict_proba(&x).dataframe(), "{}", model.name());
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn every_model_comes_back_as_it_was() {
        let ds: Dataset = data();
        regressor(LinearRegression::new(), &ds);
        regressor(Ridge::new().alpha(3.0), &ds);
        regressor(Lasso::new(), &ds);
        regressor(ElasticNet::new(), &ds);
        regressor(RidgeCV::new(), &ds);
        regressor(DummyRegressor::new(), &ds);
        regressor(DecisionTreeRegressor::new().max_depth(4), &ds);
        regressor(RandomForestRegressor::new().n_estimators(5), &ds);
        regressor(ExtraTreesRegressor::new().n_estimators(5), &ds);
        regressor(HistGradientBoostingRegressor::new().max_iter(10), &ds);
        regressor(XGBRegressor::new().n_estimators(10), &ds);
        regressor(LGBMRegressor::new().n_estimators(10).min_child_samples(5), &ds);
        regressor(CatBoostRegressor::new().iterations(10), &ds);
        classifier(LogisticRegression::new(), &ds);
        classifier(LogisticRegressionCV::new().Cs(3), &ds);
        classifier(DecisionTreeClassifier::new(), &ds);
        classifier(RandomForestClassifier::new().n_estimators(5), &ds);
        classifier(ExtraTreesClassifier::new().n_estimators(5), &ds);
        classifier(HistGradientBoostingClassifier::new().max_iter(10), &ds);
        classifier(XGBClassifier::new().n_estimators(10), &ds);
        classifier(LGBMClassifier::new().n_estimators(10).min_child_samples(5), &ds);
        classifier(CatBoostClassifier::new().iterations(10), &ds);
        classifier(AdaBoostClassifier::new(), &ds);
        // No probabilities: its scores and settings come back.
        let x: Dataset = ds.drop(&["y"]);
        let mut ridge: RidgeClassifier = RidgeClassifier::new().class_weight("balanced");
        ridge.try_train(&x, "class").unwrap();
        let file = path("RidgeClassifier");
        ridge.try_save(&file).unwrap();
        let loaded: RidgeClassifier = RidgeClassifier::try_load_from(&file).unwrap();
        assert_eq!(loaded.decision_function(&x).dataframe(), ridge.decision_function(&x).dataframe());
        assert_eq!((loaded.params(), loaded.predict(&x)), (ridge.params(), ridge.predict(&x)));
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn a_pipeline_comes_back_with_its_preprocessing() {
        let n: usize = 60;
        let city: Vec<Option<&str>> = (0..n).map(|i| if i % 9 == 0 { None } else { Some(["a", "b", "c"][i % 3]) }).collect();
        let hours: Vec<Option<f64>> = (0..n).map(|i| if i % 7 == 0 { None } else { Some((i * 13 % 17) as f64) }).collect();
        let y: Vec<f64> = (0..n).map(|i| (i * 13 % 17) as f64 * 0.5 + (i % 3) as f64).collect();
        let ds = Dataset::new(df!("city" => city, "hours" => hours, "shop" => (0..n).map(|i| ["x", "y"][i % 2]).collect::<Vec<_>>(), "y" => y).unwrap(), "");
        let prep = ColumnTransformer::new()
            .add("num", &["hours"], steps![SimpleImputer::new().strategy("median"), StandardScaler::new()])
            .add("cat", &["city"], steps![SimpleImputer::new().strategy("most_frequent"), OneHotEncoder::new()])
            .add("te", &["shop"], steps![TargetEncoder::new().target_type("continuous")]);
        let mut pipe = Pipeline::new(prep, LinearRegression::new());
        pipe.try_train(&ds, "y").unwrap();
        let file = path("pipeline");
        pipe.try_save(&file).unwrap();
        let loaded: Pipeline<LinearRegression> = Pipeline::try_load_from(&file).unwrap();
        assert_eq!(loaded.predict(&ds), pipe.predict(&ds));
        assert_eq!(loaded.prep.get_feature_names_out(), pipe.prep.get_feature_names_out());
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn a_wrong_or_broken_file_is_refused() {
        let ds: Dataset = data().drop(&["class"]);
        let mut model = LinearRegression::new();
        model.try_train(&ds, "y").unwrap();
        let file = path("refused");
        model.try_save(&file).unwrap();
        let other = Ridge::try_load_from(&file).unwrap_err().to_string();
        assert!(other.contains("holds a LinearRegression, not a Ridge"), "{other}");
        let bytes: Vec<u8> = std::fs::read(&file).unwrap();
        std::fs::write(&file, &bytes[..30]).unwrap();
        assert!(LinearRegression::try_load_from(&file).unwrap_err().to_string().contains("cut short"));
        std::fs::write(&file, b"not a model at all").unwrap();
        assert!(LinearRegression::try_load_from(&file).unwrap_err().to_string().contains("not a model saved"));
        std::fs::remove_file(file).unwrap();
        assert!(LinearRegression::new().try_save(path("unfitted")).unwrap_err().to_string().contains("not fitted"));
    }
}
