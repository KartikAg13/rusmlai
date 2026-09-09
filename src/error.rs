use thiserror::Error;
use thiserror_context::{Context, impl_context};

use crate::dataset::error::DatasetError;

#[derive(Debug, Error)]
pub enum RusMLAIErrorInner {
    #[error(transparent)]
    Data(#[from] DatasetError),
}

impl_context!(RusMLAIError(RusMLAIErrorInner));

#[macro_export]
macro_rules! fatal {
    ($err:expr, $func_name:expr) => {{
        eprintln!("\n[rusmlai error in {}]\n{}\n", $func_name, $err);
        panic!("execution stopped");
    }};
}
