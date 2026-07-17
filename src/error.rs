use thiserror::Error;
use thiserror_context::{Context, impl_context};

use crate::data::error::DataError;

#[derive(Debug, Error)]
pub enum RusMLAIErrorInner {
    #[error(transparent)]
    Data(#[from] DataError),
}

impl_context!(RusMLAIError(RusMLAIErrorInner));

#[macro_export]
macro_rules! fatal {
    ($err:expr, $func_name:expr) => {{
        eprintln!("ERROR: {}", format!("{:?}", $err));
        panic!("Function: {}", $func_name);
    }};
}
