use std::sync::Arc;

use polars::frame::DataFrame;

use crate::data::dataset::{
    Dataset,
    metadata::{dto::MetadataDto, info::ColumnInfo},
};

pub mod dto;
pub mod info;
pub mod stats;

pub struct Metadata {
    pub height: usize,
    pub width: usize,
    pub columns: Vec<Arc<ColumnInfo>>,
}

impl Metadata {
    pub fn new(dataframe: &DataFrame) -> Self {
        let width: usize = dataframe.width();
        let mut data: Vec<Arc<ColumnInfo>> = Vec::with_capacity(width);
        for column in dataframe.columns() {
            data.push(Arc::new(ColumnInfo::new(&column)));
        }
        Self {
            height: dataframe.height(),
            width: width,
            columns: data,
        }
    }

    pub fn after_transform(
        &self,
        new_dataset: &Dataset,
        affected_columns: Vec<String>,
        row_count_changed: bool,
    ) -> Metadata {
        let columns = new_dataset
            .df
            .get_column_names()
            .iter()
            .map(|name| {
                let name_s = name.to_string();
                let is_affected = row_count_changed || affected_columns.contains(&name_s);

                if !is_affected {
                    if let Some(existing) = self.columns.iter().find(|c| c.name == name_s) {
                        return Arc::clone(existing);
                    }
                }
                let col = new_dataset.df.column(name).unwrap();
                Arc::new(ColumnInfo::new(col))
            })
            .collect();

        Metadata {
            height: new_dataset.df.height(),
            width: new_dataset.df.width(),
            columns,
        }
    }
}

impl From<MetadataDto> for Metadata {
    fn from(dto: MetadataDto) -> Self {
        Self {
            height: dto.height,
            width: dto.width,
            columns: dto
                .columns
                .into_iter()
                .map(|c| Arc::new(c.into()))
                .collect(),
        }
    }
}
