//! Every reader gives the table of `students.csv`: TSV, Parquet, Excel and SQLite here, and
//! PostgreSQL / MySQL when `RUSMLAI_TEST_POSTGRES` / `RUSMLAI_TEST_MYSQL` hold the URL of a
//! server whose `students` table holds that CSV (`cargo test --test read -- --ignored`).

use rusmlai::prelude::*;

const DATA: &str = "tests/fixtures/data";

/// The numbers of `column` as text, so NaN (a null) compares equal to NaN.
fn values(ds: &Dataset, column: &str) -> Vec<String> {
    ds.to_list(column).iter().map(|v| format!("{v}")).collect()
}

fn same_as_the_csv(ds: &Dataset, source: &str) {
    let csv: Dataset = read_csv(format!("{DATA}/students.csv"));
    assert_eq!(ds.shape(), csv.shape(), "{source}");
    assert_eq!(ds.columns(), csv.columns(), "{source}");
    for column in ["id", "age", "cgpa", "passed"] {
        assert_eq!(
            values(ds, column),
            values(&csv, column),
            "{source}: {column}"
        );
    }
    let names = |d: &Dataset| d.try_categorical_stats("name").unwrap().categories.clone();
    assert_eq!(names(ds), names(&csv), "{source}");
}

#[test]
fn every_file_format_reads_the_same_table() {
    same_as_the_csv(&read_csv(format!("{DATA}/students.tsv")), "tsv");
    same_as_the_csv(&read_parquet(format!("{DATA}/students.parquet")), "parquet");
    same_as_the_csv(&read_excel(format!("{DATA}/students.xlsx"), ""), "xlsx");
    same_as_the_csv(&read_excel(format!("{DATA}/students.xls"), ""), "xls");
    let db: String = format!("{DATA}/students.db");
    same_as_the_csv(&read_sql("SELECT * FROM students", &db), "sqlite");
}

#[test]
fn a_server_password_never_reaches_an_error_message() {
    for url in [
        "postgresql://me:hunter2@127.0.0.1:1/school",
        "mysql://me:hunter2@127.0.0.1:1/school",
    ] {
        let err: String = try_read_sql("SELECT 1", url).unwrap_err().to_string();
        assert!(
            err.contains("127.0.0.1:1") && !err.contains("hunter2"),
            "{err}"
        );
    }
}

/// The URL in `var`, or `None` with a note that the test was skipped.
fn server(var: &str) -> Option<String> {
    let url: Option<String> = std::env::var(var).ok();
    if url.is_none() {
        eprintln!("skipped: set {var} to a server URL");
    }
    url
}

const COLUMNS: &str = "SELECT id, name, age, cgpa, batch, passed FROM students ORDER BY id";

#[test]
#[ignore = "needs a PostgreSQL server in RUSMLAI_TEST_POSTGRES"]
fn postgres_reads_the_same_table() {
    let Some(url) = server("RUSMLAI_TEST_POSTGRES") else {
        return;
    };
    same_as_the_csv(&read_sql(COLUMNS, &url), "postgres");
    // `numeric` has no f64 reading of its own: the message says how to cast it.
    let err: String = try_read_sql("SELECT fee FROM students", &url)
        .unwrap_err()
        .to_string();
    assert!(err.contains("fee::float8"), "{err}");
}

#[test]
#[ignore = "needs a MySQL server in RUSMLAI_TEST_MYSQL"]
fn mysql_reads_the_same_table() {
    let Some(url) = server("RUSMLAI_TEST_MYSQL") else {
        return;
    };
    same_as_the_csv(&read_sql(COLUMNS, &url), "mysql");
}
