// use std::{fs, path::PathBuf};

// use rusmlai::data::dataset::{Dataset, read_csv};

fn main() {
    //     let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //     let csv = root.join("tests/fixtures/kaggle_playground-series-s6e1/train.csv");
    //     let out = root.join("tests/fixtures/kaggle_playground-series-s6e1/plot");
    //     fs::create_dir_all(&out).unwrap();
    //     let ds: Dataset = read_csv(&csv);

    //     write(
    //         &out,
    //         "bar_gender_count.svg",
    //         ds.bar("gender", "", "", "mean"),
    //     );
    //     write(
    //         &out,
    //         "bar_gender_exam_score.svg",
    //         ds.bar("gender", "exam_score", "", "mean"),
    //     );
    //     write(
    //         &out,
    //         "bar_gender_course.svg",
    //         ds.bar("gender", "exam_score", "course", "mean"),
    //     );
    //     write(
    //         &out,
    //         "count_gender_course.svg",
    //         ds.count("gender", "course"),
    //     );
    //     write(&out, "describe_all_mean.svg", ds.describe_bar("", "mean"));
    //     write(
    //         &out,
    //         "describe_exam_score.svg",
    //         ds.describe_bar("exam_score", ""),
    //     );
    //     write(&out, "pie_course.svg", ds.pie("course"));
    //     write(&out, "hist_exam_score.svg", ds.hist("exam_score", "", 20));
    //     write(
    //         &out,
    //         "hist_exam_score_gender.svg",
    //         ds.hist("exam_score", "gender", 20),
    //     );
    //     write(&out, "box_exam_score.svg", ds.boxplot("", "exam_score", ""));
    //     write(
    //         &out,
    //         "box_gender_course.svg",
    //         ds.boxplot("gender", "exam_score", "course"),
    //     );
    //     write(
    //         &out,
    //         "strip_gender.svg",
    //         ds.stripplot("gender", "exam_score", ""),
    //     );
    //     write(
    //         &out,
    //         "scatter_study_hours.svg",
    //         ds.scatter(&["study_hours"], "exam_score", ""),
    //     );
    //     write(
    //         &out,
    //         "scatter_two_x.svg",
    //         ds.scatter(&["study_hours", "sleep_hours"], "exam_score", "gender"),
    //     );
    //     write(&out, "line_age.svg", ds.line("age", &["exam_score"], ""));
    //     write(
    //         &out,
    //         "line_two_y.svg",
    //         ds.line("age", &["study_hours", "sleep_hours"], ""),
    //     );
    //     write(
    //         &out,
    //         "regplot_gender.svg",
    //         ds.regplot("study_hours", "exam_score", "gender"),
    //     );
    //     write(
    //         &out,
    //         "joint_scatter.svg",
    //         ds.jointplot("study_hours", "exam_score", "", "scatter"),
    //     );
    //     write(&out, "pairplot.svg", ds.pairplot(&[], "gender"));
    //     write(
    //         &out,
    //         "heatmap_mean.svg",
    //         ds.heatmap("gender", "course", "exam_score", "mean"),
    //     );
    //     write(
    //         &out,
    //         "heatmap_count.svg",
    //         ds.heatmap("gender", "course", "", "count"),
    //     );
    //     write(&out, "corrplot.svg", ds.corrplot());

    //     println!("wrote plots to {}", out.display());
    // }

    // fn write(dir: &PathBuf, name: &str, svg: String) {
    //     let path = dir.join(name);
    //     fs::write(&path, svg).unwrap();
    //     println!("  {name}");
}
