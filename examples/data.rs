use rusmlai::prelude::*;

fn main() {
    let dataset: Dataset = read_csv("/home/kartikag234/Documents/rusmlai/tests/fixtures/train.csv");
    dataset.metadata(&[]);
    dataset.head(5);
    dataset.describe();
    dataset.info();

    // Each chart saves itself to tests/fixtures/metadata/plots/{title}.svg.
    dataset.count("course", "");
    dataset.count("study_method", "gender");
    dataset.bar("gender", "exam_score", "course", "mean");
    dataset.bar("age", "exam_score", "", "median");

    // Options chain like Python keyword arguments.
    dataset
        .bar("sleep_quality", "exam_score", "", "mean")
        .title("Mean score by sleep quality")
        .order(&["poor", "average", "good"])
        .color("#E8C547")
        .ylim(55.0, 70.0);
    dataset
        .count("facility_rating", "exam_difficulty")
        .title("Exam difficulty by facility rating")
        .theme("light")
        .palette("colorblind")
        .hue_order(&["easy", "moderate", "hard"])
        .show();

    // An explicit path replaces the auto-save.
    dataset
        .bar("internet_access", "study_hours", "", "mean")
        .save("tests/fixtures/metadata/plots/internet_vs_study_hours.svg");

    dataset.pie("course");
    dataset.pie("study_method").palette("colorblind").top(3);
    dataset.describe_bar("", "mean");
    dataset.describe_bar("exam_score", "");
    dataset.describe_bar("study_hours", "count");

    dataset.hist("exam_score", "", 30);
    dataset.hist("study_hours", "gender", 12);
    dataset.boxplot("", "exam_score", "");
    dataset.boxplot("course", "exam_score", "");
    dataset.boxplot("sleep_quality", "exam_score", "gender");
    dataset.stripplot("course", "exam_score", "");
    dataset.stripplot("sleep_quality", "exam_score", "internet_access");
}
