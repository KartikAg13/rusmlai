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
        .hue_order(&["easy", "moderate", "hard"]);

    // An explicit path replaces the auto-save.
    dataset
        .bar("internet_access", "study_hours", "", "mean")
        .save("tests/fixtures/metadata/plots/internet_vs_study_hours.svg");
}
