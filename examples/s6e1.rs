use rusmlai::prelude::*;

fn main() {
    let dataset: Dataset =
        read_csv("/home/kartikag234/Documents/rusmlai/tests/fixtures/s6e1/train.csv");
    dataset.metadata(&[]);
    dataset.head(5);
    dataset.describe();
    dataset.info();

    // Each chart saves itself to tests/fixtures/s6e1/metadata/plots/{title}.png.

    // Who the students are: the default dark card, one rusmlai color per category.
    dataset.count("course", "").title("Students per course");
    dataset
        .count("study_method", "gender")
        .title("How each gender studies")
        .palette("bright");
    dataset
        .pie("study_method")
        .title("Study methods")
        .theme("light")
        .palette("pastel");
    dataset
        .pie("course")
        .title("The four biggest courses")
        .top(4)
        .colors(&["#F94144", "#F8961E", "#F9C74F", "#90BE6D", "#577590"]);

    // What moves the score.
    dataset
        .bar("sleep_quality", "exam_score", "", "mean")
        .title("Mean score by sleep quality")
        .order(&["poor", "average", "good"])
        .color("#E8C547")
        .ylim(55.0, 70.0)
        .hline(62.51, "overall mean");
    dataset
        .bar("course", "exam_score", "exam_difficulty", "mean")
        .title("Mean score by course and exam difficulty")
        .theme("light")
        .palette("colorblind")
        .hue_order(&["easy", "moderate", "hard"])
        .ylim(50.0, 70.0);
    dataset
        .count("facility_rating", "exam_difficulty")
        .title("Exam difficulty by facility rating")
        .order(&["low", "medium", "high"])
        .hue_order(&["easy", "moderate", "hard"])
        .color_map(&[
            ("easy", "#2A9D8F"),
            ("moderate", "#E9C46A"),
            ("hard", "#E76F51"),
        ]);
    dataset
        .bar("age", "exam_score", "", "median")
        .title("Median score by age")
        .color("#7EB6FF")
        .ylim(55.0, 66.0);
    dataset
        .describe_bar("exam_score", "")
        .title("exam_score at a glance")
        .color("#C792EA");

    // How scores are spread.
    dataset
        .hist("exam_score", "", 40)
        .title("Exam scores")
        .color("#FF9F1C");
    dataset
        .hist("study_hours", "internet_access", 16)
        .title("Study hours with and without internet")
        .theme("light")
        .color_map(&[("yes", "#3A86FF"), ("no", "#FF006E")]);
    dataset
        .boxplot("course", "exam_score", "")
        .title("Scores per course")
        .color_map(&[
            ("b.tech", "#F94144"),
            ("b.sc", "#F3722C"),
            ("b.com", "#F8961E"),
            ("bca", "#F9C74F"),
            ("bba", "#90BE6D"),
            ("ba", "#43AA8B"),
            ("diploma", "#577590"),
        ]);
    dataset
        .boxplot("sleep_quality", "exam_score", "gender")
        .title("Scores by sleep quality and gender")
        .order(&["poor", "average", "good"])
        .theme("light")
        .palette("muted");
    dataset
        .stripplot("exam_difficulty", "exam_score", "sleep_quality")
        .title("Every dot is a student")
        .order(&["easy", "moderate", "hard"])
        .hue_order(&["poor", "average", "good"])
        .colors(&["#EF476F", "#FFD166", "#06D6A0"]);

    // Two numbers at once.
    dataset
        .heatmap("course", "study_method", "exam_score", "mean")
        .title("Mean score by course and study method")
        .cmap("viridis");
    dataset
        .heatmap("sleep_quality", "facility_rating", "", "")
        .title("Students by sleep and facilities")
        .theme("light")
        .cmap("blues");
    dataset
        .drop(&["id"])
        .corrplot()
        .title("What correlates with the score")
        .cmap("coolwarm");
    dataset
        .scatter("study_hours", "exam_score", "exam_difficulty")
        .title("Study hours against score")
        .palette("tab10")
        .hue_order(&["easy", "moderate", "hard"]);
    dataset
        .regplot("class_attendance", "exam_score", "")
        .title("Attendance pays off")
        .sample(8000)
        .color("#5BC0EB")
        .xlim(40.0, 100.0)
        .ylim(0.0, 100.0);
    dataset
        .line("sleep_hours", "exam_score", "sleep_quality")
        .title("Mean score by hours of sleep")
        .theme("light")
        .palette("dark")
        .hue_order(&["poor", "average", "good"])
        .xlabel("sleep (hours)")
        .ylabel("mean exam score");

    // Sorting, sideways bars, shares, stacks, reference lines, legend placement, color names.
    dataset
        .bar("course", "exam_score", "", "mean")
        .title("Courses ranked by mean score")
        .sort("value")
        .horizontal()
        .ylim(60.0, 64.0);
    dataset
        .count("study_method", "")
        .title("Study methods A to Z")
        .sort("name")
        .horizontal()
        .percent()
        .color("teal");
    dataset
        .count("exam_difficulty", "sleep_quality")
        .title("Share of students by difficulty and sleep")
        .order(&["easy", "moderate", "hard"])
        .hue_order(&["poor", "average", "good"])
        .stacked()
        .percent()
        .colors(&["tomato", "gold", "mediumseagreen"])
        .legend("bottom");
    dataset
        .hist("exam_score", "exam_difficulty", 20)
        .title("Scores stacked by difficulty")
        .hue_order(&["easy", "moderate", "hard"])
        .stacked()
        .vline(62.51, "mean");
    dataset
        .scatter("study_hours", "exam_score", "")
        .title("Eight thousand students")
        .sample(8000)
        .color("tab:orange")
        .hline(62.51, "mean score")
        .vline(4.0, "4 hours");

    // Shapes and gaps: pair plot, densities, violins, a Q-Q plot, missing values.
    dataset
        .pairplot(
            &[
                "study_hours",
                "class_attendance",
                "sleep_hours",
                "exam_score",
            ],
            "study_method",
        )
        .title("How the numbers move together");
    dataset
        .kdeplot("exam_score", "sleep_quality")
        .title("Score density by sleep quality")
        .hue_order(&["poor", "average", "good"]);
    dataset
        .hist("exam_score", "", 30)
        .title("Exam scores with a density line")
        .kde();
    dataset
        .violinplot("course", "exam_score", "")
        .title("Score shapes by course")
        .sort("value");
    dataset
        .qqplot("exam_score")
        .title("Are exam scores normal?");
    dataset.missing().title("Missing values per column");
    dataset.missing_matrix();

    // An explicit path replaces the auto-save; show() also opens the viewer.
    dataset
        .bar("internet_access", "study_hours", "", "mean")
        .title("Study hours by internet access")
        .color_map(&[("yes", "#3A86FF"), ("no", "#FF006E")])
        .save("tests/fixtures/s6e1/metadata/plots/internet_vs_study_hours.svg");
    dataset
        .count("study_method", "")
        .title("How students study")
        .theme("light")
        .color("#4C72B0")
        .show();
}
