use rusmlai::prelude::*;

// One color per diagnosis, used on every chart split by it.
const DIAGNOSIS: &[(&str, &str)] = &[("Presence", "#D1495B"), ("Absence", "#00798C")];

fn main() {
    let dataset: Dataset =
        read_csv("/home/kartikag234/Documents/rusmlai/tests/fixtures/s6e2/train.csv");
    dataset.metadata(&[]);
    dataset.head(5);
    dataset.describe();
    dataset.info();

    // Preprocessing: scale the measurements, one-hot the multi-value codes. The 0/1 codes
    // (Sex, FBS over 120, Exercise angina) are read by no add, so "passthrough" keeps them.
    let x: Dataset = dataset.drop(&["id", "Heart Disease"]);
    let mut prep: ColumnTransformer = ColumnTransformer::new()
        .add(
            "num",
            &["Age", "BP", "Cholesterol", "Max HR", "ST depression"],
            steps![StandardScaler::new()],
        )
        .add(
            "code",
            &[
                "Chest pain type",
                "EKG results",
                "Slope of ST",
                "Number of vessels fluro",
                "Thallium",
            ],
            steps![OneHotEncoder::new()],
        )
        .remainder("passthrough");
    let features: Dataset = prep.fit_transform(&x);
    println!(
        "{} features: {:?}",
        features.columns().len(),
        prep.get_feature_names_out()
    );
    features.head(5);

    // Every chart below starts from a light card with seaborn's "deep" colors.
    // Each chart saves itself to tests/fixtures/s6e2/metadata/plots/{title}.png.
    set_theme("light");
    set_palette("deep");

    // The target.
    dataset
        .countplot("Heart Disease", "")
        .title("Heart disease in the training set")
        .color_map(DIAGNOSIS);
    dataset
        .pie("Heart Disease")
        .title("Share with heart disease")
        .theme("dark")
        .color_map(DIAGNOSIS);

    // Codes that separate the two groups.
    dataset
        .countplot("Chest pain type", "Heart Disease")
        .title("Chest pain type by diagnosis")
        .subtitle("type 4 is chest pain without symptoms (asymptomatic)")
        .color_map(DIAGNOSIS);
    dataset
        .countplot("Thallium", "Heart Disease")
        .title("Thallium test by diagnosis")
        .order(&["3", "7", "6"])
        .color_map(DIAGNOSIS);
    dataset
        .countplot("Number of vessels fluro", "Sex")
        .title("Vessels seen on fluoroscopy, by sex")
        .theme("dark")
        .palette("bright");
    dataset
        .bar(
            "Chest pain type",
            "ST depression",
            "Exercise angina",
            "mean",
        )
        .title("ST depression by chest pain and angina")
        .palette("pastel")
        .ylabel("mean ST depression (mm)");

    // The vitals.
    dataset
        .hist("Max HR", "Heart Disease", 16)
        .title("Maximum heart rate by diagnosis")
        .color_map(DIAGNOSIS)
        .xlabel("max heart rate (bpm)");
    dataset
        .hist("Cholesterol", "", 50)
        .title("Cholesterol")
        .theme("dark")
        .color("#F4A259")
        .xlabel("serum cholesterol (mg/dl)")
        .xlim(100.0, 400.0);
    dataset
        .hist("Age", "Sex", 12)
        .title("Age by sex")
        .colors(&["#9B5DE5", "#00BBF9"]);
    dataset
        .boxplot("", "Cholesterol", "")
        .title("Cholesterol outliers")
        .theme("dark")
        .color("#F25F5C");
    dataset
        .boxplot("Heart Disease", "Max HR", "Sex")
        .title("Maximum heart rate by diagnosis and sex")
        .palette("dark");
    dataset
        .stripplot("Slope of ST", "ST depression", "Heart Disease")
        .title("ST depression by slope of ST")
        .color_map(DIAGNOSIS);
    dataset
        .describe_bar("Max HR", "")
        .title("Max HR at a glance")
        .theme("dark")
        .color("#00F5D4");

    // Pairs of numbers.
    dataset
        .heatmap("Chest pain type", "Thallium", "", "")
        .title("Patients by chest pain and thallium")
        .theme("dark")
        .cmap("magma");
    dataset
        .heatmap("Number of vessels fluro", "Slope of ST", "Max HR", "mean")
        .title("Mean max heart rate by vessels and slope")
        .cmap("viridis");
    dataset
        .drop(&["id"])
        .corrplot()
        .title("Correlation of the measurements")
        .cmap("rdbu")
        .size(1100.0, 680.0);
    dataset
        .scatter("Age", "Max HR", "Heart Disease")
        .title("Age against maximum heart rate")
        .color_map(DIAGNOSIS);
    dataset
        .regplot("Age", "Max HR", "")
        .title("Maximum heart rate falls with age")
        .color("#2E86AB")
        .ylim(60.0, 210.0);
    dataset
        .regplot("BP", "Cholesterol", "Sex")
        .title("Blood pressure against cholesterol")
        .theme("dark")
        .palette("colorblind")
        .xlim(90.0, 200.0);
    dataset
        .plot("Age", "Max HR", "Heart Disease")
        .title("Mean maximum heart rate by age")
        .color_map(DIAGNOSIS);
    dataset
        .plot("Age", "ST depression", "Sex")
        .title("Mean ST depression by age and sex")
        .theme("dark")
        .colors(&["#FEE440", "#F15BB5"]);

    // Sorting, sideways bars, shares, stacks, reference lines, legend placement, color names.
    dataset
        .countplot("Chest pain type", "Heart Disease")
        .title("Diagnosis share by chest pain type")
        .horizontal()
        .stacked()
        .percent()
        .color_map(&[("Presence", "crimson"), ("Absence", "teal")])
        .legend("bottom");
    dataset
        .countplot("Number of vessels fluro", "")
        .title("Vessels, most common first")
        .sort("value")
        .percent()
        .theme("dark");
    dataset
        .hist("Cholesterol", "Heart Disease", 24)
        .title("Cholesterol against the 240 mg/dl line")
        .stacked()
        .color_map(DIAGNOSIS)
        .xlim(100.0, 400.0)
        .vline(240.0, "high: 240");
    dataset
        .plot("Age", "Max HR", "")
        .title("Mean max heart rate with a 150 bpm line")
        .color("darkorange")
        .hline(150.0, "150 bpm");
    dataset
        .boxplot("Chest pain type", "Max HR", "Heart Disease")
        .title("Max HR by chest pain and diagnosis")
        .sort("value")
        .color_map(DIAGNOSIS)
        .legend("bottom");

    // Shapes and gaps: pair plot, densities, violins, a Q-Q plot, missing values.
    dataset
        .pairplot(
            &["Age", "BP", "Cholesterol", "Max HR", "ST depression"],
            "Heart Disease",
        )
        .title("Vitals in pairs")
        .color_map(DIAGNOSIS);
    dataset
        .kdeplot("Max HR", "Heart Disease")
        .title("Max heart rate density by diagnosis")
        .color_map(DIAGNOSIS);
    dataset
        .hist("Cholesterol", "", 40)
        .title("Cholesterol with a density line")
        .xlim(100.0, 400.0)
        .kde();
    dataset
        .violinplot("Heart Disease", "Age", "Sex")
        .title("Age by diagnosis and sex");
    dataset
        .qqplot("Cholesterol")
        .title("Is cholesterol normal?");
    dataset.missing();
}
