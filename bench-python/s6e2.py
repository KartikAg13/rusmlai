"""examples/s6e2.rs in pandas, scikit-learn, seaborn and matplotlib: python3 bench-python/s6e2.py

The same rules as s6e1.py: the same charts where Python allows, the same sample sizes for the
charts that sample rows, and no bootstrapped error bars.
"""

import common  # first: starts the clock before the imports
from common import chart, out_dir, report, stage, top_with_other

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = out_dir("s6e2")
# One color per diagnosis, used on every chart split by it.
DIAGNOSIS: dict[str, str] = {"Presence": "#D1495B", "Absence": "#00798C"}

with stage("load", "read_csv"):
    dataset = pd.read_csv(common.ROOT / "tests/fixtures/s6e2/train.csv")

with stage("tables", "metadata"):
    meta = pd.DataFrame(
        {"dtype": dataset.dtypes.astype(str), "nulls": dataset.isna().sum(), "distinct": dataset.nunique()}
    ).join(dataset.describe(include="all").T)
    print(meta)
with stage("tables", "head"):
    print(dataset.head(5))
with stage("tables", "describe"):
    print(dataset.describe())
with stage("tables", "info"):
    dataset.info()

# Preprocessing: scale the measurements, one-hot the multi-value codes, keep the 0/1 codes.
x = dataset.drop(columns=["id", "Heart Disease"])
prep = ColumnTransformer(
    [
        ("num", StandardScaler(), ["Age", "BP", "Cholesterol", "Max HR", "ST depression"]),
        (
            "code",
            OneHotEncoder(sparse_output=False),
            ["Chest pain type", "EKG results", "Slope of ST", "Number of vessels fluro", "Thallium"],
        ),
    ],
    remainder="passthrough",
)
with stage("preprocessing", "ColumnTransformer fit"):
    prep.fit(x)
with stage("preprocessing", "ColumnTransformer transform"):
    features = prep.transform(x)
names = prep.get_feature_names_out()
print(f"{len(names)} features: {list(names)}")
print(pd.DataFrame(features[:5], columns=names))

# Every chart below starts with seaborn's "deep" colors.
sns.set_palette("deep")

# The target.
with chart("countplot", "Heart disease in the training set", OUT) as ax:
    sns.countplot(data=dataset, x="Heart Disease", hue="Heart Disease", palette=DIAGNOSIS, legend=False, ax=ax)
with chart("pie", "Share with heart disease", OUT) as ax:
    counts = top_with_other(dataset["Heart Disease"].value_counts(), 8)
    ax.pie(counts, labels=counts.index, colors=[DIAGNOSIS[c] for c in counts.index], autopct="%.1f%%")

# Codes that separate the two groups.
with chart("countplot", "Chest pain type by diagnosis", OUT) as ax:
    sns.countplot(data=dataset, x="Chest pain type", hue="Heart Disease", palette=DIAGNOSIS, ax=ax)
    ax.text(0.5, 1.01, "type 4 is chest pain without symptoms (asymptomatic)", transform=ax.transAxes, ha="center")
with chart("countplot", "Thallium test by diagnosis", OUT) as ax:
    sns.countplot(data=dataset, x="Thallium", hue="Heart Disease", order=[3, 7, 6], palette=DIAGNOSIS, ax=ax)
with chart("countplot", "Vessels seen on fluoroscopy, by sex", OUT) as ax:
    sns.countplot(data=dataset, x="Number of vessels fluro", hue="Sex", palette="bright", ax=ax)
with chart("bar", "ST depression by chest pain and angina", OUT) as ax:
    sns.barplot(data=dataset, x="Chest pain type", y="ST depression", hue="Exercise angina",
                palette="pastel", errorbar=None, ax=ax)
    ax.set_ylabel("mean ST depression (mm)")

# The vitals.
with chart("hist", "Maximum heart rate by diagnosis", OUT) as ax:
    sns.histplot(data=dataset, x="Max HR", hue="Heart Disease", bins=16, palette=DIAGNOSIS, ax=ax)
    ax.set_xlabel("max heart rate (bpm)")
with chart("hist", "Cholesterol", OUT) as ax:
    sns.histplot(data=dataset, x="Cholesterol", bins=50, binrange=(100, 400), color="#F4A259", ax=ax)
    ax.set_xlabel("serum cholesterol (mg/dl)")
    ax.set_xlim(100, 400)
with chart("hist", "Age by sex", OUT) as ax:
    sns.histplot(data=dataset, x="Age", hue="Sex", bins=12, palette=["#9B5DE5", "#00BBF9"], ax=ax)
with chart("boxplot", "Cholesterol outliers", OUT) as ax:
    sns.boxplot(data=dataset, y="Cholesterol", color="#F25F5C", ax=ax)
with chart("boxplot", "Maximum heart rate by diagnosis and sex", OUT) as ax:
    sns.boxplot(data=dataset, x="Heart Disease", y="Max HR", hue="Sex", palette="dark", ax=ax)
with chart("stripplot", "ST depression by slope of ST", OUT) as ax:
    sns.stripplot(data=dataset.sample(2500, random_state=0), x="Slope of ST", y="ST depression",
                  hue="Heart Disease", palette=DIAGNOSIS, ax=ax)
with chart("describe_bar", "Max HR at a glance", OUT) as ax:
    s = dataset["Max HR"].describe()
    ax.bar(["min", "q25", "median", "mean", "q75", "max", "std"],
           [s["min"], s["25%"], s["50%"], s["mean"], s["75%"], s["max"], s["std"]], color="#00F5D4")

# Pairs of numbers.
with chart("heatmap", "Patients by chest pain and thallium", OUT) as ax:
    sns.heatmap(pd.crosstab(dataset["Thallium"], dataset["Chest pain type"]), annot=True, fmt="d", cmap="magma", ax=ax)
with chart("heatmap", "Mean max heart rate by vessels and slope", OUT) as ax:
    grid = dataset.pivot_table(index="Slope of ST", columns="Number of vessels fluro", values="Max HR", aggfunc="mean")
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="viridis", ax=ax)
with chart("corrplot", "Correlation of the measurements", OUT) as ax:
    ax.figure.set_size_inches(11, 6.8)
    corr = dataset.drop(columns=["id"]).corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", vmin=-1, vmax=1, ax=ax)
with chart("scatter", "Age against maximum heart rate", OUT) as ax:
    sns.scatterplot(data=dataset.sample(2500, random_state=0), x="Age", y="Max HR", hue="Heart Disease",
                    palette=DIAGNOSIS, ax=ax)
with chart("regplot", "Maximum heart rate falls with age", OUT) as ax:
    sns.regplot(data=dataset.sample(2500, random_state=0), x="Age", y="Max HR", ci=None, color="#2E86AB", ax=ax)
    ax.set_ylim(60, 210)
with chart("regplot", "Blood pressure against cholesterol", OUT):
    # regplot has no hue; lmplot is seaborn's regplot with one.
    sns.lmplot(data=dataset.sample(2500, random_state=0), x="BP", y="Cholesterol", hue="Sex", ci=None,
               palette="colorblind")
    plt.xlim(90, 200)
with chart("plot", "Mean maximum heart rate by age", OUT) as ax:
    sns.lineplot(data=dataset, x="Age", y="Max HR", hue="Heart Disease", palette=DIAGNOSIS, errorbar=None, ax=ax)
with chart("plot", "Mean ST depression by age and sex", OUT) as ax:
    sns.lineplot(data=dataset, x="Age", y="ST depression", hue="Sex", palette=["#FEE440", "#F15BB5"],
                 errorbar=None, ax=ax)

# Sorting, sideways bars, shares, stacks, reference lines, legend placement, color names.
with chart("countplot", "Diagnosis share by chest pain type", OUT) as ax:
    shares = pd.crosstab(dataset["Chest pain type"], dataset["Heart Disease"], normalize="all") * 100
    shares.plot.barh(stacked=True, color=[{"Presence": "crimson", "Absence": "teal"}[c] for c in shares.columns], ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
with chart("countplot", "Vessels, most common first", OUT) as ax:
    order = dataset["Number of vessels fluro"].value_counts().index
    sns.countplot(data=dataset, x="Number of vessels fluro", order=order, stat="percent", ax=ax)
with chart("hist", "Cholesterol against the 240 mg/dl line", OUT) as ax:
    sns.histplot(data=dataset, x="Cholesterol", hue="Heart Disease", bins=24, binrange=(100, 400),
                 multiple="stack", palette=DIAGNOSIS, ax=ax)
    ax.set_xlim(100, 400)
    ax.axvline(240.0, label="high: 240")
with chart("plot", "Mean max heart rate with a 150 bpm line", OUT) as ax:
    sns.lineplot(data=dataset, x="Age", y="Max HR", color="darkorange", errorbar=None, ax=ax)
    ax.axhline(150.0, label="150 bpm")
with chart("boxplot", "Max HR by chest pain and diagnosis", OUT) as ax:
    order = dataset.groupby("Chest pain type")["Max HR"].median().sort_values(ascending=False).index
    sns.boxplot(data=dataset, x="Chest pain type", y="Max HR", hue="Heart Disease", order=order,
                palette=DIAGNOSIS, ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

# Shapes and gaps: pair plot, densities, violins, a Q-Q plot, missing values.
with chart("pairplot", "Vitals in pairs", OUT):
    sns.pairplot(dataset.sample(1000, random_state=0), vars=["Age", "BP", "Cholesterol", "Max HR", "ST depression"],
                 hue="Heart Disease", palette=DIAGNOSIS, diag_kind="hist")
with chart("kdeplot", "Max heart rate density by diagnosis", OUT) as ax:
    sns.kdeplot(data=dataset, x="Max HR", hue="Heart Disease", palette=DIAGNOSIS, ax=ax)
with chart("hist", "Cholesterol with a density line", OUT) as ax:
    sns.histplot(data=dataset, x="Cholesterol", bins=40, binrange=(100, 400), kde=True, ax=ax)
    ax.set_xlim(100, 400)
with chart("violinplot", "Age by diagnosis and sex", OUT) as ax:
    sns.violinplot(data=dataset, x="Heart Disease", y="Age", hue="Sex", ax=ax)
with chart("qqplot", "Is cholesterol normal?", OUT) as ax:
    scipy.stats.probplot(dataset["Cholesterol"], dist="norm", plot=ax)
with chart("missing", "Missing values per column", OUT) as ax:
    msno.bar(dataset, ax=ax)

report("s6e2")
