"""examples/s6e1.rs in pandas, scikit-learn, seaborn and matplotlib: python3 bench-python/s6e1.py

The charts match ours where Python allows: the same columns, estimators, orders, colors and
limits. Charts that sample rows in rusmlai (strip, scatter, regplot, pairplot) sample the same
number of rows here. seaborn's bootstrapped error bars are off (`errorbar=None`), because our
bars and lines have none.
"""

import common  # first: starts the clock before the imports
from common import chart, out_dir, report, stage, top_with_other

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = out_dir("s6e1")
LEVELS: list[str] = ["easy", "moderate", "hard"]
SLEEP: list[str] = ["poor", "average", "good"]

with stage("load", "read_csv"):
    dataset = pd.read_csv(common.ROOT / "tests/fixtures/s6e1/train.csv")

with stage("tables", "metadata"):
    # Ours reads these from the cache file; pandas computes them on every run.
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

# Preprocessing: X is every column but the id and the target.
x = dataset.drop(columns=["id", "exam_score"])
prep = ColumnTransformer(
    [
        (
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            ["age", "study_hours", "class_attendance", "sleep_hours"],
        ),
        (
            "cat",
            OneHotEncoder(sparse_output=False),
            ["gender", "course", "internet_access", "sleep_quality", "study_method", "facility_rating", "exam_difficulty"],
        ),
    ]
)
with stage("preprocessing", "ColumnTransformer fit"):
    prep.fit(x)
with stage("preprocessing", "ColumnTransformer transform"):
    features = prep.transform(x)
names = prep.get_feature_names_out()
print(f"{len(names)} features: {list(names)}")
print(pd.DataFrame(features[:5], columns=names))

# Who the students are.
with chart("countplot", "Students per course", OUT) as ax:
    sns.countplot(data=dataset, x="course", ax=ax)
with chart("countplot", "How each gender studies", OUT) as ax:
    sns.countplot(data=dataset, x="study_method", hue="gender", palette="bright", ax=ax)
with chart("pie", "Study methods", OUT) as ax:
    counts = top_with_other(dataset["study_method"].value_counts(), 8)
    ax.pie(counts, labels=counts.index, colors=sns.color_palette("pastel"), autopct="%.1f%%")
with chart("pie", "The four biggest courses", OUT) as ax:
    counts = top_with_other(dataset["course"].value_counts(), 4)
    ax.pie(counts, labels=counts.index, colors=["#F94144", "#F8961E", "#F9C74F", "#90BE6D", "#577590"], autopct="%.1f%%")

# What moves the score.
with chart("bar", "Mean score by sleep quality", OUT) as ax:
    sns.barplot(data=dataset, x="sleep_quality", y="exam_score", order=SLEEP, color="#E8C547", errorbar=None, ax=ax)
    ax.set_ylim(55, 70)
    ax.axhline(62.51, label="overall mean")
with chart("bar", "Mean score by course and exam difficulty", OUT) as ax:
    sns.barplot(
        data=dataset, x="course", y="exam_score", hue="exam_difficulty", hue_order=LEVELS,
        palette="colorblind", errorbar=None, ax=ax,
    )
    ax.set_ylim(50, 70)
with chart("countplot", "Exam difficulty by facility rating", OUT) as ax:
    sns.countplot(
        data=dataset, x="facility_rating", hue="exam_difficulty", order=["low", "medium", "high"],
        hue_order=LEVELS, palette={"easy": "#2A9D8F", "moderate": "#E9C46A", "hard": "#E76F51"}, ax=ax,
    )
with chart("bar", "Median score by age", OUT) as ax:
    sns.barplot(data=dataset, x="age", y="exam_score", estimator="median", color="#7EB6FF", errorbar=None, ax=ax)
    ax.set_ylim(55, 66)
with chart("describe_bar", "exam_score at a glance", OUT) as ax:
    s = dataset["exam_score"].describe()
    ax.bar(["min", "q25", "median", "mean", "q75", "max", "std"],
           [s["min"], s["25%"], s["50%"], s["mean"], s["75%"], s["max"], s["std"]], color="#C792EA")

# How scores are spread.
with chart("hist", "Exam scores", OUT) as ax:
    sns.histplot(data=dataset, x="exam_score", bins=40, color="#FF9F1C", ax=ax)
with chart("hist", "Study hours with and without internet", OUT) as ax:
    sns.histplot(data=dataset, x="study_hours", hue="internet_access", bins=16,
                 palette={"yes": "#3A86FF", "no": "#FF006E"}, ax=ax)
with chart("boxplot", "Scores per course", OUT) as ax:
    sns.boxplot(
        data=dataset, x="course", y="exam_score", hue="course", legend=False, ax=ax,
        palette={"b.tech": "#F94144", "b.sc": "#F3722C", "b.com": "#F8961E", "bca": "#F9C74F",
                 "bba": "#90BE6D", "ba": "#43AA8B", "diploma": "#577590"},
    )
with chart("boxplot", "Scores by sleep quality and gender", OUT) as ax:
    sns.boxplot(data=dataset, x="sleep_quality", y="exam_score", hue="gender", order=SLEEP, palette="muted", ax=ax)
with chart("stripplot", "Every dot is a student", OUT) as ax:
    sns.stripplot(
        data=dataset.sample(2500, random_state=0), x="exam_difficulty", y="exam_score", hue="sleep_quality",
        order=LEVELS, hue_order=SLEEP, palette=["#EF476F", "#FFD166", "#06D6A0"], ax=ax,
    )

# Two numbers at once.
with chart("heatmap", "Mean score by course and study method", OUT) as ax:
    grid = dataset.pivot_table(index="study_method", columns="course", values="exam_score", aggfunc="mean")
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="viridis", ax=ax)
with chart("heatmap", "Students by sleep and facilities", OUT) as ax:
    sns.heatmap(pd.crosstab(dataset["facility_rating"], dataset["sleep_quality"]), annot=True, fmt="d", cmap="Blues", ax=ax)
with chart("corrplot", "What correlates with the score", OUT) as ax:
    corr = dataset.drop(columns=["id"]).corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
with chart("scatter", "Study hours against score", OUT) as ax:
    sns.scatterplot(data=dataset.sample(2500, random_state=0), x="study_hours", y="exam_score",
                    hue="exam_difficulty", hue_order=LEVELS, palette="tab10", ax=ax)
with chart("regplot", "Attendance pays off", OUT) as ax:
    sns.regplot(data=dataset.sample(8000, random_state=0), x="class_attendance", y="exam_score",
                ci=None, color="#5BC0EB", ax=ax)
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
with chart("plot", "Mean score by hours of sleep", OUT) as ax:
    sns.lineplot(data=dataset, x="sleep_hours", y="exam_score", hue="sleep_quality", hue_order=SLEEP,
                 palette="dark", errorbar=None, ax=ax)
    ax.set_xlabel("sleep (hours)")
    ax.set_ylabel("mean exam score")

# Sorting, sideways bars, shares, stacks, reference lines, legend placement, color names.
with chart("bar", "Courses ranked by mean score", OUT) as ax:
    means = dataset.groupby("course")["exam_score"].mean().sort_values(ascending=False)
    ax.barh(means.index, means.values)
    ax.invert_yaxis()
    ax.set_xlim(60, 64)
with chart("countplot", "Study methods A to Z", OUT) as ax:
    sns.countplot(data=dataset, y="study_method", order=sorted(dataset["study_method"].unique()),
                  stat="percent", color="teal", ax=ax)
with chart("countplot", "Share of students by difficulty and sleep", OUT) as ax:
    shares = pd.crosstab(dataset["exam_difficulty"], dataset["sleep_quality"], normalize="all") * 100
    shares.loc[LEVELS, SLEEP].plot.bar(stacked=True, color=["tomato", "gold", "mediumseagreen"], ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
with chart("hist", "Scores stacked by difficulty", OUT) as ax:
    sns.histplot(data=dataset, x="exam_score", hue="exam_difficulty", hue_order=LEVELS, bins=20, multiple="stack", ax=ax)
    ax.axvline(62.51, label="mean")
with chart("scatter", "Eight thousand students", OUT) as ax:
    sns.scatterplot(data=dataset.sample(8000, random_state=0), x="study_hours", y="exam_score", color="tab:orange", ax=ax)
    ax.axhline(62.51, label="mean score")
    ax.axvline(4.0, label="4 hours")

# Shapes and gaps: pair plot, densities, violins, a Q-Q plot, missing values.
with chart("pairplot", "How the numbers move together", OUT):
    sns.pairplot(dataset.sample(1000, random_state=0), vars=["study_hours", "class_attendance", "sleep_hours", "exam_score"],
                 hue="study_method", diag_kind="hist")
with chart("kdeplot", "Score density by sleep quality", OUT) as ax:
    sns.kdeplot(data=dataset, x="exam_score", hue="sleep_quality", hue_order=SLEEP, ax=ax)
with chart("hist", "Exam scores with a density line", OUT) as ax:
    sns.histplot(data=dataset, x="exam_score", bins=30, kde=True, ax=ax)
with chart("violinplot", "Score shapes by course", OUT) as ax:
    order = dataset.groupby("course")["exam_score"].median().sort_values(ascending=False).index
    sns.violinplot(data=dataset, x="course", y="exam_score", order=order, ax=ax)
with chart("qqplot", "Are exam scores normal?", OUT) as ax:
    scipy.stats.probplot(dataset["exam_score"], dist="norm", plot=ax)
with chart("missing", "Missing values per column", OUT) as ax:
    msno.bar(dataset, ax=ax)
with chart("missing_matrix", "Missing values by row", OUT):
    msno.matrix(dataset)

# An explicit path; ours also opens the last chart in a viewer, which is not timed here.
with chart("bar", "Study hours by internet access", OUT, OUT / "internet_vs_study_hours.svg") as ax:
    sns.barplot(data=dataset, x="internet_access", y="study_hours", hue="internet_access", legend=False,
                palette={"yes": "#3A86FF", "no": "#FF006E"}, errorbar=None, ax=ax)
with chart("countplot", "How students study", OUT) as ax:
    sns.countplot(data=dataset, x="study_method", color="#4C72B0", ax=ax)

report("s6e1")
