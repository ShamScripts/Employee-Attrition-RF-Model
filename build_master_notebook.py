"""
Builds 01_Random_Forest_Attrition_Complete.ipynb — Employee attrition with Random Forest.
Run: python build_master_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "01_Random_Forest_Attrition_Complete.ipynb"

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in s.split("\n")]})

def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in s.split("\n")], "outputs": [], "execution_count": None})

# =============================================================================
md("""# Employee attrition — Random Forest  
### Predict turnover with **Random Forest**; compare **logistic regression** and **decision tree** on IBM HR data

**How to run:** execute all cells **top to bottom** from the project folder (same folder as `ibm_hr_attrition.csv` and `plots/`).

---

### In this notebook

1. How **Random Forest** relates to **logistic regression** and a single **decision tree**.  
2. A full workflow: **data → preprocess → train → evaluate → interpret**.  
3. **Train vs test** accuracy, **confusion matrices**, **ROC/AUC**, **feature importance**.  
4. **Predicted probabilities** as an attrition **risk score**.

---

### Table of contents

| Section | Content |
|---------|---------|
| **1 · Introduction** | Goal: teach Random Forest; attrition as illustration |
| **2 · Dataset overview** | Load IBM data, quick exploration |
| **3 · Data preprocessing** | Cleaning, encoding, train/test split |
| **4 · Concepts** | LR limits, tree overfitting, RF: bagging, random features, voting |
| **5 · Logistic regression (baseline)** | Linear benchmark |
| **6 · Decision tree (baseline)** | Nonlinear rules; **overfitting** (train ≫ test) |
| **7 · Random Forest (main model)** | Ensemble; **diagram** + sample **trees** |
| **8 · Model evaluation** | Metrics, **confusion matrices**, ROC |
| **9 · Feature importance** | Plot + plain-language interpretation |
| **10 · Comparison** | DT vs RF (and LR): **accuracy** bars |
| **11 · Attrition risk score** | Probabilities, not only labels |
| **12 · What-if analysis** | Change one input (e.g. overtime) |
| **13 · Conclusion** | Takeaways, limitations, references |

---""")

md("""## 1 · Introduction

**Goal:** use **Random Forest** — an **ensemble** of decision trees trained with **bootstrap samples** and **random feature subsets** at each split, combined by **majority vote** (classification).

**Use case:** predict **employee attrition** (`Attrition` = Yes/No) from IBM-style HR data (mixed numeric and categorical fields). **Logistic regression** and a **decision tree** are trained on the **same split** for comparison.

- **Logistic regression** — linear baseline.  
- **Decision tree** — single tree; can **overfit** when deep.  
- **Random Forest** — many trees; often **better generalization** than one deep tree.

---""")

md("""## 2 · Dataset overview

**What we do:** load `ibm_hr_attrition.csv`, check shape and types, optionally subsample for fast demos, visualize **class balance** and a few relationships.

**Why:** understand the **target** (`Attrition`) and whether classes are **imbalanced** (common in attrition).

**What to expect:** more “No” than “Yes”; features include demographics, job, pay, satisfaction, tenure.

---""")

code("""# --- Setup: imports, paths, reproducibility ---
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
from sklearn.feature_selection import f_classif

import holoviews as hv
import hvplot.pandas
hv.extension("bokeh")

ROOT = Path.cwd()
if not (ROOT / "ibm_hr_attrition.csv").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "ibm_hr_attrition.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Attrition"
RANDOM_STATE = 42
USE_SMALL_SAMPLE = False   # set True for ~500-row stratified demo
SAMPLE_N = 500

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

print("Dataset path:", DATA_PATH.resolve())""")

code("""# Load data; optional stratified subsample for weak machines / short class periods
df_raw = pd.read_csv(DATA_PATH)

if USE_SMALL_SAMPLE and len(df_raw) > SAMPLE_N:
    df_raw, _ = train_test_split(
        df_raw, train_size=SAMPLE_N, stratify=df_raw[TARGET], random_state=RANDOM_STATE
    )
    df_raw = df_raw.reset_index(drop=True)
    print(f"[Demo mode] Using n = {len(df_raw)} rows (stratified).")
else:
    print(f"Using full sample: n = {len(df_raw)} rows.")

print("Shape (rows, columns):", df_raw.shape)
df_raw.head()""")

code("""df_raw.info()""")

code("""# Quick variable lists (numeric vs categorical) — informs preprocessing
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_raw.select_dtypes(include=["object", "category"]).columns.tolist()
print("Numeric columns:", len(num_cols), "| Categorical:", len(cat_cols))""")

code("""# Exploratory plots: class balance and two relationships
df_eda = df_raw.copy()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df_eda[TARGET].value_counts().plot(kind="bar", ax=axes[0], color=["#2c7bb6", "#d7191c"], rot=0)
axes[0].set_title("Attrition — counts")
df_eda[TARGET].value_counts(normalize=True).plot(kind="bar", ax=axes[1], color=["#2c7bb6", "#d7191c"], rot=0)
axes[1].set_title("Attrition — class proportion")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_eda_balance.png", bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if "Age" in df_eda.columns:
    sns.boxplot(data=df_eda, x=TARGET, y="Age", ax=axes[0], palette=["#2c7bb6", "#d7191c"])
    axes[0].set_title("Age vs attrition")
if "MonthlyIncome" in df_eda.columns:
    sns.kdeplot(data=df_eda, x="MonthlyIncome", hue=TARGET, common_norm=False, ax=axes[1])
    axes[1].set_title("Monthly income vs attrition")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_eda_age_income.png", bbox_inches="tight")
plt.show()

if "OverTime" in df_eda.columns:
    ct = pd.crosstab(df_eda["OverTime"], df_eda[TARGET], normalize="index") * 100
    ct.plot(kind="bar", figsize=(6, 4), color=["#2c7bb6", "#d7191c"], rot=0)
    plt.ylabel("Percent within overtime group")
    plt.title("Attrition % by overtime")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig_eda_overtime.png", bbox_inches="tight")
    plt.show()""")

md("""## 3 · Data preprocessing

**What we do:** remove rows with missing values; drop **ID** and **constant** columns; encode `Attrition` as 0/1; **one-hot encode** categoricals (`drop_first=True` to avoid redundant dummies); **stratified** train/test split.

**Why:** `sklearn` needs numeric matrices; dropping IDs/constants avoids **spurious** fits; stratification keeps similar **churn rate** in train and test when classes are imbalanced.

**What to expect:** a design matrix `X` and vector `y`; train/test shapes about 80%/20%.

---""")

code("""# Cleaning: drop NA; remove uninformative columns (IBM attrition standard)
df = df_raw.copy().dropna()
DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
print("Shape after cleaning:", df.shape)

# Target: Yes -> 1, No -> 0
y = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
X = df.drop(columns=[TARGET])
X = pd.get_dummies(X, drop_first=True)

print("Encoded feature matrix:", X.shape)
print("Positive class rate (left company):", round(y.mean(), 4))""")

code("""# Stratified split: same churn rate approximately in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Train churn rate:", round(y_train.mean(), 4), "| Test:", round(y_test.mean(), 4))""")

md("""## 4 · Concepts — why Random Forest?

### Limitations of logistic regression (for this problem)

Logistic regression assumes a **linear** relationship between features and the **log-odds** of leaving. It is fast and interpretable, but **real HR processes** are often **nonlinear** (interactions, thresholds). LR is our **linear baseline**, not the “truth.”

### Why a single decision tree can overfit

A decision tree asks a sequence of yes/no questions until it reaches a **leaf** prediction. A **deep** tree can **memorize** training noise: **training accuracy** looks excellent, but **test accuracy** drops. That **gap** is the classic sign of **overfitting** (low bias, high variance on small/noisy regions).

**Analogy:** one expert who has **memorized** past cases but **does not generalize** to new employees.

### What is Random Forest?

**Random Forest** trains **many** decision trees on **different bootstrap samples** of rows (**bagging**). At each split, only a **random subset of features** is considered (**random feature selection**). For a new employee, each tree votes; the forest takes the **majority vote** (**aggregation**).

**Analogies:**

- **Bagging:** different study groups each review a **random resample** of past employees.  
- **Random features:** each group is **not allowed** to look at all variables at once — forces **diversity** of trees.  
- **Voting:** the **wisdom of the crowd** — wrong trees partly cancel out; **variance** often drops vs one tree.

**Reference:** L. Breiman (2001), *Random Forests*, Machine Learning.

---""")

md("""## 5 · Logistic regression (baseline)

**What we do:** fit a **pipeline**: scale features, then logistic regression (`max_iter` large enough to converge).

**Why:** strong **linear** baseline; probabilities via `predict_proba`.

**What to expect:** decent test accuracy; may underperform if decision boundaries are strongly nonlinear.

---""")

code("""# Logistic regression with scaling (stable optimization on mixed one-hot + numeric)
model_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=10000, solver="lbfgs", random_state=RANDOM_STATE)),
])
model_lr.fit(X_train, y_train)

acc_lr_tr = accuracy_score(y_train, model_lr.predict(X_train))
acc_lr_te = accuracy_score(y_test, model_lr.predict(X_test))
print("Logistic Regression — train accuracy:", round(acc_lr_tr, 4))
print("Logistic Regression — test  accuracy:", round(acc_lr_te, 4))""")

md("""## 6 · Decision tree (baseline) — overfitting in action

**What we do:** train a **deep** tree (`max_depth=None`, `min_samples_leaf=1`) so **train accuracy** is very high.

**Why:** a very flexible tree can **memorize** the training set (high train accuracy, lower test accuracy). This **train vs test gap** motivates **ensembles** like Random Forest.

**What to expect:** train accuracy near 1.0; test accuracy **lower**; visible **gap** between train and test.

---""")

code("""# Deep decision tree: prone to memorizing training set (high train acc, lower test acc)
model_dt = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
)
model_dt.fit(X_train, y_train)

pred_dt_tr = model_dt.predict(X_train)
pred_dt_te = model_dt.predict(X_test)
acc_dt_tr = accuracy_score(y_train, pred_dt_tr)
acc_dt_te = accuracy_score(y_test, pred_dt_te)

print("Decision Tree — train accuracy:", round(acc_dt_tr, 4))
print("Decision Tree — test  accuracy:", round(acc_dt_te, 4))
print("Train − test gap (overfitting signal):", round(acc_dt_tr - acc_dt_te, 4))""")

md("""## 7 · Random Forest (main model)

**What we do:** fit `RandomForestClassifier(n_estimators=100)`; draw a **schematic** (bootstrap → trees → vote); show **2–3** shallow `plot_tree` views of **different** trees in the forest.

**Why:** this is the **core algorithm** for the lecture: **variance reduction** via averaging **diverse** trees.

**What to expect:** test accuracy often **beats** the single deep tree; **smaller** train–test gap than the overfit tree.

---""")

code("""# Random Forest: ensemble of trees (bootstrap + random splits + majority vote)
N_TREES = 100
model_rf = RandomForestClassifier(
    n_estimators=N_TREES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model_rf.fit(X_train, y_train)

pred_rf_tr = model_rf.predict(X_train)
pred_rf_te = model_rf.predict(X_test)
acc_rf_tr = accuracy_score(y_train, pred_rf_tr)
acc_rf_te = accuracy_score(y_test, pred_rf_te)

print("Random Forest — train accuracy:", round(acc_rf_tr, 4))
print("Random Forest — test  accuracy:", round(acc_rf_te, 4))
print("Train − test gap:", round(acc_rf_tr - acc_rf_te, 4))""")

code("""# Visual: Random Forest flow — data -> many trees -> majority vote -> prediction
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.axis("off")
ax.text(5, 4.1, "Random Forest: many trees → majority vote", ha="center", fontsize=12, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.2, 1.5), 1.3, 1.0, boxstyle="round,pad=0.05", fc="#e8f4fc", ec="#333"))
ax.text(0.85, 2.0, "Bootstrap\\nsamples", ha="center", va="center", fontsize=9)

xs = [2.4, 3.5, 4.6, 5.7]
for i, x in enumerate(xs):
    ax.add_patch(FancyBboxPatch((x, 1.55), 0.85, 0.9, boxstyle="round,pad=0.02", fc="#fff8e7", ec="#333"))
    ax.text(x + 0.42, 2.0, f"Tree\\n{i+1}", ha="center", va="center", fontsize=8)

ax.add_patch(FancyBboxPatch((7.1, 1.55), 1.2, 0.9, boxstyle="round,pad=0.02", fc="#e8f8e8", ec="#333"))
ax.text(7.7, 2.0, "Majority\\nvote", ha="center", va="center", fontsize=9)

for x in xs:
    ax.annotate("", xy=(x + 0.42, 2.0), xytext=(0.85, 2.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.1))
    ax.annotate("", xy=(7.7, 2.0), xytext=(x + 0.85, 2.0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

ax.text(5, 0.45, "Flow: input features → each tree predicts class → forest aggregates by voting.",
        ha="center", fontsize=9, style="italic")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_rf_diagram.png", bbox_inches="tight")
plt.show()

print("ASCII sketch:  features  →  [Tree 1 … Tree %d]  →  vote  →  Stay/Leave" % N_TREES)""")

code("""# Sample three different trees from the forest (depth capped for slides)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, ax in enumerate(axes):
    plot_tree(
        model_rf.estimators_[i],
        max_depth=3,
        feature_names=list(X.columns),
        class_names=["Stay", "Leave"],
        filled=True,
        rounded=True,
        fontsize=5,
        ax=ax,
    )
    ax.set_title("Estimator %d (depth shown ≤ 3)" % (i + 1))
plt.suptitle("Random Forest contains diverse trees (different bootstrap samples & splits)", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_rf_sample_trees.png", bbox_inches="tight")
plt.show()""")

md("""## 8 · Model evaluation

**What we do:** compare **accuracy**, **precision**, **recall**, **F1** on the **test** set; plot **confusion matrix** heatmaps; plot **ROC** curves.

**Why:** accuracy alone is misleading when **“leave”** is rare; precision/recall describe **errors** differently; ROC summarizes **ranking** of predicted probabilities.

**What to expect:** Random Forest often competitive or best on **test**; compare shapes of confusion matrices.

---""")

code("""# Dictionary of fitted models for uniform evaluation
models = {
    "Logistic Regression": model_lr,
    "Decision Tree": model_dt,
    "Random Forest": model_rf,
}

def metrics_row(name, model):
    pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }

metrics_table = pd.DataFrame([metrics_row(n, m) for n, m in models.items()])
metrics_table""")

code("""# Confusion matrices (test set) — heatmaps
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
labels_cm = [["Pred stay", "Pred leave"], ["True stay", "True leave"]]
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels_cm[0], yticklabels=labels_cm[1])
    ax.set_title(name)
plt.suptitle("Confusion matrices — held-out test data", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_confusion_matrices.png", bbox_inches="tight")
plt.show()""")

code("""for name, model in models.items():
    print("=" * 60)
    print(name)
    print(classification_report(y_test, model.predict(X_test), target_names=["Stay", "Leave"], zero_division=0))""")

code("""# ROC curves and AUC (test set)
roc_list = []
plt.figure(figsize=(7, 6))
for name, model in models.items():
    p1 = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, p1)
    a = auc(fpr, tpr)
    roc_list.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": name}))
    plt.plot(fpr, tpr, lw=2, label="%s (AUC = %.3f)" % (name, a))

plt.plot([0, 1], [0, 1], "k--", alpha=0.35)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves — test set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_roc.png", bbox_inches="tight")
plt.show()

roc_df = pd.concat(roc_list, ignore_index=True)
roc_df.hvplot.line(
    x="fpr", y="tpr", by="model", width=680, height=480,
    xlabel="False positive rate", ylabel="True positive rate",
    title="ROC (interactive)",
    legend="bottom",
)""")

md("""## 9 · Feature importance (Random Forest)

**What we do:** plot `model_rf.feature_importances_` (Gini **mean decrease impurity** in `sklearn`).

**Why:** see which inputs the forest **used most often** in splits — useful for **discussion**, not proof of **causality** (observational HR data).

**What to expect:** variables like **overtime**, **income**, **tenure** often rank high; wording should stay cautious (“associated with,” not “causes”).

---""")

code("""# Random Forest: Gini-based feature importance (sklearn default)
importance_series = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top_k = importance_series.head(18)

fig, ax = plt.subplots(figsize=(9, 6))
top_k.sort_values().plot(kind="barh", ax=ax, color="#1a9850")
ax.set_title("Random Forest — feature importance (top 18)")
ax.set_xlabel("Importance (Gini decrease, aggregated over trees)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_feature_importance.png", bbox_inches="tight")
plt.show()

# Top features by importance (for interpretation — not causal)
print("Top 8 features (highest importance scores):")
for feat, val in importance_series.head(8).items():
    print("  %-40s  %.4f" % (feat, val))""")

md("""**Interpretation:**  
Higher importance means the feature was used more often in splits that reduced **Gini impurity** across trees. That indicates **association** with attrition in this model — not proof of **causation**. Real decisions need domain review and fairness checks.

---""")

md("""## 10 · Comparison — Decision Tree vs Random Forest (and logistic regression)

**What we do:** bar chart of **test accuracy** for all three models, plus **train vs test** accuracy side-by-side.

**Why:** make the **main punchline** visible: a **single overfit tree** vs an **ensemble**; LR as linear reference.

**What to expect:** RF often **highest or near-highest test accuracy**; DT may show **largest** train–test gap if deep.

---""")

code("""# Accuracy comparison: test set + train vs test bars
test_acc = {name: accuracy_score(y_test, m.predict(X_test)) for name, m in models.items()}
train_acc = {name: accuracy_score(y_train, m.predict(X_train)) for name, m in models.items()}

compare_df = pd.DataFrame({
    "train": [train_acc["Logistic Regression"], train_acc["Decision Tree"], train_acc["Random Forest"]],
    "test": [test_acc["Logistic Regression"], test_acc["Decision Tree"], test_acc["Random Forest"]],
}, index=["Logistic Regression", "Decision Tree", "Random Forest"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = compare_df["test"].plot(kind="bar", ax=axes[0], color=["#4a90d9", "#fdae61", "#2ca25f"], rot=15)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Test set accuracy — model comparison")
axes[0].set_ylim(0, 1.05)

compare_df.plot(kind="bar", ax=axes[1], color=["#4a90d9", "#e66101"], rot=15)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Train vs test — overfitting vs generalization")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_accuracy_comparison.png", bbox_inches="tight")
plt.show()

compare_df.round(4)""")

md("""**Why Random Forest often wins vs one deep tree:** the forest **averages** many **noisy** tree predictions; **bagging** and **random features** make trees **less correlated**, so errors cancel. **Logistic regression** stays useful as a **simple, linear** reference.

---""")

md("""## 11 · Attrition risk score (predicted probability)

**What we do:** use `predict_proba[:, 1]` from the Random Forest as **P(leave)** — an **attrition risk score** between 0 and 1.

**Why:** a **probability** is often more actionable than a hard label — you can set thresholds (e.g. flag high-risk cases).

**What to expect:** scores often differ between people who **left** vs **stayed**; overlap between groups is normal.

---""")

code("""# Attrition risk score = P(attrition = 1 | x) from Random Forest
risk_score_test = model_rf.predict_proba(X_test)[:, 1]

example_df = pd.DataFrame({
    "actual_left": y_test.values,
    "risk_score": np.round(risk_score_test, 4),
}).head(12)
example_df""")

code("""# Distribution of risk scores by true label
plt.figure(figsize=(7, 4))
plt.hist(risk_score_test[y_test == 0], bins=22, alpha=0.55, label="Actually stayed", color="#2c7bb6")
plt.hist(risk_score_test[y_test == 1], bins=22, alpha=0.55, label="Actually left", color="#d7191c")
plt.xlabel("Random Forest P(leave) — attrition risk score")
plt.ylabel("Count")
plt.title("Distribution of predicted leave probabilities (test set)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_risk_score_distribution.png", bbox_inches="tight")
plt.show()""")

md("""## 12 · What-if analysis (bonus)

**What we do:** take **one** test row; flip a **single** interpretable feature if present (here: **overtime** one-hot column), holding others fixed; compare **risk score**.

**Why:** shows **local sensitivity** of the model — useful for discussion (**not** a causal claim).

**What to expect:** if overtime increases predicted leave probability, the model **associates** overtime with attrition in this dataset.

---""")

code("""# What-if: toggle overtime-related column if it exists after get_dummies
row = X_test.iloc[[0]].copy()
base_prob = model_rf.predict_proba(row)[0, 1]

# Common encodings: OverTime_Yes, or Overtime_Yes — match any column containing OverTime
ot_cols = [c for c in row.columns if "OverTime" in c or "Overtime" in c]
print("Baseline P(leave) for first test row:", round(base_prob, 4))
print("Overtime-related columns found:", ot_cols)

if ot_cols:
    row_whatif = row.copy()
    for c in ot_cols:
        # Flip 0/1 to simulate changing overtime status
        row_whatif[c] = 1.0 - row_whatif[c]
    new_prob = model_rf.predict_proba(row_whatif)[0, 1]
    print("After flipping overtime indicator(s):", round(new_prob, 4))
    print("Delta:", round(new_prob - base_prob, 4))
else:
    # Fallback: nudge first numeric column slightly
    num_cols_row = row.select_dtypes(include=[np.number]).columns
    if len(num_cols_row):
        c0 = num_cols_row[0]
        row_alt = row.copy()
        row_alt[c0] = row_alt[c0] * 1.1
        print("No OverTime column — nudging first numeric feature:", c0)
        print("P(leave) after +10% on", c0, ":", round(model_rf.predict_proba(row_alt)[0, 1], 4))""")

md("""## 13 · Conclusion

### Takeaways

1. **Random Forest** combines **many** decision trees trained on **bootstrap** data with **random feature subsets** at splits, then **votes** — typically **lower variance** than one deep tree and often **better generalization** (test accuracy, smaller train–test gap).  
2. **Logistic regression** is a strong **linear** baseline; **decision trees** are **interpretable** but **overfit** when deep.  
3. **Feature importances** and **risk scores** support **discussion**; **causality** and **fairness** require extra work beyond this notebook.

### Limitations

- **Observational** data — associations are not **causal**.  
- **Deployment** needs governance, **bias** review, and possibly **probability calibration**.  
- **Hyperparameters** are set for a clear comparison, not for maximum Kaggle-style score.

### References

1. L. Breiman (2001). *Random Forests.* Machine Learning 45:5–32.  
2. F. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.  
3. IBM HR Analytics Employee Attrition (educational dataset).

---

**End of notebook.**""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
