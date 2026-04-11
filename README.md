# Employee Attrition — Random Forest

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3%2B-yellowgreen.svg)](https://scikit-learn.org/)

---

## What this repository is

This is a **single-notebook** machine learning walkthrough. You get:

- A **CSV dataset** of employee records (`ibm_hr_attrition.csv`).
- One Jupyter notebook (**`01_Random_Forest_Attrition_Complete.ipynb`**) that goes from raw data to trained models, evaluation plots, and interpretation.
- A **`requirements.txt`** listing Python packages so you can recreate the environment.

You do **not** need other scripts or notebooks—everything runs inside that one `.ipynb` file.

---

## What you will learn (in plain terms)

| Topic | What the notebook does |
|--------|-------------------------|
| **Problem** | Predict whether an employee is likely to **leave** (`Attrition` = Yes vs No)—a standard HR analytics classification task. |
| **Why Random Forest** | It combines many **decision trees** (each trained on random subsets of data and features) and **votes**—usually more stable than one very deep tree, which can **overfit**. |
| **Baselines** | **Logistic regression** (linear, fast) and a **deep decision tree** (flexible, prone to overfitting) use the **same train/test split** so comparisons are fair. |
| **Class imbalance** | Most employees “stay,” so **accuracy** can look good even when the model rarely predicts “leave.” The notebook uses **class weights** and stresses **precision, recall, F1**, and **ROC-AUC**, not accuracy alone. |
| **Hyperparameters** | A **RandomizedSearchCV** pass tunes the Random Forest on the training fold; you see **tuned vs untuned** behavior. |
| **Interpretation** | **Feature importance** (association, not causation), **predicted probabilities** as a **risk score**, and a simple **what-if** on overtime—plus notes on **fairness** and responsible use. |

---

## Who this is for

- **Students** learning Random Forest and classification metrics on tabular data.
- **Practitioners** who want a **reproducible template** (same split, saved figures, clear sections).
- **Reviewers** who need to see **end-to-end** code and narrative in one place.

---

## Files in this folder (self-explanation)

| File | Purpose |
|------|---------|
| **`01_Random_Forest_Attrition_Complete.ipynb`** | The full project: data loading, EDA, preprocessing, models, evaluation, tuning, risk scores, conclusion. **Start here.** |
| **`ibm_hr_attrition.csv`** | The dataset; the notebook expects it in the **same directory** as the notebook (repository root). |
| **`requirements.txt`** | Python dependencies; install with `pip install -r requirements.txt` before running the notebook. |
| **`README.md`** | This file—how to run and what to expect. |
| **`.gitignore`** | Tells Git to ignore virtual environments and notebook checkpoints (not required to run the project). |

After you run the notebook, a **`plots/`** folder appears with **PNG figures** (EDA, ROC, confusion matrices, etc.). It is safe to delete `plots/`; the notebook recreates it.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Notebook outline (section map)](#notebook-outline-section-map)
- [Outputs and figures](#outputs-and-figures)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Citation and references](#citation-and-references)
- [License](#license)

---

## Prerequisites

- Python **3.10+** (3.11 works well)
- **pip** (or conda)
- **Jupyter**, **VS Code**, or **Cursor** with Jupyter support

---

## Installation

### 1. Get the code

Clone or download this repository and `cd` into the project folder (the folder that contains `ibm_hr_attrition.csv`).

### 2. Virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Jupyter kernel (optional)

```bash
python -m ipykernel install --user --name=ml_attrition --display-name="Python (ml_attrition)"
```

---

## How to run

1. Activate the environment where you installed `requirements.txt`.
2. Open the project folder so that **`ibm_hr_attrition.csv`** and **`01_Random_Forest_Attrition_Complete.ipynb`** sit in the **same** working directory (repository root).
3. Open the notebook in Jupyter or your editor.
4. Run **Kernel → Restart & Run All**, or execute cells **from top to bottom** once.

**Faster demo (optional):** in the notebook’s setup section, set `USE_SMALL_SAMPLE = True` to use a smaller stratified subset of rows.

---

## Notebook outline (section map)

| § | What happens |
|---|----------------|
| **1** | Introduction—why Random Forest and how it compares to a single tree and logistic regression. |
| **2** | Load data, quick exploration, EDA plots (balance, age/income, overtime). |
| **3** | Preprocessing—drop IDs/constants, one-hot encoding, **stratified** train/test split. |
| **4** | Concepts—linear vs tree models, overfitting, bagging and random features; short **Gini** intuition. |
| **5** | **Logistic regression** baseline (`class_weight` for imbalance). |
| **6** | **Deep decision tree**—shows train vs test gap (overfitting). |
| **7** | **Random Forest**—diagram, three **separate** sample-tree figures; **§7.1** hyperparameter search (`RandomizedSearchCV`). |
| **8** | **Evaluation**—metrics table, confusion matrices, ROC (static matplotlib; optional interactive hvPlot if installed). |
| **9** | **Feature importance**—interpretation (association ≠ causation). |
| **10** | **Model comparison**—train vs test accuracy bars. |
| **11** | **Risk score**—predicted probability of leaving. |
| **12** | **What-if** example (e.g., overtime flip)—local sensitivity, not causal proof. |
| **13** | **Conclusion**—takeaways, limitations, fairness / responsible use, references. |

---

## Outputs and figures

Running the notebook writes PNGs under **`plots/`**, for example:

| Pattern / file | Content |
|----------------|---------|
| `fig_eda_*.png` | Class balance, age/income, overtime-related views |
| `fig_rf_diagram.png` | Schematic of bootstrap → trees → vote |
| `fig_rf_sample_tree_1.png` … `fig_rf_sample_tree_3.png` | Three sample trees (one file each) |
| `fig_confusion_matrices.png` | Confusion matrices for all three models |
| `fig_roc.png` | ROC curves |
| `fig_feature_importance.png` | Feature importance bars |
| `fig_accuracy_comparison.png` | Test accuracy and train vs test comparison |
| `fig_risk_score_distribution.png` | Distribution of predicted leave probabilities |

If **holoviews / hvplot / bokeh** are installed, an extra **interactive** ROC view may appear in the notebook; if not, the static ROC file is enough.

---

## Dependencies

Declared in **`requirements.txt`** (core stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `jupyter`, `matplotlib`, `seaborn`; optional interactive stack: `hvplot`, `holoviews`, `bokeh`).

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| `FileNotFoundError` for the CSV | Set the notebook’s working directory to the **repository root** (where `ibm_hr_attrition.csv` lives). |
| Import errors | Run `pip install -r requirements.txt` in the **same** environment as the Jupyter kernel. |
| hvPlot / interactive ROC errors | Ignore them; the notebook still saves **`plots/fig_roc.png`**. |
| Slow run or low memory | Set `USE_SMALL_SAMPLE = True` in the notebook. |
| RandomizedSearchCV is slow | Normal on modest CPUs; reduce `n_iter` in the search cell if you need a quicker pass. |

---

## Citation and references

**Dataset:** IBM HR Analytics Employee Attrition (public release).

**Methods:**

1. L. Breiman (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
2. F. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.

**Example credit line:**

> IBM HR Analytics Employee Attrition dataset. Models implemented with scikit-learn. Random Forest after Breiman (2001).

---

## License

Use this project for **personal learning and research** unless you add your own license. The **IBM HR Analytics** dataset is subject to the terms of the source you obtained it from.
