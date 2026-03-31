# Employee Attrition — Random Forest

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3%2B-yellowgreen.svg)](https://scikit-learn.org/)

End-to-end **machine learning** project: predict **employee attrition** (whether someone leaves the company) using **Random Forest**, with **logistic regression** and **decision tree** models for comparison on the same train/test split. The notebook walks through **data prep**, **training**, **evaluation** (metrics, confusion matrices, ROC), **feature importance**, **predicted probabilities** (attrition risk), and a simple **what-if** example.

---

## Table of contents

- [Overview](#overview)
- [What this project includes](#what-this-project-includes)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Notebook outline](#notebook-outline)
- [Outputs and figures](#outputs-and-figures)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Regenerating the notebook](#regenerating-the-notebook)
- [Citation and references](#citation-and-references)
- [License](#license)

---

## Overview

| Item | Description |
|------|-------------|
| **Notebook** | `01_Random_Forest_Attrition_Complete.ipynb` |
| **Task** | Binary classification: `Attrition` = Yes / No |
| **Models** | **Random Forest** (main focus), **decision tree**, **logistic regression** (baselines) |
| **Workflow** | Load data → clean & encode → train/test split → fit models → metrics & plots → feature importance → **risk scores** (`predict_proba`) → optional what-if |

All project files sit in the **repository root** except figures saved under **`plots/`**.

---

## What this project includes

- **Random Forest** explained in context: **bagging** (bootstrap samples), **random feature subsets** at splits, **majority voting**, and how that differs from a **single tree**.
- **Comparison** of train vs test accuracy (especially **overfitting** on a deep decision tree vs the forest).
- **Visuals** — RF flow diagram, sample trees, confusion matrices, ROC, feature importance bars, accuracy comparison.
- **Reproducible** setup: `requirements.txt`, bundled CSV, fixed random seed.
- Optional **`USE_SMALL_SAMPLE`** for a smaller stratified subset if you want a quicker run.

See **`GUIDE.md`** for a short section-by-section map and a run checklist.

---

## Repository structure

```
.
├── README.md
├── GUIDE.md                               # Quick section map & checklist
├── requirements.txt
├── build_master_notebook.py               # Regenerates the .ipynb from source
├── ibm_hr_attrition.csv
├── 01_Random_Forest_Attrition_Complete.ipynb
├── plots/                                 # PNG outputs (created when you run the notebook)
│   └── .gitkeep
└── .gitignore
```

---

## Dataset

| Property | Detail |
|----------|--------|
| **File** | `ibm_hr_attrition.csv` |
| **Rows** | ~1,470 (or ~500 with `USE_SMALL_SAMPLE = True`) |
| **Target** | `Attrition` — Yes / No |
| **Features** | Numeric and categorical HR fields (e.g. age, income, department, overtime, satisfaction, tenure) |
| **Source** | IBM HR Analytics **employee attrition** dataset (public release, common for analytics examples) |

The notebook drops ID and constant columns (`EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`) before modeling.

---

## Prerequisites

- Python **3.10+** (3.11 recommended)
- **pip** (or conda)
- **Jupyter** in the browser, or **VS Code / Cursor** with Jupyter support

---

## Installation

### 1. Clone or download

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

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

1. Activate your environment.
2. Open a terminal in the **repository root** (where `ibm_hr_attrition.csv` lives).
3. Start Jupyter or open **`01_Random_Forest_Attrition_Complete.ipynb`** in your editor.
4. **Restart & Run All**, or run cells from top to bottom.
5. For a **faster run**, in **section 2** set:

```python
USE_SMALL_SAMPLE = True
```

---

## Notebook outline

| § | Topic |
|---|--------|
| **1** | Introduction — goals; Random Forest focus; attrition as use case |
| **2** | Dataset — load, explore |
| **3** | Preprocessing — clean, encode, stratified split |
| **4** | Concepts — logistic regression, decision trees, Random Forest |
| **5** | Logistic regression (baseline) |
| **6** | Decision tree (baseline) — train vs test, overfitting |
| **7** | Random Forest — diagram, sample trees |
| **8** | Evaluation — metrics, confusion matrices, ROC |
| **9** | Feature importance |
| **10** | Model comparison — accuracy |
| **11** | Attrition risk score — probabilities |
| **12** | What-if example |
| **13** | Conclusion — limitations, references |

---

## Outputs and figures

Running the notebook saves PNGs under **`plots/`**, for example:

| File (examples) | Content |
|-------------------|---------|
| `fig_eda_*.png` | Exploratory plots |
| `fig_rf_diagram.png` | Random Forest flow |
| `fig_rf_sample_trees.png` | Sample trees from the forest |
| `fig_confusion_matrices.png` | Confusion matrices |
| `fig_roc.png` | ROC curves |
| `fig_feature_importance.png` | Feature importances |
| `fig_accuracy_comparison.png` | Accuracy comparison |
| `fig_risk_score_distribution.png` | Risk score distribution |

An interactive ROC line is also drawn in the notebook (hvPlot / Holoviews / Bokeh) when those packages are installed.

---

## Dependencies

See **`requirements.txt`**: `numpy`, `pandas`, `scipy`, `scikit-learn`, `jupyter`, `matplotlib`, `seaborn`, `hvplot`, `holoviews`, `bokeh`.

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| CSV not found | Run Jupyter / IDE with **working directory** = repo root. |
| Import errors | `pip install -r requirements.txt` in the **same** environment as the notebook kernel. |
| hvPlot fails | Static ROC is still saved as `plots/fig_roc.png`; you can skip the interactive cell. |
| Slow or low memory | Set `USE_SMALL_SAMPLE = True` in section 2. |

---

## Regenerating the notebook

The notebook is built from **`build_master_notebook.py`**:

```bash
python build_master_notebook.py
```

This overwrites **`01_Random_Forest_Attrition_Complete.ipynb`**.

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

---

<p align="center">
  <b>Questions or improvements?</b> Open an issue or submit a pull request on GitHub.
</p>
