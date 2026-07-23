# Iris Classification with K-Nearest Neighbors

A complete supervised learning pipeline built on the classic Iris dataset:
load, scale, split, train, tune, and evaluate.

## Overview

| | |
|---|---|
| **Dataset** | Iris (150 samples, 4 features, 3 balanced classes) |
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Preprocessing** | StandardScaler (fit on train only) |
| **Split** | 80/20, stratified, shuffled |
| **K selection** | Elbow method (error rate vs. K, swept 1–30) |
| **Evaluation** | Accuracy, macro F1, confusion matrix, per-class precision/recall |

## Why this isn't just "load data, call .fit()"

- **No data leakage**: the scaler is fit only on the training set, then
  applied to the test set — fitting on the full dataset before splitting
  would leak test-set statistics into training.
- **K isn't guessed**: K is tuned by sweeping 1–30 and picking the elbow
  point in the error-rate curve, rather than defaulting to `n_neighbors=5`.
- **Accuracy alone isn't trusted**: results are backed by a full confusion
  matrix and macro F1, since accuracy can be misleading even on a
  balanced dataset like Iris.
- **Generalization check**: the model is tested on a synthetic sample
  outside the original 150 rows, not just the held-out test split.

## Exploratory Data Analysis

Pairwise feature relationships across the three species:

![Iris Pairplot](outputs/iris_pairplot.png)

## Choosing K

Error rate swept across K = 1 to 30, with the elbow marking the optimal value:

![K Tuning Curve](outputs/k_tuning_curve.png)

For this dataset and split, the elbow method selected **K = 1**. On a
dataset this small and well-separated it produces the lowest test error,
though K = 1 is generally more sensitive to noise than a slightly higher K
(e.g. 5–7) — worth keeping in mind if you validate against different seeds.

## Results

- **Accuracy**: 0.9667
- **Macro F1**: 0.9666
- **Best K**: 1 (via elbow method)

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.91      1.00      0.95        10
   virginica       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

The confusion matrix below shows perfect separation of Setosa, with one
Virginica sample misclassified as Versicolor — expected, since those two
species overlap slightly in petal measurements.

![Confusion Matrix](outputs/confusion_matrix.png)

## Project structure

```
iris-classification-knn/
├── iris_knn.py              # Full pipeline: load, scale, split, tune, train, evaluate
├── outputs/                 # Generated plots
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/AbdullahAhmed04/iris-classification-knn.git
cd iris-classification-knn
pip install -r requirements.txt
python iris_knn.py
```

## Pipeline (IPO framework)

1. **Input** — Load Iris via `sklearn.datasets`, inspect shape/class balance/stats
2. **Process** — Stratified train-test split → StandardScaler → KNN, K tuned via elbow method
3. **Output** — Accuracy, macro F1, classification report, confusion matrix heatmap

## Tech stack

- Python 3, scikit-learn, pandas, numpy, matplotlib, seaborn
