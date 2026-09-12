# Credit Card Fraud Detection

Exploratory analysis and decision-tree baselines on the Kaggle credit card fraud dataset, with a focus on the two things that make the problem hard: extreme class imbalance (0.17% fraud) and heavy-tailed, unnormalized features. Compares plain training against SMOTE and Borderline-SMOTE oversampling, then asks how much ensembles (voting, bagging, random forests, boosting, stacking) buy on the same split.

## Data

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud). 284,807 transactions over two days, 492 of them fraudulent. Features are 28 PCA components (`V1`–`V28`) plus raw `Time` and `Amount`.

The CSV is 150 MB and is not included. Download `creditcard.csv` from Kaggle and place it in the repo root.

## What the notebooks do

### Part 1: `fraud_detection.ipynb`

**Feature analysis**
- Projects `V1`–`V28` to 3-D with incremental PCA to see how separable fraud is.
- Time of day: fraud is over-represented during the low-volume night hours (roughly hours 1–8 and 24–32 of the 48-hour window).
- Amount: fraud averages $122 vs. $88 for legitimate transactions, with the difference concentrated in the upper 40% of the distribution.

**Preprocessing**
- `Time` → hour of day → `StandardScaler`
- `Amount` → `RobustScaler` (IQR-based, so the long tail does not dominate)
- 60/40 train/test split so the test set keeps enough fraud cases (199) for stable metrics

**Models**: `DecisionTreeClassifier`, alone and inside `imblearn` pipelines with `SMOTE` and `BorderlineSMOTE`.

### Part 2: `ensemble_models.ipynb`

Same preprocessing and 60/40 split, then four ensemble families from scikit-learn:

- **Voting** (hard and soft) over logistic regression, Gaussian naive Bayes, and a decision tree, with each sub-classifier also scored alone.
- **Bagging** of logistic regression under four settings (10 vs. 30 estimators, 20% vs. 60% sample fraction, 50% feature subsampling), plus **random forests** with 40 and 100 trees against a single tree.
- **Boosting** with decision stumps: AdaBoost (50 rounds) and gradient boosting (40 rounds).
- **Stacking** of a random forest, logistic regression, and a linear SVM under a logistic-regression meta-classifier.

## Results

Test set: 113,923 transactions, 199 fraudulent. Metrics are for the fraud class.

| Model | Precision | Recall | F1 | False positives | Missed fraud |
|---|---|---|---|---|---|
| Decision tree | 0.77 | 0.73 | 0.75 | 43 | 53 |
| Decision tree + SMOTE | 0.43 | 0.77 | 0.55 | 206 | 46 |
| Decision tree + Borderline-SMOTE | 0.77 | 0.70 | 0.73 | 42 | 60 |

Oversampling buys a few extra caught frauds at a steep cost in false positives (SMOTE roughly quintuples them), and Borderline-SMOTE is a wash. A single tree tops out around 0.75 F1 on this data, which motivates the ensembles below.

**Ensembles** (same test set; the decision tree is re-trained here, so its numbers differ slightly from the table above):

| Model | Precision | Recall | F1 | False positives | Missed fraud |
|---|---|---|---|---|---|
| Logistic regression | 0.89 | 0.59 | 0.71 | 15 | 81 |
| Gaussian naive Bayes | 0.06 | 0.83 | 0.12 | 2,396 | 33 |
| Decision tree | 0.75 | 0.74 | 0.75 | 48 | 52 |
| Hard voting (LR + NB + tree) | 0.79 | 0.76 | 0.78 | 41 | 47 |
| Soft voting (LR + NB + tree) | 0.81 | 0.75 | 0.78 | 36 | 49 |
| Bagged LR, 10 × 20% samples | 0.89 | 0.57 | 0.70 | 14 | 85 |
| Bagged LR, 30 × 20% samples | 0.88 | 0.58 | 0.70 | 15 | 84 |
| Bagged LR, 10 × 60% samples | 0.89 | 0.57 | 0.69 | 14 | 86 |
| Bagged LR, 10 × 20% samples, 50% features | 0.88 | 0.53 | 0.66 | 15 | 94 |
| **Random forest, 40 trees** | **0.94** | **0.76** | **0.84** | 10 | 48 |
| Random forest, 100 trees | 0.94 | 0.76 | 0.84 | 9 | 48 |
| AdaBoost (stumps) | 0.78 | 0.66 | 0.72 | 37 | 67 |
| Gradient boosting (stumps) | 0.89 | 0.60 | 0.72 | 15 | 79 |
| Stacking (RF + LR + linear SVM → LR) | 0.95 | 0.66 | 0.78 | 7 | 68 |

Random forests are the clear winner: 0.84 F1 with only 9-10 false positives among 113,724 legitimate transactions, and 100 trees add nothing over 40. Voting lifts three mediocre models to 0.78 F1. Bagging a linear model does nothing (all settings land at or below plain logistic regression), which is expected since bagging only averages out variance. Stacking has the best precision but lower recall than the forest, and is the most expensive to train.

## Running it

```bash
pip install -r requirements.txt
# place creditcard.csv in this directory
jupyter notebook fraud_detection.ipynb   # part 1: analysis and single-tree baselines
jupyter notebook ensemble_models.ipynb   # part 2: ensembles
```

Part 2 takes roughly 8 minutes on a laptop; the stacking ensemble and gradient boosting are the slow cells.

## Layout

```
fraud_detection.ipynb   analysis, preprocessing, single decision tree with and without SMOTE
ensemble_models.ipynb   voting, bagging, random forest, boosting, stacking on the same split
viz_util.py             2-D/3-D scatter helper, timing context manager, evaluation printer
requirements.txt
```
