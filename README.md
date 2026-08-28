# Credit Card Fraud Detection

Exploratory analysis and decision-tree baselines on the Kaggle credit card fraud dataset, with a focus on the two things that make the problem hard: extreme class imbalance (0.17% fraud) and heavy-tailed, unnormalized features. Compares plain training against SMOTE and Borderline-SMOTE oversampling.

## Data

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud). 284,807 transactions over two days, 492 of them fraudulent. Features are 28 PCA components (`V1`–`V28`) plus raw `Time` and `Amount`.

The CSV is 150 MB and is not included. Download `creditcard.csv` from Kaggle and place it in the repo root.

## What the notebook does

**Feature analysis**
- Projects `V1`–`V28` to 3-D with incremental PCA to see how separable fraud is.
- Time of day: fraud is over-represented during the low-volume night hours (roughly hours 1–8 and 24–32 of the 48-hour window).
- Amount: fraud averages $122 vs. $88 for legitimate transactions, with the difference concentrated in the upper 40% of the distribution.

**Preprocessing**
- `Time` → hour of day → `StandardScaler`
- `Amount` → `RobustScaler` (IQR-based, so the long tail does not dominate)
- 60/40 train/test split so the test set keeps enough fraud cases (199) for stable metrics

**Models**: `DecisionTreeClassifier`, alone and inside `imblearn` pipelines with `SMOTE` and `BorderlineSMOTE`.

## Results

Test set: 113,923 transactions, 199 fraudulent. Metrics are for the fraud class.

| Model | Precision | Recall | F1 | False positives | Missed fraud |
|---|---|---|---|---|---|
| Decision tree | 0.77 | 0.73 | 0.75 | 43 | 53 |
| Decision tree + SMOTE | 0.43 | 0.77 | 0.55 | 206 | 46 |
| Decision tree + Borderline-SMOTE | 0.77 | 0.70 | 0.73 | 42 | 60 |

Oversampling buys a few extra caught frauds at a steep cost in false positives (SMOTE roughly quintuples them), and Borderline-SMOTE is a wash. A single tree tops out around 0.75 F1 on this data; ensembles are the natural next step.

## Running it

```bash
pip install -r requirements.txt
# place creditcard.csv in this directory
jupyter notebook fraud_detection.ipynb
```

## Layout

```
fraud_detection.ipynb   analysis, preprocessing, training, evaluation
viz_util.py             2-D/3-D scatter helper and a timing context manager
requirements.txt
```
