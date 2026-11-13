# Parkinson's Disease Classification from Voice Measurements

A classifier that separates people with Parkinson's disease (PD) from healthy controls using 22 acoustic measurements of a sustained vowel. The focus is an **honest evaluation**: every recording from the same person stays on one side of every split, so the model is scored on people it has never heard.

> This is a learning and research project on a small public dataset (32 people). It is not a diagnostic tool.

## Results at a glance

|                                                  | Logistic Regression (final) | Always-predict-PD baseline |
| ------------------------------------------------ | --------------------------- | -------------------------- |
| ROC-AUC, mean ± std over 5 random subject splits | **0.779 ± 0.069**           | 0.50                       |
| Balanced accuracy, same 5 splits                 | **0.706 ± 0.046**           | 0.50                       |
| Healthy recall (specificity)                     | 0.692 ± 0.092               | 0.00                       |
| PD recall (sensitivity)                          | 0.721 ± 0.035               | 1.00                       |

Across the 5 splits, ROC-AUC ranged from 0.667 to 0.846. A single split can therefore look much better or worse than the method really is, which is why the mean and spread are reported.

## Data

[UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) (Little et al., 2007). Facts below were verified directly from the file, not taken from documentation.

| Fact                       | Value                                                                           |
| -------------------------- | ------------------------------------------------------------------------------- |
| Recordings (rows)          | 195                                                                             |
| People (subjects)          | 32: 24 with PD, 8 healthy                                                       |
| Recordings per person      | 6 (29 people) or 7 (3 people)                                                   |
| Class balance (recordings) | 147 PD / 48 healthy (75% / 25%)                                                 |
| Features                   | 22 acoustic measures (pitch, jitter, shimmer, noise ratios, nonlinear measures) |
| Missing values             | None                                                                            |

The subject ID is parsed from the `name` column (`phon_R01_S01_1` → subject `S01`).

## The key decision: evaluate by person, not by recording

Each person was recorded about 6 times, and their recordings are near-copies. A random train/test split puts the same person on both sides, so a model can recognise the **voice** instead of the **disease** and report inflated scores.

Every split in this project is therefore grouped by subject (`StratifiedGroupKFold`). The effective sample size is about 32 people, not 195 rows.

## Method

**Preprocessing** (all fitted inside training folds only, via a scikit-learn `Pipeline`):

1. Removed `Jitter:DDP` and `Shimmer:DDA`: each is exactly 3× another feature (`MDVP:RAP`, `Shimmer:APQ3`), verified by their ratio.
2. `PowerTransformer` (Yeo-Johnson) to reduce strong right skew (e.g. NHR skew 4.2 → under 1).
3. `RobustScaler` (median and IQR), because several features have many outliers.
4. `SelectKBest` (ANOVA F-test) feature selection, with `k` ∈ {8, 12, all} tuned.

**Models:** Logistic Regression, SVM (RBF kernel), K-Nearest Neighbors, Random Forest. Class imbalance handled with `class_weight="balanced"` where supported.

**Evaluation:** nested cross-validation.

- Outer loop: 5 subject-grouped folds that measure performance on unseen people.
- Inner loop: 3 subject-grouped folds that tune hyperparameters (by ROC-AUC) using only the outer training people.

**Metrics:** accuracy, precision, recall (both classes), F1, balanced accuracy, ROC-AUC, PR-AUC for the healthy class, and a subject-level majority vote.

## Model comparison

Mean ± std across the 5 outer folds (seed 42).

| Model               | Accuracy      | Balanced acc.     | PD recall     | Healthy recall | ROC-AUC           | PR-AUC (healthy)  |
| ------------------- | ------------- | ----------------- | ------------- | -------------- | ----------------- | ----------------- |
| Logistic Regression | 0.763 ± 0.145 | **0.767 ± 0.097** | 0.750 ± 0.220 | 0.783 ± 0.201  | **0.852 ± 0.131** | **0.678 ± 0.269** |
| SVM (RBF)           | 0.384 ± 0.242 | 0.578 ± 0.126     | 0.223 ± 0.306 | 0.933 ± 0.149  | 0.830 ± 0.105     | 0.642 ± 0.208     |
| KNN                 | 0.791 ± 0.106 | 0.619 ± 0.178     | 0.921 ± 0.112 | 0.317 ± 0.375  | 0.736 ± 0.140     | 0.488 ± 0.262     |
| Random Forest       | 0.706 ± 0.124 | 0.593 ± 0.104     | 0.769 ± 0.172 | 0.417 ± 0.283  | 0.755 ± 0.115     | 0.574 ± 0.210     |
| Dummy (always PD)   | 0.75          | 0.50              | 1.00          | 0.00           | 0.50              | about 0.25        |

Seed 42 was the most favourable of the 5 seeds tested for Logistic Regression, so the 5-seed figures in "Results at a glance" are the ones to quote.

**What the comparison shows**

- **Accuracy is misleading here.** KNN has the highest accuracy, but it mostly predicts PD and correctly identifies only about a third of healthy recordings.
- **SVM ranks well within a fold but thresholds badly.** Its per-fold ROC-AUC (0.83) is close to Logistic Regression's, but tuning chose very small `C` values, and the default decision threshold then labels most people healthy. Tuning on a threshold-free metric does not guarantee a usable threshold.
- **Per-fold AUC is the fairer summary.** SVM's AUC computed on all 195 pooled predictions is only 0.74, because each fold's model produces scores on a different scale; pooling them mixes scales. Per-fold AUC avoids this.
- **Logistic Regression was chosen** as the final model. It is the most balanced across both classes and the simplest to explain. Its lead over SVM on ranking (ROC-AUC) is within the fold-to-fold noise, so this is a judgement, not a statistically proven win.

**Subject level (Logistic Regression, seed 42):** 7 of 8 healthy people and 20 of 24 PD people classified correctly by majority vote over their recordings (3 tied votes, counted as PD).

## What drives the predictions

The final model uses strong regularization (`C = 0.001`) and keeps all 20 features. Its coefficients are small and similar in size, so it acts close to a weighted sum of the voice measures.

- **Push towards PD:** higher `spread1`, `PPE`, `spread2` (largest weights), plus higher jitter, shimmer and noise measures.
- **Push towards healthy:** higher `HNR` (harmonics-to-noise ratio) and higher fundamental-frequency measures (`Fo`, `Flo`, `Fhi`).
- **Permutation importance** on held-out folds agrees on the top three: `spread1`, `PPE`, `spread2`. Most other features show near-zero importance individually, because they are highly correlated: shuffling one leaves its correlated partners intact.

## Limitations

- **Small sample:** 32 people, only 8 healthy. One healthy person changes healthy recall by about 0.125.
- **No external test set:** one recording setup and one session. Results may not transfer to other microphones, languages or hospitals.
- **No stage information:** the data cannot support claims about early-stage detection.
- **Possible confounders:** age and sex are not in the data, so the model may partly learn those group differences.
- **Model chosen after seeing outer-fold results** (4 candidates), which adds a small optimistic bias.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/Aadiie/Parkinson_Prediction.git
cd Parkinson_Prediction
uv sync
uv run jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`: data checks, EDA, split strategy, preprocessing checks, baselines.
2. `notebooks/02_modeling.ipynb`: model comparison, seed stability, feature importance, final model. The final model is saved to `models/` (not committed; it is regenerated by the notebook).

## Project structure

```
├── data/raw/parkinsons.data      # UCI dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── data/load.py              # load data, extract subject IDs, sanity checks
│   ├── models/pipeline.py        # preprocessing + selection + model, hyperparameter grids
│   ├── models/train_final.py     # tune on all subjects, save final pipeline
│   └── evaluation/
│       ├── nested_cv.py          # subject-grouped nested cross-validation
│       └── metrics.py            # recording-, fold- and subject-level metrics
├── pyproject.toml / uv.lock      # dependencies
└── README.md
```

## Reference

Little, M. A., McSharry, P. E., Roberts, S. J., Costello, D. A. E., & Moroz, I. M. (2007). Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection. _BioMedical Engineering OnLine_, 6(23).
