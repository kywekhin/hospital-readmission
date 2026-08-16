# Hospital Readmission Risk Predictor

A machine learning project that predicts 30-day hospital readmission risk for diabetic patients, delivered as an easy-to-use web application with insight on the inputted data point.

🔗 **[Live Dashboard](https://diabetic-patient-readmission.onrender.com/)**

---

## The Problem

For diabetic patients, false positives on readmission risk causes unnecessary medical charges to a patient and false negatives may lead to worsening complications from diabetes. The challenge is that clinicians at discharge often lack a quick, data-driven way to flag which patients need close follow-up. This tool addresses that gap.

---

## What It Does

A clinician fills in 11 patient fields at the point of discharge and receives:

- A **risk tier** (Low / Medium / High) based on a tuned probability threshold
- A **population distribution chart** showing where the patient falls relative to ~20,000 test cases
- A **feature importance chart** showing the top 10 predictors driving the model
- A plain-language summary for non-technical clinical staff

---

## Dataset

[Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — UCI Machine Learning Repository

- **101,766 encounters** across 130 US hospitals
- **50 features** covering patient demographics, diagnoses, medications, and lab results
- Target: readmitted within 30 days (binary — 11% positive rate)
- Original authors: Strack, DeShazo, Gennings, Olmo, Ventura, Cios, and Clore

> The raw CSV is not committed to this repo due to size. Download it from the UCI link above and place it at `data/diabetic_data.csv` before running `train.py`.

---

## Methodology

### Data Cleaning
- Replaced `'?'` placeholder strings with `NaN` 
- Dropped columns with >40% missing values (`weight`, `max_glu_serum`, `medical_specialty`), with exceptions where cross-tabulation revealed MAR (Missing At Random) patterns. In those cases `'Unknown'` was retained as a meaningful category
- Dropped near-constant columns (>90% single value) identified by a natural gap in the dominance distribution between `glyburide` (89.5%) and `pioglitazone` (92.8%)
- Dropped `race` to avoid encoding potential systemic bias as a predictive signal
- Reduced ~700 unique ICD-9 diagnosis codes to 19 clinical categories using standard code range mapping

### Feature Engineering
- `A1Cresult` converted to binary (was the test ordered at all?) — consistent with Strack et al.'s finding that A1C *measurement* is more predictive than the result value itself
- Age brackets label-encoded ordinally (0–9)
- Medication dosage changes encoded as 0–3 (No/Steady/Down/Up)
- Diagnosis columns one-hot encoded (57 dummy columns across diag_1/2/3)

### Class Imbalance
The target variable is heavily imbalanced (~11% positive). Two strategies were combined:
- `class_weight='balanced'` in the Random Forest to penalise minority class errors during training
- **SMOTE** (Synthetic Minority Oversampling Technique) on the training set only — the test set retains the real-world distribution

### Train/Test Split
Split was performed at the **patient level** (on `patient_nbr`), not the row level, to prevent the same patient's multiple visits from appearing in both train and test sets to prevent data leakage.

### Model
- **Random Forest Classifier** was chosen for robustness to mixed feature types, resistance to multicollinearity, and built-in feature importance
- `n_estimators=50`, `max_depth=10` (depth constraint reduces file size from ~288MB to manageable without meaningful accuracy loss)

### Evaluation
| Metric | Value |
|---|---|
| AUC-ROC | 0.61 |
| Recall (class 1, threshold 0.20) | 0.40 |
| Precision (class 1, threshold 0.20) | 0.16 |
| F1 (class 1, threshold 0.20) | 0.22 |

The default 0.50 threshold produced near-zero recall on the minority class. Threshold was tuned to **0.20** by checking F1 values for various thresholds. AUC of 0.61 is above random (0.50) but below strong commercial models (~0.70), which is consistent with the documentation on this dataset. Lack of personal but important data (vitals, labs, lifestyle) prevents being able to learn on what could've been strong predictors. The threshold was chosen to maximise recall (catching actual readmissions), which trades off precision so roughly 84% of flagged patients will not be readmitted but in the trade off between extra cost of staying in the hospitals and missing a potentially lethal complication on a discharged patient, I have chosen to err on the safe side. 


### Risk Tiers
| Tier | Probability |
|---|---|
| 🟢 Low | < 10% |
| 🟡 Medium | 10–21% |
| 🔴 High | > 21% |

---
