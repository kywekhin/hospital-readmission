import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib, os

# Load data
df = pd.read_csv('data/diabetic_data.csv')

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Drop high-missing columns
df = df.drop(['weight', 'max_glu_serum', 'medical_specialty'], axis=1)

# Fill missing values
cols_to_fill = ['payer_code', 'race', 'diag_1', 'diag_2', 'diag_3']
df[cols_to_fill] = df[cols_to_fill].replace(np.nan, 'Unknown')

# Drop near-constant columns (>90% one value)
for col in df.columns:
    if df[col].value_counts(normalize=True).max() > 0.90:
        df = df.drop(col, axis=1)

# Bin ICD-9 codes into 19 clinical categories
def bin_codes(code):
    if code.startswith('V'): return 'V_codes'
    elif code.startswith('E'): return 'E_codes'
    elif code.split('.')[0].isnumeric():
        num = int(code.split('.')[0])
        ranges = [
            (1, 139, 'Infectious and parasitic diseases'),
            (140, 239, 'Neoplasms'),
            (240, 279, 'Endocrine, nutritional and metabolic diseases, and immunity disorders'),
            (280, 289, 'Diseases of the blood and blood-forming organs'),
            (290, 319, 'Mental disorders'),
            (320, 389, 'Diseases of the nervous system and sense organs'),
            (390, 459, 'Diseases of the circulatory system'),
            (460, 519, 'Diseases of the respiratory system'),
            (520, 579, 'Diseases of the digestive system'),
            (580, 629, 'Diseases of the genitourinary system'),
            (630, 679, 'Complications of pregnancy, childbirth, and the puerperium'),
            (680, 709, 'Diseases of the skin and subcutaneous tissue'),
            (710, 739, 'Diseases of the musculoskeletal system and connective tissue'),
            (740, 759, 'Congenital anomalies'),
            (760, 779, 'Certain conditions originating in the perinatal period'),
            (780, 799, 'Symptoms, signs, and ill-defined conditions'),
            (800, 999, 'Injury and poisoning')]
        for low, high, label in ranges:
            if low <= num <= high: return label
    else: return 'Unknown'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(bin_codes)

# Binary target: readmitted within 30 days = 1
df['readmitted'] = df['readmitted'].replace({'NO': 0, '>30': 0, '<30': 1}).astype(int)

# Drop columns not useful for prediction
df = df.drop(['race', 'gender', 'payer_code'], axis=1)

# A1Cresult: was it measured at all? (binary)
df['A1Cresult'] = df['A1Cresult'].replace(to_replace=['>7', '>8', 'Norm'], value=1)
df['A1Cresult'] = df['A1Cresult'].fillna(0)

# Ordinal encode age brackets
age_originals = ['[90-100)', '[80-90)', '[70-80)', '[60-70)', '[50-60)',
                 '[40-50)', '[30-40)', '[20-30)', '[10-20)', '[0-10)']
age_encoded = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
df['age'] = df['age'].replace(age_originals, age_encoded).astype(int)

# Encode binary and ordinal medication columns
df['change'] = df['change'].replace(['Ch', 'No'], [1, 0]).astype(int)
df['diabetesMed'] = df['diabetesMed'].replace(['Yes', 'No'], [1, 0]).astype(int)
df[['metformin', 'glipizide', 'glyburide', 'insulin']] = (
    df[['metformin', 'glipizide', 'glyburide', 'insulin']]
    .replace(['No', 'Steady', 'Down', 'Up'], [0, 1, 2, 3])
    .astype(int)
)

# One-hot encode diagnosis columns
df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], dtype=int)

# Patient-level split to prevent data leakage
unique_patients = df['patient_nbr'].unique()
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
train_df = df[df['patient_nbr'].isin(train_patients)]
test_df  = df[df['patient_nbr'].isin(test_patients)]

x_train = train_df.drop('readmitted', axis=1).drop(['patient_nbr', 'encounter_id'], axis=1)
y_train = train_df['readmitted']

x_test  = test_df.drop('readmitted', axis=1).drop(['patient_nbr', 'encounter_id'], axis=1)

# SMOTE to address 1:9 class imbalance
sm = SMOTE(random_state=42)
x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)

# Train with max_depth to constrain file size (unconstrained = ~288MB)
rf_smote = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_smote.fit(x_train_sm, y_train_sm)

# Test-set probabilities — saved for the dashboard distribution chart
y_proba_sm = rf_smote.predict_proba(x_test)[:, 1]

# Feature importances — saved for the dashboard importance chart
importances = pd.Series(rf_smote.feature_importances_, index=x_train.columns).sort_values(ascending=False)

# Save all artifacts
os.makedirs('models', exist_ok=True)
joblib.dump(rf_smote,                    'models/rf_smote.pkl')
joblib.dump(x_train.columns.tolist(),   'models/feature_columns.pkl')
joblib.dump(y_proba_sm,                 'models/y_proba_sm.pkl')
joblib.dump(importances,                'models/feature_importances.pkl')

# Default values for non-user-input columns (mode of training set)
non_input_cols = [col for col in x_train.columns if col not in [
    'age', 'num_lab_procedures', 'num_medications',
    'number_diagnoses', 'metformin', 'insulin', 'number_inpatient', 'A1Cresult'
] and not col.startswith('diag_')]
defaults = x_train[non_input_cols].mode().iloc[0].to_dict()
joblib.dump(defaults, 'models/defaults.pkl')

print("Training complete. Models saved.")
