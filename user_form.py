from dash import Dash, dcc, html, Input, Output, State, callback
import joblib
import pandas as pd
import numpy as np


# loading in the captured model and column names from the notebook
rf_model = joblib.load('models/rf_smote.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
defaults = joblib.load('models/defaults.pkl')


# creating diagnoses column labels by cutting off the prefix in the col name

def clean_label(label):
    if label == 'E_codes': return 'Externally caused injury'
    if label == 'V_codes': return 'Pre-existing potential health hazard'
    return label

diag_1_categories = [clean_label(col.replace('diag_1_', '')) for col in feature_columns if col.startswith('diag_1_')]
diag_2_categories = [clean_label(col.replace('diag_2_', '')) for col in feature_columns if col.startswith('diag_2_')]
diag_3_categories = [clean_label(col.replace('diag_3_', '')) for col in feature_columns if col.startswith('diag_3_')]


app = Dash(__name__)

# html body with components
app.layout = html.Div([
    html.H1("Hospital Readmission Risk Predictor"),

    html.Label("Age"),
    dcc.Dropdown(id='age',
                options=['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                value='[50-60)'),

    html.Label("Primary Diagnosis"),
    dcc.Dropdown(id='diag_1', options=diag_1_categories, value=diag_1_categories[0]),

    html.Label("Secondary Diagnosis"),
    dcc.Dropdown(id='diag_2', options=diag_2_categories, value=diag_2_categories[0]),

    html.Label("Additional Diagnosis"),
    dcc.Dropdown(id='diag_3', options=diag_3_categories, value=diag_3_categories[0]),

    html.Label("Was A1C Measured?"),
    dcc.Dropdown(id='A1Cresult', options=['No', 'Yes'], value='No'),

    html.Label("Number of Lab Procedures"),
    dcc.Input(id='num_lab_procedures', type='number', min=0, max=132, value=8),

    html.Label("Number of Medications"),
    dcc.Input(id='num_medications', type='number', min=0, max=81, value=8),

    html.Label("Number of Diagnoses"),
    dcc.Input(id='number_diagnoses', type='number', min=0, max=16, value=4),

    html.Label("Number of Inpatient Visits"),
    dcc.Input(id='number_inpatient', type='number', min=0, max=21, value=0),

    html.Label("Metformin"),
    dcc.Dropdown(id='metformin', options=['No change in dosage', 'Dosage remained steady', 'Dosage went down', 'Dosage went up'], value='No change in dosage'),

    html.Label("Insulin"),
    dcc.Dropdown(id='insulin', options=['No change in dosage', 'Dosage remained steady', 'Dosage went down', 'Dosage went up'], value='No change in dosage'),

    html.Button('Predict', id='predict-btn', n_clicks=0),
    html.Div(id='prediction-output')
])

# first parameter is the button id to insert at, second is the value to replace with
@callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('diag_1', 'value'),
    State('diag_2', 'value'),
    State('diag_3', 'value'),
    State('A1Cresult', 'value'),
    State('num_lab_procedures', 'value'),
    State('num_medications', 'value'),
    State('number_diagnoses', 'value'),
    State('number_inpatient', 'value'),
    State('metformin', 'value'),
    State('insulin', 'value'),
    prevent_initial_call=True
)
def predict(n_clicks, age, diag_1, diag_2, diag_3, a1c,
            num_lab, num_med, num_diag, num_inpatient, metformin, insulin):
    
    if n_clicks == 0: return ""

    inputs_list = defaults.copy()
    # we are using copy to get the col names copied over

    inputs_list['age'] = age
    inputs_list['A1Cresult'] = a1c
    inputs_list['num_lab_procedures'] = num_lab
    inputs_list['num_medications'] = num_med
    inputs_list['number_diagnoses'] = num_diag
    inputs_list['number_inpatient'] = num_inpatient
    inputs_list['metformin'] = metformin
    inputs_list['insulin'] = insulin

    age_originals = ['[90-100)', '[80-90)', '[70-80)', '[60-70)', '[50-60)', '[40-50)', '[30-40)', '[20-30)', '[10-20)', '[0-10)']
    age_encoded = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    for i in range(10):
        if age == age_originals[i]:
            inputs_list['age'] = age_encoded[i]
            break

    if a1c == 'No':
        inputs_list['A1Cresult'] = 0
    else: inputs_list['A1Cresult'] = 1

    measures_original = ['No change in dosage', 'Dosage remained steady', 'Dosage went down', 'Dosage went up']
    measures_encoded = [0, 1, 2, 3]

    for i in range(4):
            if metformin == measures_original[i]:
                inputs_list['metformin'] = measures_encoded[i]
            if insulin == measures_original[i]:
                inputs_list['insulin'] = measures_encoded[i]

    inputs_list[f'diag_1_{diag_1}'] = 1
    inputs_list[f'diag_2_{diag_2}'] = 1
    inputs_list[f'diag_3_{diag_3}'] = 1

    if diag_1 == 'Pre-exsiting potential health hazard': inputs_list['diag_1_V_codes'] = 1
    if diag_2 == 'Pre-exsiting potential health hazard': inputs_list['diag_2_V_codes'] = 1
    if diag_3 == 'Pre-exsiting potential health hazard': inputs_list['diag_3_V_codes'] = 1

    if diag_1 == 'Externally caused injury': inputs_list['diag_1_E_codes'] = 1
    if diag_2 == 'Externally caused injury': inputs_list['diag_2_E_codes'] = 1
    if diag_3 == 'Externally caused injury': inputs_list['diag_3_E_codes'] = 1

    for col in feature_columns:
        if col not in inputs_list:
            inputs_list[col] = 0

    input_df = pd.DataFrame([inputs_list])[feature_columns]

    proba = rf_model.predict_proba(input_df)[0][1]

    if proba < 0.1:
        return "Low Risk"
    elif proba < 0.21:
        return "Medium Risk"
    else:
        return "High Risk"


if __name__ == '__main__':
    app.run(debug=True)