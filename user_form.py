from dash import Dash, dcc, html, Input, Output, State, callback
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load model artifacts
rf_model             = joblib.load('models/rf_smote.pkl')
feature_columns      = joblib.load('models/feature_columns.pkl')
defaults             = joblib.load('models/defaults.pkl')
y_proba_population   = joblib.load('models/y_proba_sm.pkl')   # test-set probabilities for distribution chart
feature_importances  = joblib.load('models/feature_importances.pkl')

# Diagnosis label encoding and reversal
def clean_label(label):
    """Convert internal model column names to human-readable dropdown labels."""
    if label == 'E_codes': return 'Externally caused injury'
    if label == 'V_codes': return 'Pre-existing potential health hazard'
    return label

def reverse_label(label):
    """Convert dropdown label back to the model's internal column name."""
    if label == 'Pre-existing potential health hazard': return 'V_codes'
    if label == 'Externally caused injury': return 'E_codes'
    return label

diag_1_categories = [clean_label(col.replace('diag_1_', '')) for col in feature_columns if col.startswith('diag_1_')]
diag_2_categories = [clean_label(col.replace('diag_2_', '')) for col in feature_columns if col.startswith('diag_2_')]
diag_3_categories = [clean_label(col.replace('diag_3_', '')) for col in feature_columns if col.startswith('diag_3_')]

# Actual app features
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Hospital Readmission Risk Predictor"),
    html.P("Enter patient details at discharge to assess 30-day readmission risk."),

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

    html.Label("Was A1C Measured During This Admission?"),
    dcc.Dropdown(id='A1Cresult', options=['No', 'Yes'], value='No'),

    html.Label("Number of Lab Procedures"),
    dcc.Input(id='num_lab_procedures', type='number', min=0, max=132, value=8),

    html.Label("Number of Medications"),
    dcc.Input(id='num_medications', type='number', min=0, max=81, value=8),

    html.Label("Number of Diagnoses"),
    dcc.Input(id='number_diagnoses', type='number', min=0, max=16, value=4),

    html.Label("Number of Previous Inpatient Visits"),
    dcc.Input(id='number_inpatient', type='number', min=0, max=21, value=0),

    html.Label("Metformin"),
    dcc.Dropdown(id='metformin',
                 options=[
                     {'label': 'Not Prescribed',    'value': 'No change in dosage'},
                     {'label': 'No Change',          'value': 'Dosage remained steady'},
                     {'label': 'Dosage Decreased',   'value': 'Dosage went down'},
                     {'label': 'Dosage Increased',   'value': 'Dosage went up'}
                 ],
                 value='No change in dosage'),

    html.Label("Insulin"),
    dcc.Dropdown(id='insulin',
                 options=[
                     {'label': 'Not Prescribed',    'value': 'No change in dosage'},
                     {'label': 'No Change',          'value': 'Dosage remained steady'},
                     {'label': 'Dosage Decreased',   'value': 'Dosage went down'},
                     {'label': 'Dosage Increased',   'value': 'Dosage went up'}
                 ],
                 value='No change in dosage'),

    html.Button('Predict Readmission Risk', id='predict-btn', n_clicks=0),
    html.Div(id='prediction-output')
])

# ── Callback ──────────────────────────────────────────────────────────────────
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

    if n_clicks == 0:
        return ""

    # Start from default values for non-user-input columns
    inputs_list = defaults.copy()

    # Encode age bracket → integer (0-9)
    age_map = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
               '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    inputs_list['age'] = age_map.get(age, 5)

    # Encode A1C (was it measured?)
    inputs_list['A1Cresult'] = 1 if a1c == 'Yes' else 0

    # Numeric inputs passed through directly
    inputs_list['num_lab_procedures'] = num_lab
    inputs_list['num_medications']    = num_med
    inputs_list['number_diagnoses']   = num_diag
    inputs_list['number_inpatient']   = num_inpatient

    # Encode medication dosage change
    measures_map = {
        'No change in dosage':    0,
        'Dosage remained steady': 1,
        'Dosage went down':       2,
        'Dosage went up':         3
    }
    inputs_list['metformin'] = measures_map.get(metformin, 0)
    inputs_list['insulin']   = measures_map.get(insulin, 0)

    # Initialise all one-hot columns to 0
    for col in feature_columns:
        if col not in inputs_list:
            inputs_list[col] = 0

    # Set the selected diagnosis one-hot columns to 1
    # reverse_label maps display labels back to internal model column names (e.g. V_codes, E_codes)
    inputs_list[f'diag_1_{reverse_label(diag_1)}'] = 1
    inputs_list[f'diag_2_{reverse_label(diag_2)}'] = 1
    inputs_list[f'diag_3_{reverse_label(diag_3)}'] = 1

    # Build input DataFrame matching training column order exactly
    input_df = pd.DataFrame([inputs_list])[feature_columns]

    # Predict probability of readmission within 30 days
    proba = rf_model.predict_proba(input_df)[0][1]

    # Risk tier based on tuned threshold (0.20 chosen by F1 optimisation at training)
    if proba < 0.10:
        risk_label = "🟢 Low Risk"
        color = "#2ecc71"
    elif proba < 0.21:
        risk_label = "🟡 Medium Risk"
        color = "#f39c12"
    else:
        risk_label = "🔴 High Risk"
        color = "#e74c3c"

    # ── Distribution chart ────────────────────────────────────────────────────
    # Shows where this patient falls relative to all test-set patients
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=y_proba_population,
        nbinsx=40,
        name='All Patients',
        marker_color='#95a5a6',
        opacity=0.75
    ))
    dist_fig.add_vline(
        x=proba,
        line_color=color,
        line_width=3,
        annotation_text=f"This patient ({proba:.1%})",
        annotation_position="top right"
    )
    dist_fig.update_layout(
        title="Where This Patient Falls in the Population",
        xaxis_title="Predicted Readmission Probability",
        yaxis_title="Number of Patients",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ── Feature importance chart ──────────────────────────────────────────────
    # Global top-10 features by model importance
    top_features = feature_importances.head(10)
    imp_fig = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker_color='#3498db'
    ))
    imp_fig.update_layout(
        title="Top 10 Features That Drive This Model's Predictions",
        xaxis_title="Importance Score",
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ── Compose result panel ──────────────────────────────────────────────────
    return html.Div([
        html.H2(risk_label, style={'color': color, 'margin-top': '20px'}),
        html.P(
            f"Predicted readmission probability: {proba:.1%}",
            style={'font-size': '1.1em'}
        ),
        html.Hr(),
        dcc.Graph(figure=dist_fig),
        dcc.Graph(figure=imp_fig),
        html.Hr(),
        html.Div([
            html.H4("About This Prediction"),
            html.Ul([
                html.Li("Model: Random Forest (50 trees, max depth 10) with SMOTE resampling"),
                html.Li("AUC-ROC: 0.61 — above random chance (0.50), below strong clinical models (~0.70)"),
                html.Li("Risk threshold: probabilities ≥ 0.10 indicate elevated risk, ≥ 0.21 indicate high risk"),
                html.Li("This tool is a screening aid, not a clinical diagnosis. Always defer to clinical judgment.")
            ])
        ], style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '5px',
            'margin-top': '10px'
        })
    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
