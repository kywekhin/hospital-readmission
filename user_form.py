from dash import Dash, dcc, html, Input, Output, State, callback
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Load model artifacts ──────────────────────────────────────────────────────
rf_model             = joblib.load('models/rf_smote.pkl')
feature_columns      = joblib.load('models/feature_columns.pkl')
defaults             = joblib.load('models/defaults.pkl')
y_proba_population   = joblib.load('models/y_proba_sm.pkl')
feature_importances  = joblib.load('models/feature_importances.pkl')

# ── Diagnosis label helpers ───────────────────────────────────────────────────
def clean_label(label):
    if label == 'E_codes': return 'Externally caused injury'
    if label == 'V_codes': return 'Pre-existing potential health hazard'
    return label

def reverse_label(label):
    if label == 'Pre-existing potential health hazard': return 'V_codes'
    if label == 'Externally caused injury': return 'E_codes'
    return label

diag_1_categories = [clean_label(col.replace('diag_1_', '')) for col in feature_columns if col.startswith('diag_1_')]
diag_2_categories = [clean_label(col.replace('diag_2_', '')) for col in feature_columns if col.startswith('diag_2_')]
diag_3_categories = [clean_label(col.replace('diag_3_', '')) for col in feature_columns if col.startswith('diag_3_')]

# ── Shared styles ─────────────────────────────────────────────────────────────
COLORS = {
    'primary':    '#1a56a0',   # accessible dark blue
    'light_bg':   '#f0f4fa',   # soft blue-tinted background
    'white':      '#ffffff',
    'border':     '#b3c6e0',
    'text':       '#1a1a2e',   # near-black for high contrast
    'subtext':    '#4a5568',
    'low':        '#1a7a4a',   # accessible green
    'medium':     '#b45309',   # accessible amber (not yellow — fails contrast)
    'high':       '#b91c1c',   # accessible red
    'chart_blue': '#2563eb',
    'chart_grey': '#94a3b8',
}

FIELD = {
    'margin-bottom': '20px',
    'display': 'flex',
    'flex-direction': 'column',
    'gap': '6px',
}

LABEL = {
    'font-size': '1rem',
    'font-weight': '600',
    'color': COLORS['text'],
    'letter-spacing': '0.01em',
}

INPUT = {
    'width': '100%',
    'padding': '10px 14px',
    'font-size': '1rem',
    'border': f"1.5px solid {COLORS['border']}",
    'border-radius': '6px',
    'color': COLORS['text'],
    'background-color': COLORS['white'],
    'box-sizing': 'border-box',
    'outline-offset': '2px',
}

DROPDOWN_STYLE = {
    'font-size': '1rem',
    'color': COLORS['text'],
}

SECTION_CARD = {
    'background-color': COLORS['white'],
    'border': f"1px solid {COLORS['border']}",
    'border-radius': '10px',
    'padding': '24px 28px',
    'margin-bottom': '24px',
}

def field(label_text, component):
    """Wrap a label and input component into an accessible stacked field block."""
    return html.Div([
        html.Label(label_text, style=LABEL),
        component
    ], style=FIELD)

# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(__name__, title="Readmission Risk Predictor")

app.index_string = '''
<!DOCTYPE html>
<html lang="en">
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f0f4fa;
            color: #1a1a2e;
            line-height: 1.6;
            min-height: 100vh;
        }

        /* Accessible focus ring on all interactive elements */
        a:focus, button:focus, input:focus, select:focus,
        [tabindex]:focus, .Select-control:focus-within {
            outline: 3px solid #2563eb;
            outline-offset: 2px;
        }

        /* Dash dropdown overrides */
        .Select-control {
            border: 1.5px solid #b3c6e0 !important;
            border-radius: 6px !important;
            min-height: 44px !important;   /* WCAG touch target */
            font-size: 1rem !important;
        }
        .Select-value-label { color: #1a1a2e !important; }
        .Select-menu-outer { border: 1.5px solid #b3c6e0 !important; border-radius: 6px !important; }
        .VirtualizedSelectOption { font-size: 1rem !important; min-height: 40px !important; }
        .VirtualizedSelectFocusedOption { background-color: #dbeafe !important; color: #1a1a2e !important; }

        /* Number inputs */
        input[type=number] {
            -moz-appearance: textfield;
            min-height: 44px;
        }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; }

        /* Predict button */
        #predict-btn {
            display: block;
            width: 100%;
            padding: 14px;
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            background-color: #1a56a0;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            letter-spacing: 0.02em;
            transition: background-color 0.15s ease;
            min-height: 52px;
        }
        #predict-btn:hover  { background-color: #154080; }
        #predict-btn:active { background-color: #0f2d5e; }

        /* Skip-to-content for screen readers */
        .skip-link {
            position: absolute; top: -40px; left: 0;
            background: #1a56a0; color: white;
            padding: 8px; z-index: 9999;
            font-size: 1rem; border-radius: 0 0 6px 0;
        }
        .skip-link:focus { top: 0; }
    </style>
</head>
<body>
    <a href="#main-form" class="skip-link">Skip to main content</a>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Header(
        html.Div([
            html.H1("Hospital Readmission Risk Predictor",
                    style={'font-size': '1.75rem', 'font-weight': '800',
                           'color': COLORS['white'], 'margin-bottom': '4px'}),
            html.P("Discharge-time screening tool for 30-day readmission risk in diabetic patients.",
                   style={'font-size': '1rem', 'color': '#cce0ff', 'margin': 0})
        ], style={'max-width': '760px', 'margin': '0 auto', 'padding': '0 16px'}),
        style={'background-color': COLORS['primary'],
               'padding': '24px 0', 'margin-bottom': '32px'}
    ),

    # ── Main content ──────────────────────────────────────────────────────────
    html.Main([

        # ── Patient info card ─────────────────────────────────────────────────
        html.Section([
            html.H2("Patient Information",
                    style={'font-size': '1.15rem', 'font-weight': '700',
                           'color': COLORS['primary'], 'margin-bottom': '20px',
                           'border-bottom': f"2px solid {COLORS['light_bg']}",
                           'padding-bottom': '10px'}),

            field("Age", dcc.Dropdown(
                id='age',
                options=['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                         '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
                value='[50-60)',
                style=DROPDOWN_STYLE,
                clearable=False
            )),

            field("Primary Diagnosis", dcc.Dropdown(
                id='diag_1', options=diag_1_categories,
                value=diag_1_categories[0], style=DROPDOWN_STYLE, clearable=False
            )),

            field("Secondary Diagnosis", dcc.Dropdown(
                id='diag_2', options=diag_2_categories,
                value=diag_2_categories[0], style=DROPDOWN_STYLE, clearable=False
            )),

            field("Additional Diagnosis", dcc.Dropdown(
                id='diag_3', options=diag_3_categories,
                value=diag_3_categories[0], style=DROPDOWN_STYLE, clearable=False
            )),

        ], style=SECTION_CARD),

        # ── Admission metrics card ────────────────────────────────────────────
        html.Section([
            html.H2("Admission Metrics",
                    style={'font-size': '1.15rem', 'font-weight': '700',
                           'color': COLORS['primary'], 'margin-bottom': '20px',
                           'border-bottom': f"2px solid {COLORS['light_bg']}",
                           'padding-bottom': '10px'}),

            field("Was A1C Measured During This Admission?", dcc.Dropdown(
                id='A1Cresult', options=['No', 'Yes'],
                value='No', style=DROPDOWN_STYLE, clearable=False
            )),

            field("Number of Lab Procedures", dcc.Input(
                id='num_lab_procedures', type='number',
                min=0, max=132, value=8, style=INPUT
            )),

            field("Number of Medications", dcc.Input(
                id='num_medications', type='number',
                min=0, max=81, value=8, style=INPUT
            )),

            field("Number of Diagnoses", dcc.Input(
                id='number_diagnoses', type='number',
                min=0, max=16, value=4, style=INPUT
            )),

            field("Number of Previous Inpatient Visits", dcc.Input(
                id='number_inpatient', type='number',
                min=0, max=21, value=0, style=INPUT
            )),

        ], style=SECTION_CARD),

        # ── Medications card ──────────────────────────────────────────────────
        html.Section([
            html.H2("Medication Changes at Discharge",
                    style={'font-size': '1.15rem', 'font-weight': '700',
                           'color': COLORS['primary'], 'margin-bottom': '20px',
                           'border-bottom': f"2px solid {COLORS['light_bg']}",
                           'padding-bottom': '10px'}),

            field("Metformin", dcc.Dropdown(
                id='metformin',
                options=[
                    {'label': 'Not Prescribed',  'value': 'No change in dosage'},
                    {'label': 'No Change',        'value': 'Dosage remained steady'},
                    {'label': 'Dosage Decreased', 'value': 'Dosage went down'},
                    {'label': 'Dosage Increased', 'value': 'Dosage went up'},
                ],
                value='No change in dosage', style=DROPDOWN_STYLE, clearable=False
            )),

            field("Insulin", dcc.Dropdown(
                id='insulin',
                options=[
                    {'label': 'Not Prescribed',  'value': 'No change in dosage'},
                    {'label': 'No Change',        'value': 'Dosage remained steady'},
                    {'label': 'Dosage Decreased', 'value': 'Dosage went down'},
                    {'label': 'Dosage Increased', 'value': 'Dosage went up'},
                ],
                value='No change in dosage', style=DROPDOWN_STYLE, clearable=False
            )),

        ], style=SECTION_CARD),

        # ── Submit ────────────────────────────────────────────────────────────
        html.Button('Predict Readmission Risk', id='predict-btn', n_clicks=0),

        # ── Results ───────────────────────────────────────────────────────────
        html.Div(id='prediction-output', style={'margin-top': '32px'})

    ], id='main-form',
       style={'max-width': '760px', 'margin': '0 auto',
              'padding': '0 16px 60px'}),

], style={'background-color': COLORS['light_bg'], 'min-height': '100vh'})


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

    # Build input vector from defaults
    inputs_list = defaults.copy()

    # Encode age bracket → integer (0–9)
    age_map = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
               '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    inputs_list['age'] = age_map.get(age, 5)

    # A1C binary
    inputs_list['A1Cresult'] = 1 if a1c == 'Yes' else 0

    # Numeric inputs
    inputs_list['num_lab_procedures'] = num_lab
    inputs_list['num_medications']    = num_med
    inputs_list['number_diagnoses']   = num_diag
    inputs_list['number_inpatient']   = num_inpatient

    # Medication dosage encoding
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

    # Set selected diagnosis columns to 1
    inputs_list[f'diag_1_{reverse_label(diag_1)}'] = 1
    inputs_list[f'diag_2_{reverse_label(diag_2)}'] = 1
    inputs_list[f'diag_3_{reverse_label(diag_3)}'] = 1

    input_df = pd.DataFrame([inputs_list])[feature_columns]
    proba    = rf_model.predict_proba(input_df)[0][1]

    # Risk tier
    if proba < 0.10:
        risk_label = "🟢 Low Risk"
        risk_color  = COLORS['low']
        risk_bg     = '#f0fdf4'
        risk_border = '#86efac'
        risk_desc   = "This patient shows low likelihood of readmission within 30 days."
    elif proba < 0.21:
        risk_label = "🟡 Medium Risk"
        risk_color  = COLORS['medium']
        risk_bg     = '#fffbeb'
        risk_border = '#fcd34d'
        risk_desc   = "This patient shows elevated risk. Consider follow-up scheduling."
    else:
        risk_label = "🔴 High Risk"
        risk_color  = COLORS['high']
        risk_bg     = '#fef2f2'
        risk_border = '#fca5a5'
        risk_desc   = "This patient shows high risk. Clinical review and close monitoring recommended."

    # Distribution chart
    dist_fig = go.Figure()
    dist_fig.add_trace(go.Histogram(
        x=y_proba_population, nbinsx=40,
        marker_color=COLORS['chart_grey'], opacity=0.75, name='All Patients'
    ))
    dist_fig.add_vline(
        x=proba, line_color=risk_color, line_width=3,
        annotation_text=f"This patient ({proba:.1%})",
        annotation_font_color=risk_color,
        annotation_position="top right"
    )
    dist_fig.update_layout(
        title=dict(text="Patient Risk vs. Population Distribution", font=dict(size=15)),
        xaxis_title="Predicted Readmission Probability",
        yaxis_title="Number of Patients",
        showlegend=False,
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white'],
        font=dict(color=COLORS['text'], size=13),
        margin=dict(t=50, b=50, l=50, r=30)
    )

    # Feature importance chart
    top_features = feature_importances.head(10)
    imp_fig = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker_color=COLORS['chart_blue']
    ))
    imp_fig.update_layout(
        title=dict(text="Top 10 Predictive Features (Model-Wide)", font=dict(size=15)),
        xaxis_title="Importance Score",
        yaxis=dict(autorange='reversed'),
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white'],
        font=dict(color=COLORS['text'], size=13),
        margin=dict(t=50, b=50, l=220, r=30)
    )

    return html.Div([

        # Risk badge
        html.Div([
            html.P(risk_label,
                   style={'font-size': '1.5rem', 'font-weight': '800',
                          'color': risk_color, 'margin-bottom': '6px'}),
            html.P(f"Predicted probability: {proba:.1%}",
                   style={'font-size': '1rem', 'font-weight': '600',
                          'color': risk_color, 'margin-bottom': '8px'}),
            html.P(risk_desc,
                   style={'font-size': '0.95rem', 'color': COLORS['text'], 'margin': 0})
        ], style={
            'background-color': risk_bg,
            'border': f"2px solid {risk_border}",
            'border-radius': '10px',
            'padding': '20px 24px',
            'margin-bottom': '24px'
        }),

        # Charts
        html.Div(dcc.Graph(figure=dist_fig, config={'displayModeBar': False}),
                 style=SECTION_CARD),
        html.Div(dcc.Graph(figure=imp_fig, config={'displayModeBar': False}),
                 style=SECTION_CARD),

        # Model context
        html.Div([
            html.H3("About This Prediction",
                    style={'font-size': '1rem', 'font-weight': '700',
                           'color': COLORS['primary'], 'margin-bottom': '12px'}),
            html.Ul([
                html.Li("Model: Random Forest (50 trees, max depth 10) with SMOTE resampling",
                        style={'margin-bottom': '6px'}),
                html.Li("AUC-ROC: 0.61 — above random chance (0.50); clinical models typically reach ~0.70",
                        style={'margin-bottom': '6px'}),
                html.Li("Risk thresholds: Low < 10% · Medium 10–21% · High > 21%",
                        style={'margin-bottom': '6px'}),
                html.Li("⚠️ This tool is a screening aid only. Always defer to clinical judgment.",
                        style={'font-weight': '600'})
            ], style={'padding-left': '20px', 'font-size': '0.9rem',
                      'color': COLORS['subtext'], 'line-height': '1.7'})
        ], style={**SECTION_CARD, 'border-left': f"4px solid {COLORS['primary']}"})

    ])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)