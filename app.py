import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm
import io
import os

st.set_page_config(
    page_title="CardiometabolicAI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #001f5b 0%, #003087 50%, #0072CE 100%);
        padding: 30px 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,48,135,0.3);
    }
    .main-header h1 { font-size: 2.8rem; font-weight: 700; margin: 0; }
    .main-header p { margin: 6px 0 0 0; opacity: 0.85; font-size: 1rem; }

    .risk-high {
        background: linear-gradient(135deg, #fff0f0, #ffe5e5);
        border: 2px solid #e74c3c;
        border-left: 6px solid #e74c3c;
        padding: 20px 24px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; color: #c0392b;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #fffbf0, #fff3e0);
        border: 2px solid #f39c12;
        border-left: 6px solid #f39c12;
        padding: 20px 24px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; color: #d68910;
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fff4, #e8f5e9);
        border: 2px solid #2ecc71;
        border-left: 6px solid #2ecc71;
        padding: 20px 24px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; color: #1e8449;
    }
    .urgent-flag {
        background: linear-gradient(135deg, #fff0f0, #ffe5e5);
        border: 1.5px solid #e74c3c;
        padding: 10px 14px; border-radius: 8px;
        margin: 6px 0; color: #c0392b;
        font-weight: 600; font-size: 0.9rem;
    }
    .action-item {
        background: linear-gradient(135deg, #f0f8ff, #e8f4fd);
        border-left: 3px solid #3498db;
        padding: 10px 14px; margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #1a5276;
    }
    .guideline-item {
        background: linear-gradient(135deg, #f0fff4, #e8f5e9);
        border-left: 3px solid #2ecc71;
        padding: 8px 14px; margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem; color: #1e8449;
    }
    .ref-range {
        font-size: 0.72rem; color: #888;
        margin-top: 2px; font-style: italic;
    }
    .disclaimer {
        background: #f8f9fa; border: 1px solid #dee2e6;
        padding: 12px 16px; border-radius: 8px;
        font-size: 0.78rem; color: #6c757d;
        margin-top: 20px; line-height: 1.5;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9ff, #f0f4ff);
        border: 1px solid #d0d8f0;
        border-radius: 12px; padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('xgb_model_v2.pkl')
    scaler = joblib.load('scaler_v2.pkl')
    with open('feature_names_v2.json') as f:
        features = json.load(f)
    KG = nx.read_gml('clinical_kg_evidence_based.gml')
    return model, scaler, features, KG

model, scaler, feature_names, KG = load_assets()

def agentic_clinical_reasoning(risk_category, egfr, smoking_status, pa_minutes):
    actions, urgent_flags, guidelines_cited = [], [], []

    if risk_category == 'High':
        actions.append("Urgent clinical evaluation recommended")
        urgent_flags.append("🔴 HIGH CARDIOMETABOLIC RISK - Immediate clinical review advised")
        if egfr < 60:
            actions.append("Refer to nephrology - eGFR below 60 indicates CKD Stage 3+")
            guidelines_cited.append("NICE NG203 (2021): CKD - refer if eGFR <60")
            guidelines_cited.append("KDIGO 2021: eGFR <60 warrants specialist evaluation")
            urgent_flags.append("🔴 CKD DETECTED - Nephrology referral required")
        elif egfr < 90:
            actions.append("Monitor kidney function every 12 months")
            guidelines_cited.append("NICE NG203 (2021): Annual eGFR monitoring")
        if smoking_status == 'Current':
            actions.append("Refer to NHS Stop Smoking Service immediately")
            guidelines_cited.append("NICE PH10 (2008): Smoking cessation interventions")
            urgent_flags.append("🚬 CURRENT SMOKER - Cessation support required")
        if pa_minutes < 150:
            actions.append("Physical activity programme - target 150 min/week")
            guidelines_cited.append("WHO Physical Activity Guidelines 2020")
        actions.extend(["Consider statin therapy (ACC/AHA 2019)",
                        "Blood pressure monitoring and treatment if ≥140/90 mmHg",
                        "HbA1c and fasting glucose testing for diabetes screening"])
        guidelines_cited.extend(["ACC/AHA 2019: Statin therapy for high CVD risk",
                                  "NICE NG133 (2019): Hypertension management",
                                  "ADA Standards 2024: Annual diabetes screening"])

    elif risk_category == 'Moderate':
        actions.append("Lifestyle modification programme recommended")
        guidelines_cited.append("NICE NG28 (2019): Prevent T2DM through lifestyle")
        if egfr < 60:
            actions.append("Monitor kidney function - eGFR below 60")
            guidelines_cited.append("NICE NG203 (2021): CKD monitoring")
        if smoking_status == 'Current':
            actions.append("Refer to NHS Stop Smoking Service")
            guidelines_cited.append("NICE PH10: Smoking cessation support")
        if pa_minutes < 150:
            actions.append("Increase physical activity to ≥150 minutes/week")
            guidelines_cited.append("WHO 2020: 150-300 min/week moderate intensity")
        actions.extend(["Repeat cardiometabolic screening in 6-12 months",
                        "Weight management support if BMI ≥30 kg/m²"])
        guidelines_cited.append("NICE NG136 (2022): Weight management for obesity")

    else:
        actions.extend(["Maintain current healthy lifestyle",
                        "Routine cardiometabolic screening every 3-5 years"])
        guidelines_cited.append("NICE NG28: Routine screening for low-risk adults")
        if smoking_status == 'Current':
            actions.append("Smoking cessation strongly advised")
            guidelines_cited.append("NICE PH10: Smoking cessation for all smokers")
        if pa_minutes < 150:
            actions.append("Increase physical activity to meet WHO guidelines")
            guidelines_cited.append("WHO 2020: Minimum 150 min/week activity")

    return {'urgent_flags': urgent_flags,
            'recommended_actions': actions,
            'guidelines_cited': list(set(guidelines_cited))}

def create_risk_gauge(risk_prob, risk_label):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    segments = [(0, 3.33, '#2ecc71', 'Low'),
                (3.33, 6.67, '#f39c12', 'Moderate'),
                (6.67, 10, '#e74c3c', 'High')]

    for start, end, color, label in segments:
        bar = patches.FancyBboxPatch((start, 3.2), end-start-0.1, 0.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, alpha=0.25, edgecolor=color)
        ax.add_patch(bar)
        ax.text((start+end)/2, 2.95, label, ha='center', va='top',
                fontsize=8, color=color, fontweight='bold')

    needle_x = (risk_prob / 100) * 10
    colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
    color = colors[risk_label]

    active = patches.FancyBboxPatch((0, 3.2), needle_x-0.05, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, alpha=0.85, edgecolor=color)
    ax.add_patch(active)

    ax.text(5, 2.2, f"{risk_prob}%", ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(5, 1.4, "Prediction Confidence", ha='center', va='center',
            fontsize=9, color='#666')
    ax.text(5, 0.7, f"● {risk_label} Risk", ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)

    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    plt.tight_layout(pad=0)
    return fig

def generate_pdf_report(patient_data, risk_label, risk_prob, agent_output, shap_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=18, textColor=HexColor('#003087'), spaceAfter=6)
    story.append(Paragraph("CardiometabolicAI - Clinical Risk Report", title_style))
    story.append(Paragraph("Machine Learning-Based Clinical Decision Support System", styles['Normal']))
    story.append(Paragraph("University of Liverpool | School of Computer Science & Informatics", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#003087')))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    risk_colors = {'High': '#e74c3c', 'Moderate': '#f39c12', 'Low': '#2ecc71'}
    risk_style = ParagraphStyle('Risk', parent=styles['Heading1'],
                                 fontSize=16, textColor=HexColor(risk_colors[risk_label]))
    story.append(Paragraph(f"Risk Classification: {risk_label.upper()} CARDIOMETABOLIC RISK", risk_style))
    story.append(Paragraph(f"Prediction Confidence: {risk_prob}% | Model: XGBoost V2", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Patient Clinical Profile", styles['Heading2']))
    table_data = [['Variable', 'Value', 'Variable', 'Value']]
    items = list(patient_data.items())
    for i in range(0, len(items), 2):
        row = [items[i][0], str(items[i][1])]
        row += [items[i+1][0], str(items[i+1][1])] if i+1 < len(items) else ['', '']
        table_data.append(row)

    table = Table(table_data, colWidths=[4*cm, 3*cm, 4*cm, 3*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#003087')),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor('#f8f9fa'), white]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#dee2e6')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*cm))

    if agent_output['urgent_flags']:
        story.append(Paragraph("Urgent Clinical Flags", styles['Heading2']))
        for flag in agent_output['urgent_flags']:
            flag_style = ParagraphStyle('Flag', parent=styles['Normal'],
                                       textColor=HexColor('#c0392b'), fontName='Helvetica-Bold')
            story.append(Paragraph(f"! {flag}", flag_style))
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Recommended Clinical Actions", styles['Heading2']))
    for action in agent_output['recommended_actions']:
        story.append(Paragraph(f"• {action}", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Evidence Base - Guidelines Cited", styles['Heading2']))
    for guideline in agent_output['guidelines_cited']:
        story.append(Paragraph(f"- {guideline}", styles['Normal']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Top Contributing Clinical Factors (SHAP)", styles['Heading2']))
    shap_table_data = [['Feature', 'SHAP Value', 'Direction']]
    for _, row in shap_df.iterrows():
        direction = 'Increases Risk' if row['SHAP Value'] > 0 else 'Decreases Risk'
        shap_table_data.append([row['Feature'], f"{row['SHAP Value']:.4f}", direction])

    shap_table = Table(shap_table_data, colWidths=[6*cm, 4*cm, 5*cm])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#003087')),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [HexColor('#f8f9fa'), white]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#dee2e6')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 0.5*cm))

    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#dee2e6')))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
                                     fontSize=8, textColor=HexColor('#6c757d'))
    story.append(Paragraph(
        "CLINICAL DISCLAIMER: This report is generated by an AI-based clinical decision support system. "
        "It does not replace professional medical judgement. All recommendations must be reviewed "
        "by a qualified clinician before implementation.",
        disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── HEADER ──
st.markdown("""
<div class="main-header">
    <h1>🫀 CardiometabolicAI</h1>
    <p style="font-size:1.1rem; font-weight:300">
        Intelligent Clinical Decision Support for Cardiometabolic Risk Stratification
    </p>
    <p style="font-size:0.85rem; opacity:0.7; margin-top:8px">
        Powered by Explainable Machine Learning &nbsp;|&nbsp; Evidence-Based Clinical Guidelines
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🎓 University of Liverpool")
    st.markdown("**School of Computer Science & Informatics**")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    An XGBoost-based clinical decision support system trained on 
    **NHANES 2017–2020** population health data, stratifying 
    cardiometabolic risk across **3,528 adults** using 
    **19 validated clinical features**.
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Accuracy", "66%")
        st.metric("High Recall", "72%")
    with col_b:
        st.metric("Macro F1", "0.66")
        st.metric("Features", "19")
    st.markdown("---")
    st.markdown("### 🔬 Capabilities")
    st.markdown("""
    - 🤖 ML Risk Prediction
    - 📊 SHAP Explainability
    - 🧠 Agentic Clinical Reasoning
    - 📋 NICE Guideline Citations
    - 🔍 Knowledge Graph
    - 📝 Audit Logging
    - 📄 PDF Report Download
    """)
    st.markdown("---")
    st.markdown("### 🌐 Live")
    st.markdown("[cardiometabolic-risk.streamlit.app](https://cardiometabolic-risk.streamlit.app)")

# ── INPUT FORM ──
st.markdown("## 👤 Patient Clinical Profile")
st.markdown("Enter patient measurements to generate a personalised cardiometabolic risk assessment.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🧬 Demographics")
    age = st.number_input("Age (years)", 18, 80, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race/Ethnicity", [
        "Mexican American", "Non-Hispanic Asian",
        "Non-Hispanic Black", "Non-Hispanic White",
        "Other Hispanic", "Other/Multi-Racial"
    ])

with col2:
    st.markdown("#### 🩺 Clinical Measurements")
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 130, 80,
                                    help="Normal: <80 mmHg")
    st.markdown('<div class="ref-range">Normal: &lt;80 | High: ≥90 mmHg</div>', unsafe_allow_html=True)

    waist = st.number_input("Waist Circumference (cm)", 50, 200, 90,
                             help="Risk: >88cm (F) / >102cm (M)")
    st.markdown('<div class="ref-range">Risk: &gt;88cm (Female) / &gt;102cm (Male)</div>', unsafe_allow_html=True)

    total_chol = st.number_input("Total Cholesterol (mg/dL)", 50, 450, 180,
                                  help="Desirable: <200 mg/dL")
    st.markdown('<div class="ref-range">Desirable: &lt;200 | Borderline: 200-239 | High: ≥240</div>', unsafe_allow_html=True)

    hdl = st.number_input("HDL Cholesterol (mg/dL)", 10, 200, 50,
                           help="Higher is protective")
    st.markdown('<div class="ref-range">Protective: ≥60 | Low risk: &lt;40 (M) / &lt;50 (F)</div>', unsafe_allow_html=True)

    triglycerides = st.number_input("Triglycerides (mg/dL)", 10, 400, 100,
                                     help="Normal: <150 mg/dL")
    st.markdown('<div class="ref-range">Normal: &lt;150 | Borderline: 150-199 | High: ≥200</div>', unsafe_allow_html=True)

    ldl = st.number_input("LDL Cholesterol (mg/dL)", 10, 400, 100,
                           help="Optimal: <100 mg/dL")
    st.markdown('<div class="ref-range">Optimal: &lt;100 | Borderline: 130–159 | High: ≥160</div>', unsafe_allow_html=True)

with col3:
    st.markdown("#### 🏃 Lifestyle & Metabolic")
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

    pa_minutes = st.number_input("Physical Activity (min/week)", 0, 2000, 150,
                                  help="WHO recommends ≥150 min/week")
    who_status = "✅ Meeting WHO target" if pa_minutes >= 150 else "⚠️ Below WHO target"
    st.markdown(f'<div class="ref-range">{who_status} (150 min/week)</div>', unsafe_allow_html=True)

    sedentary = st.number_input("Sedentary Hours/day", 0, 16, 6,
                                 help="High risk: >8 hours/day")
    st.markdown('<div class="ref-range">High risk: &gt;8 hours/day</div>', unsafe_allow_html=True)

    uric_acid = st.number_input("Uric Acid (mg/dL)", 1.0, 20.0, 5.0,
                                 help="Normal: 2.4–6.0 (F) / 3.4–7.0 (M)")
    st.markdown('<div class="ref-range">Normal: 2.4–6.0 (F) / 3.4–7.0 (M)</div>', unsafe_allow_html=True)

    ggt = st.number_input("GGT (U/L)", 1, 500, 25,
                           help="Normal: <45 (F) / <65 (M)")
    st.markdown('<div class="ref-range">Normal: &lt;45 (Female) / &lt;65 (Male)</div>', unsafe_allow_html=True)

    egfr = st.number_input("eGFR (mL/min/1.73m²)", 1, 200, 90,
                            help="Normal: ≥90 | CKD: <60")
    egfr_label = "✅ Normal" if egfr >= 90 else "⚠️ Mildly Reduced" if egfr >= 60 else "🔴 CKD Stage 3+"
    st.markdown(f'<div class="ref-range">Status: {egfr_label}</div>', unsafe_allow_html=True)

st.markdown("---")

predict_col, _ = st.columns([2, 3])
with predict_col:
    predict_btn = st.button("🔍 Generate Risk Assessment", type="primary", use_container_width=True)

if predict_btn:
    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
    input_data = {
        'Age': age, 'Gender': 0 if gender == 'Male' else 1,
        'Diastolic_BP': diastolic_bp, 'Waist_Circumference': waist,
        'Total_Cholesterol': total_chol, 'HDL_Cholesterol': hdl,
        'Triglycerides': triglycerides, 'LDL_Cholesterol': ldl,
        'Smoking_Status': smoking_map[smoking],
        'Total_PA_Minutes': pa_minutes, 'Sedentary_Hours': sedentary,
        'Uric_Acid': uric_acid, 'GGT': ggt, 'eGFR': egfr,
        'Race_Non-Hispanic Asian': 1 if race == 'Non-Hispanic Asian' else 0,
        'Race_Non-Hispanic Black': 1 if race == 'Non-Hispanic Black' else 0,
        'Race_Non-Hispanic White': 1 if race == 'Non-Hispanic White' else 0,
        'Race_Other Hispanic': 1 if race == 'Other Hispanic' else 0,
        'Race_Other/Multi-Racial': 1 if race == 'Other/Multi-Racial' else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    risk_map = {0: 'High', 1: 'Low', 2: 'Moderate'}
    risk_label = risk_map[prediction]
    risk_prob = round(float(probability[prediction]) * 100, 1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    shap_vals = shap_values[0, :, prediction]
    shap_df = pd.DataFrame({
        'Feature': feature_names, 'SHAP Value': shap_vals
    }).sort_values('SHAP Value', key=abs, ascending=False).head(8)

    agent_output = agentic_clinical_reasoning(risk_label, egfr, smoking, pa_minutes)

    st.markdown("## 📊 Risk Assessment Results")
    st.markdown("---")

    res_col1, res_col2 = st.columns([3, 2])

    with res_col1:
        css_class = f"risk-{risk_label.lower()}"
        icons = {'High': '🔴', 'Moderate': '🟡', 'Low': '🟢'}
        st.markdown(
            f'<div class="{css_class}">{icons[risk_label]} {risk_label.upper()} CARDIOMETABOLIC RISK</div>',
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**Probability Distribution across Risk Classes:**")
        class_names = ['High', 'Low', 'Moderate']
        class_colors = ['#e74c3c', '#2ecc71', '#f39c12']
        for i, (cls, col) in enumerate(zip(class_names, class_colors)):
            prob = round(float(probability[i]) * 100, 1)
            st.markdown(f"""
            <div style="margin:6px 0">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.85rem; margin-bottom:3px">
                    <span style="font-weight:600; color:{col}">{cls}</span>
                    <span style="color:#333">{prob}%</span>
                </div>
                <div style="background:#f0f0f0; border-radius:6px; height:12px">
                    <div style="background:{col}; width:{prob}%; height:12px;
                                border-radius:6px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        gauge_fig = create_risk_gauge(risk_prob, risk_label)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Agentic Reasoning",
        "📊 SHAP Explanation",
        "🔍 Knowledge Graph",
        "📝 Audit Log"
    ])

    with tab1:
        st.subheader("Personalised Clinical Action Plan")
        if agent_output['urgent_flags']:
            st.markdown("#### 🚨 Urgent Flags")
            for flag in agent_output['urgent_flags']:
                st.markdown(f'<div class="urgent-flag">{flag}</div>',
                            unsafe_allow_html=True)
        st.markdown("#### ✅ Recommended Actions")
        for action in agent_output['recommended_actions']:
            st.markdown(f'<div class="action-item">• {action}</div>',
                        unsafe_allow_html=True)
        st.markdown("#### 📚 Evidence Base - Guidelines Cited")
        for guideline in agent_output['guidelines_cited']:
            st.markdown(f'<div class="guideline-item">📖 {guideline}</div>',
                        unsafe_allow_html=True)

    with tab2:
        st.subheader("SHAP Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_shap = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df['SHAP Value']]
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'],
                color=colors_shap, edgecolor='white', linewidth=0.5)
        ax.axvline(x=0, color='#333', linewidth=1)
        ax.set_title(f'Top 8 Features - {risk_label} Risk Prediction',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        col_a, col_b = st.columns(2)
        with col_a:
            st.info("🔴 **Red** - pushes toward this risk category")
        with col_b:
            st.info("🔵 **Blue** - pushes away from this risk category")

    with tab3:
        st.subheader("Evidence-Based Clinical Knowledge Graph")
        st.markdown("Constructed from real NICE guideline documents - NG203, NG133, NG136, T2DM.")
        color_map_kg = {'condition': '#f39c12', 'guideline': '#3498db', 'intervention': '#2ecc71'}
        node_colors = [color_map_kg.get(KG.nodes[n].get('node_type', 'condition'), '#95a5a6')
                       for n in KG.nodes()]
        fig_kg, ax_kg = plt.subplots(figsize=(14, 9))
        fig_kg.patch.set_facecolor('#fafbfc')
        ax_kg.set_facecolor('#fafbfc')
        pos = nx.spring_layout(KG, k=2.5, seed=42)
        nx.draw_networkx_nodes(KG, pos, node_color=node_colors,
                               node_size=1800, alpha=0.95, ax=ax_kg)
        nx.draw_networkx_labels(KG, pos, font_size=7, font_weight='bold', ax=ax_kg)
        nx.draw_networkx_edges(KG, pos, edge_color='#aab', arrows=True,
                               arrowsize=12, connectionstyle='arc3,rad=0.1',
                               ax=ax_kg, width=1.2)
        legend_elements = [
            mpatches.Patch(color='#f39c12', label='Clinical Condition'),
            mpatches.Patch(color='#3498db', label='NICE Guideline'),
            mpatches.Patch(color='#2ecc71', label='Intervention')
        ]
        ax_kg.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
        ax_kg.set_title('Evidence-Based Clinical Knowledge Graph',
                        fontweight='bold', fontsize=12, pad=15)
        ax_kg.axis('off')
        st.pyplot(fig_kg)
        plt.close()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Nodes", KG.number_of_nodes())
        with c2: st.metric("Total Edges", KG.number_of_edges())
        with c3: st.metric("Guidelines", "4 NICE PDFs")

    with tab4:
        st.subheader("Prediction Audit Log")
        log_entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Age': age, 'Gender': gender, 'Race': race,
            'Diastolic_BP': diastolic_bp, 'Waist_Circumference': waist,
            'eGFR': egfr, 'Smoking_Status': smoking,
            'PA_Minutes': pa_minutes, 'Predicted_Risk': risk_label,
            'Confidence': risk_prob, 'Model': 'XGBoost_V2',
            'Urgent_Flags': len(agent_output['urgent_flags'])
        }
        log_df = pd.DataFrame([log_entry])
        try:
            existing = pd.read_csv('audit_log.csv')
            log_df = pd.concat([existing, log_df], ignore_index=True)
        except FileNotFoundError:
            pass
        log_df.to_csv('audit_log.csv', index=False)
        st.success(f"✅ Logged - {len(log_df)} total predictions recorded")
        st.dataframe(log_df.tail(5), use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Download Clinical Report")
    patient_summary = {
        'Age': age, 'Gender': gender, 'Race': race,
        'Diastolic BP': f"{diastolic_bp} mmHg",
        'Waist Circumference': f"{waist} cm",
        'Total Cholesterol': f"{total_chol} mg/dL",
        'HDL Cholesterol': f"{hdl} mg/dL",
        'Triglycerides': f"{triglycerides} mg/dL",
        'LDL Cholesterol': f"{ldl} mg/dL",
        'Smoking Status': smoking,
        'Physical Activity': f"{pa_minutes} min/week",
        'eGFR': f"{egfr} mL/min/1.73m²"
    }
    pdf_buffer = generate_pdf_report(
        patient_summary, risk_label, risk_prob, agent_output, shap_df)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name=f"cardiometabolic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Clinical Disclaimer:</strong> CardiometabolicAI is intended for clinical decision 
    support purposes only. It does not replace professional medical judgement. All recommendations 
    should be reviewed by a qualified clinician before implementation. This system has not been 
    validated for direct patient care without clinical oversight.
    </div>
    """, unsafe_allow_html=True)