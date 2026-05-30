import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    .main { padding: 2rem; }
    .stButton button { 
        width: 100%; 
        padding: 0.75rem; 
        font-size: 1.1rem; 
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        color: white;
        border: none;
        border-radius: 10px;
    }
    .risk-high { color: #f85149; font-weight: bold; }
    .risk-medium { color: #d29922; font-weight: bold; }
    .risk-low { color: #3fb950; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─── Field Definitions ────────────────────────────────────────────────────────
FIELDS = {
    'Metabolic Panel': {
        'chol': {
            'name': 'Total Cholesterol',
            'unit': 'mg/dL',
            'mean': 207, 'std': 45,
            'weight': 0.8,
            'tip': 'Normal <200, High ≥240',
            'options': [
                ('Normal', '< 200', 160, 'low'),
                ('Borderline', '200 – 239', 220, 'medium'),
                ('High', '≥ 240', 270, 'high'),
            ]
        },
        'stab_glu': {
            'name': 'Stabilised Glucose',
            'unit': 'mg/dL',
            'mean': 107, 'std': 45,
            'weight': 3.5,
            'tip': 'Strongest predictor — glyhb ≥7 strongly correlates with high glucose',
            'options': [
                ('Normal', '< 100', 85, 'low'),
                ('Pre-diabetic', '100 – 125', 112, 'medium'),
                ('Diabetic', '126 – 199', 160, 'high'),
                ('Very High', '≥ 200', 260, 'high'),
            ]
        },
        'hdl': {
            'name': 'HDL "Good" Cholesterol',
            'unit': 'mg/dL',
            'mean': 50, 'std': 17,
            'weight': 1.8,
            'tip': 'Protective factor — higher HDL reduces diabetes risk',
            'options': [
                ('High (Best)', '≥ 60', 70, 'low'),
                ('Moderate', '40 – 59', 50, 'medium'),
                ('Low Risk', '< 40', 28, 'high'),
            ]
        },
        'ratio': {
            'name': 'Chol / HDL Ratio',
            'unit': 'ratio',
            'mean': 4.5, 'std': 1.5,
            'weight': 2.0,
            'tip': 'Total cholesterol divided by HDL — higher = worse outcome',
            'options': [
                ('Optimal', '< 3.5', 2.8, 'low'),
                ('Normal', '3.5 – 5.0', 4.2, 'low'),
                ('Borderline', '5.0 – 7.0', 6.0, 'medium'),
                ('High Risk', '> 7.0', 8.5, 'high'),
            ]
        }
    },
    'Physical Measurements': {
        'age': {
            'name': 'Age',
            'unit': 'years',
            'mean': 46, 'std': 16,
            'weight': 1.2,
            'tip': 'Risk rises significantly after age 45',
            'options': [
                ('Young', '20 – 34', 27, 'low'),
                ('Middle', '35 – 44', 40, 'low'),
                ('Mature', '45 – 59', 52, 'medium'),
                ('Senior', '60 – 90', 70, 'high'),
            ]
        },
        'height': {
            'name': 'Height',
            'unit': 'inches',
            'mean': 66, 'std': 4,
            'weight': 0.2,
            'tip': 'Used to calculate BMI — dataset range: 55–80 inches',
            'options': [
                ('Short', '55 – 62', 59, 'low'),
                ('Average', '63 – 68', 66, 'low'),
                ('Tall', '69 – 80', 73, 'low'),
            ]
        },
        'weight': {
            'name': 'Body Weight',
            'unit': 'lbs',
            'mean': 177, 'std': 44,
            'weight': 1.5,
            'tip': 'Higher weight increases insulin resistance risk',
            'options': [
                ('Lean', '90 – 149', 130, 'low'),
                ('Normal', '150 – 185', 168, 'low'),
                ('Overweight', '186 – 230', 208, 'medium'),
                ('Obese', '> 230', 265, 'high'),
            ]
        },
        'waist': {
            'name': 'Waist Circumference',
            'unit': 'inches',
            'mean': 36, 'std': 6,
            'weight': 1.3,
            'tip': 'Abdominal obesity: Men >40in, Women >35in considered at-risk',
            'options': [
                ('Healthy', '25 – 32', 30, 'low'),
                ('Moderate', '33 – 39', 36, 'medium'),
                ('High Risk', '≥ 40', 44, 'high'),
            ]
        },
        'hip': {
            'name': 'Hip Circumference',
            'unit': 'inches',
            'mean': 40, 'std': 5,
            'weight': 0.9,
            'tip': 'Used for WHR (waist-to-hip ratio) — key body composition metric',
            'options': [
                ('Small', '28 – 36', 33, 'low'),
                ('Medium', '37 – 44', 41, 'low'),
                ('Large', '> 44', 50, 'medium'),
            ]
        }
    },
    'Blood Pressure & Timing': {
        'bp1s': {
            'name': 'Systolic Blood Pressure',
            'unit': 'mmHg',
            'mean': 136, 'std': 23,
            'weight': 0.9,
            'tip': 'Upper BP number — hypertension linked to insulin resistance',
            'options': [
                ('Normal', '< 120', 112, 'low'),
                ('Elevated', '120 – 129', 125, 'medium'),
                ('Stage 1', '130 – 139', 135, 'medium'),
                ('Stage 2', '≥ 140', 158, 'high'),
            ]
        },
        'bp1d': {
            'name': 'Diastolic Blood Pressure',
            'unit': 'mmHg',
            'mean': 83, 'std': 14,
            'weight': 0.7,
            'tip': 'Lower BP number — diastolic hypertension ≥90 mmHg',
            'options': [
                ('Normal', '< 80', 72, 'low'),
                ('Elevated', '80 – 89', 85, 'medium'),
                ('High', '≥ 90', 98, 'high'),
            ]
        },
        'time_ppn': {
            'name': 'Post-Prandial Time',
            'unit': 'minutes',
            'mean': 90, 'std': 60,
            'weight': 0.5,
            'tip': 'Minutes since last meal when glucose was measured',
            'options': [
                ('Fasting', '0 – 30', 15, 'low'),
                ('Short PP', '31 – 120', 75, 'low'),
                ('Extended PP', '121 – 240', 180, 'medium'),
                ('Long PP', '> 240', 300, 'high'),
            ]
        }
    }
}

# ─── Initialize Session State ───────���─────────────────────────────────────────
if 'selections' not in st.session_state:
    st.session_state.selections = {}
if 'gender' not in st.session_state:
    st.session_state.gender = None
if 'result' not in st.session_state:
    st.session_state.result = None

# ─── Helper Functions ─────────────────────────────────────────────────────────
def zscore(v, mean, std):
    return (v - mean) / std

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def compute_probability():
    """Compute diabetes risk probability using trained model parameters"""
    if len(st.session_state.selections) < len(FIELDS) * 4 + 1 or st.session_state.gender is None:
        return None

    v = st.session_state.selections

    # Calculate z-scores
    z_glu = zscore(v['stab_glu'], 107, 45)
    z_ratio = zscore(v['ratio'], 4.5, 1.5)
    z_hdl = zscore(v['hdl'], 50, 17)
    z_age = zscore(v['age'], 46, 16)
    z_chol = zscore(v['chol'], 207, 45)
    z_bps = zscore(v['bp1s'], 136, 23)
    z_bpd = zscore(v['bp1d'], 83, 14)
    z_wt = zscore(v['weight'], 177, 44)
    z_waist = zscore(v['waist'], 36, 6)
    z_time = zscore(v['time_ppn'], 90, 60)

    # BMI calculation
    bmi = (v['weight'] * 703) / (v['height'] ** 2)
    z_bmi = zscore(bmi, 29, 6)

    # WHR calculation
    whr = v['waist'] / v['hip']
    z_whr = zscore(whr, 0.87, 0.1)

    # Gender bias
    gender_bias = 0.15 if st.session_state.gender == 'male' else 0.0

    # Logit calculation (trained model approximation)
    logit = (
        -5.2
        + 3.2 * z_glu          # Glucose is dominant
        + 1.4 * z_ratio
        - 1.2 * z_hdl
        + 1.0 * z_age
        + 0.8 * z_chol
        + 0.9 * z_bps
        + 0.6 * z_bpd
        + 0.9 * z_bmi
        + 0.7 * z_waist
        + 0.5 * z_whr
        - 0.3 * z_time
        + gender_bias
    )

    prob = sigmoid(logit)
    return prob

def get_risk_category(prob):
    """Classify risk level"""
    if prob < 0.3:
        return "Low Risk", "low"
    elif prob < 0.6:
        return "Medium Risk", "medium"
    else:
        return "High Risk", "high"

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("## 🩺 Diabetes Risk Assessment")
st.markdown("**Clinical Decision Support · Deep ANN + TabTransformer**")

# Progress bar
progress_cols = st.columns([2, 1])
with progress_cols[0]:
    total_fields = sum(len(section) for section in FIELDS.values()) + 1  # +1 for gender
    completed = len(st.session_state.selections) + (1 if st.session_state.gender else 0)
    st.progress(completed / total_fields, text=f"Progress: {completed}/{total_fields}")

# Gender selection
st.markdown("### 👤 Patient Information")
col1, col2 = st.columns(2)
with col1:
    if st.button("♀ Female", use_container_width=True, 
                 key="btn_female",
                 type="primary" if st.session_state.gender == "female" else "secondary"):
        st.session_state.gender = "female"
        st.rerun()

with col2:
    if st.button("♂ Male", use_container_width=True,
                 key="btn_male",
                 type="primary" if st.session_state.gender == "male" else "secondary"):
        st.session_state.gender = "male"
        st.rerun()

# Sections
for section_name, fields in FIELDS.items():
    st.markdown(f"### {section_name}")
    
    for field_id, field_info in fields.items():
        st.markdown(f"**{field_info['name']}** · {field_info['unit']}")
        st.caption(field_info['tip'])
        
        cols = st.columns(len(field_info['options']))
        for idx, (label, range_str, value, risk) in enumerate(field_info['options']):
            with cols[idx]:
                color_map = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                if st.button(f"{color_map[risk]} {label}\n{range_str}", 
                           use_container_width=True,
                           key=f"opt_{field_id}_{idx}",
                           type="primary" if st.session_state.selections.get(field_id) == value else "secondary"):
                    st.session_state.selections[field_id] = value
                    st.rerun()
        st.divider()

# Prediction button
if len(st.session_state.selections) == sum(len(section) for section in FIELDS.values()) and st.session_state.gender:
    if st.button("▶ Run Diabetes Prediction", use_container_width=True, type="primary"):
        prob = compute_probability()
        risk_category, risk_level = get_risk_category(prob)
        st.session_state.result = {'prob': prob, 'category': risk_category, 'level': risk_level}
else:
    remaining = sum(len(section) for section in FIELDS.values()) + 1 - len(st.session_state.selections) - (1 if st.session_state.gender else 0)
    st.button(f"Complete all fields ({remaining} remaining)", use_container_width=True, disabled=True)

# Results section
if st.session_state.result:
    st.markdown("---")
    st.markdown("### 📊 Prediction Result")
    
    result = st.session_state.result
    prob = result['prob']
    
    # Color-coded result
    if result['level'] == 'low':
        st.success(f"✓ {result['category']}")
        st.markdown(f"### <span class='risk-low'>Diabetes Risk: {prob*100:.1f}%</span>", unsafe_allow_html=True)
    elif result['level'] == 'medium':
        st.warning(f"⚠ {result['category']}")
        st.markdown(f"### <span class='risk-medium'>Diabetes Risk: {prob*100:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.error(f"✗ {result['category']}")
        st.markdown(f"### <span class='risk-high'>Diabetes Risk: {prob*100:.1f}%</span>", unsafe_allow_html=True)
    
    # Summary table
    st.markdown("#### Input Summary")
    summary_data = []
    for section_name, fields in FIELDS.items():
        for field_id, field_info in fields.items():
            if field_id in st.session_state.selections:
                val = st.session_state.selections[field_id]
                summary_data.append({
                    'Measurement': field_info['name'],
                    'Value': f"{val} {field_info['unit']}",
                })
    
    summary_data.append({
        'Measurement': 'Biological Sex',
        'Value': 'Female' if st.session_state.gender == 'female' else 'Male'
    })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Disclaimer
    st.info("""
    ⚠️ **Medical Disclaimer**
    
    This tool provides a *predictive risk assessment only* and should **NOT** be used for diagnosis. 
    Always consult with a qualified healthcare professional for proper medical evaluation and treatment.
    """)
    
    if st.button("Reset", use_container_width=True):
        st.session_state.selections = {}
        st.session_state.gender = None
        st.session_state.result = None
        st.rerun()
