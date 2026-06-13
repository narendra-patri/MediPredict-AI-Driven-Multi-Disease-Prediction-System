"""
Run with: streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict — AI Driven Multi-Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

/* -----------------------------
   GLOBAL APP
----------------------------- */

.stApp {
    background:
        radial-gradient(circle at top left, #1e3a8a 0%, transparent 40%),
        radial-gradient(circle at top right, #0891b2 0%, transparent 35%),
        linear-gradient(135deg,#020617,#0f172a,#1e293b);
    color: white;
    font-family: 'Inter', sans-serif;
}

/* Smooth animations */

* {
    transition: all 0.3s ease;
}

# .main {
#     animation: fadeIn 0.8s ease;
# }

# @keyframes fadeIn {
#     from {
#         opacity:0;
#         transform: translateY(15px);
#     }
#     to {
#         opacity:1;
#         transform: translateY(0);
#     }
# }

/* -----------------------------
   HERO TITLE
----------------------------- */

.main-title {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(
        90deg,
        #38BDF8,
        #60A5FA,
        #22C55E
    );
     -webkit-background-clip: text;
    # -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -2px;
}

.sub-title {
    text-align: center;
    color: #CBD5E1;
    font-size: 1.2rem;
    margin-bottom: 3rem;
}

/* -----------------------------
   GLASS CARDS
----------------------------- */

.info-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 1.5rem;
    color: white;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.25);
}

.info-card:hover {
    transform: translateY(-4px);
}

div[data-testid="stVerticalBlock"] {
    border-radius: 16px;
}

.info-card:hover,
[data-testid="metric-container"]:hover {
    transform: translateY(-6px);
}
/* -----------------------------
   METRICS
----------------------------- */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow:
        0 8px 20px rgba(0,0,0,0.2);
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
}

/* -----------------------------
   INPUT FIELDS
----------------------------- */

.stNumberInput input,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
}

label {
    color: #E2E8F0 !important;
    font-weight: 600;
}

/* -----------------------------
   BUTTON
----------------------------- */

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.9rem;
    font-weight: 700;
    font-size: 1rem;

    background: linear-gradient(
        135deg,
        #2563EB,
        #06B6D4
    );

    color: white;

    box-shadow:
        0 10px 25px rgba(37,99,235,0.35);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:
        0 15px 35px rgba(37,99,235,0.45);
}

/* -----------------------------
   SIDEBAR
----------------------------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0f172a,
            #111827
        );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* -----------------------------
   SUCCESS RESULT
----------------------------- */

.result-box-safe {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 6px solid #10B981;
    border-radius: 16px;
    padding: 1.4rem;
    color: #D1FAE5;
    font-weight: 700;
    backdrop-filter: blur(12px);
}

/* -----------------------------
   DANGER RESULT
----------------------------- */

.result-box-danger {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 6px solid #EF4444;
    border-radius: 16px;
    padding: 1.4rem;
    color: #FECACA;
    font-weight: 700;
    backdrop-filter: blur(12px);
}

/* -----------------------------
   DATAFRAME
----------------------------- */

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

/* -----------------------------
   PROGRESS BAR
----------------------------- */

.stProgress > div > div {
    background:
        linear-gradient(
            90deg,
            #06B6D4,
            #10B981
        );
}

/* -----------------------------
   FOOTER
----------------------------- */

.footer {
    text-align: center;
    color: #94A3B8;
    margin-top: 2rem;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

@st.cache_resource
def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🏥 MediPredict")
st.sidebar.markdown("AI-Driven Multi-Disease Prediction")
st.sidebar.markdown("---")

disease = st.sidebar.radio(
    "Select Disease to Predict:",
    ["🏠 Home", "🩸 Diabetes", "❤️ Heart Disease",
     "🧠 Parkinson's Disease", "🫘 Kidney Disease", "🫀 Liver Disease"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Disclaimer**")
st.sidebar.caption("This tool is for educational purposes only. Always consult a medical professional for diagnosis.")

# ─────────────────────────────────────────────
# Medical Term Funtion
# ─────────────────────────────────────────────

def medical_term():
    st.markdown("### 🔎 Medical Term Search")
    medical_terms = {
                "BMI": "Body Mass Index – measures body fat based on height and weight.",
                "Pregnancies": "The total number of times a woman has been pregnant, including current and past pregnancies.",
                "Skin Thickness": "The thickness of the skin fold measured in millimeters, used to estimate body fat levels.",
                "Diabetes Pedigree Function": "A score that indicates the likelihood of diabetes based on family history.",
                "Glucose": "The amount of sugar present in the blood.",
                "Insulin": "A hormone that helps regulate blood sugar levels.",
                "Blood Pressure": "The force exerted by circulating blood against artery walls.",
                "Exercise Induced Angina": "Chest pain that occurs during physical activity due to reduced blood flow to the heart.",
                "Fasting Blood Sugar > 120 mg/dL": "Indicates whether blood sugar levels are higher than normal after fasting.",
                "Oldpeak (ST Depression)": "A measure of changes in the heart's electrical activity during exercise.",
                "Chest Pain Type": "The type of chest discomfort experienced by the patient.",
                "ATA": "Atypical Angina – chest pain that is not typical of heart disease.",
                "NAP": "Non-Anginal Pain – chest pain not related to heart problems.",
                "ASY": "Asymptomatic – no chest pain symptoms present.",
                "TA": "Typical Angina – chest pain commonly associated with heart disease.",
                "Resting ECG": "Results of the heart's electrical activity test while at rest.",
                "Normal": "Normal heart electrical activity.",
                "ST": "Abnormal ST segment indicating possible heart issues.",
                "LVH": "Left Ventricular Hypertrophy – thickening of the heart's left pumping chamber.",
                "ST Slope": "The direction of the ST segment during exercise testing.",
                "Up": "Upward slope, generally considered normal.",
                "Flat": "Flat slope, may indicate heart disease.",
                "Down": "Downward slope, often associated with heart problems.",
                "Resting Blood Pressure": "The blood pressure measured when a person is relaxed and resting.",
                "Max Heart Rate Achieved": "The highest heart rate reached during physical activity or testing.",
                "Cholesterol": "A waxy substance found in the blood that can affect heart health.",
                "Albumin": "A protein made by the liver that helps maintain fluid balance.",
                "Creatinine": "A waste product used to assess kidney function.",
                "Hemoglobin": "A protein in red blood cells that carries oxygen.",
                "Bilirubin": "A yellow substance produced during red blood cell breakdown.",
                "Parkinson's Disease": "A neurological disorder affecting movement and coordination.",
                "MDVP:Fo(Hz)": "The average frequency (pitch) of a person's voice.",
                "MDVP:Fhi(Hz)": "The highest frequency (pitch) recorded in the voice.",
                "MDVP:Flo(Hz)": "The lowest frequency (pitch) recorded in the voice.",
                "MDVP:Jitter(%)": "Measures small variations in voice frequency from cycle to cycle.",
                "MDVP:Jitter(Abs)": "The absolute amount of variation in voice frequency.",
                "MDVP:RAP": "A measure of short-term changes in voice frequency.",
                "MDVP:PPQ": "Measures irregularities in voice frequency over several cycles.",
                "Jitter:DDP": "A measure of frequency instability in the voice.",
                "MDVP:Shimmer": "Measures small variations in voice loudness.",
                "MDVP:Shimmer(dB)": "The variation in voice loudness measured in decibels.",
                "Shimmer:APQ3": "Measures short-term changes in voice loudness over 3 cycles.",
                "Shimmer:APQ5": "Measures short-term changes in voice loudness over 5 cycles.",
                "MDVP:APQ": "Measures overall variation in voice loudness.",
                "Shimmer:DDA": "A measure of instability in voice loudness.",
                "NHR": "Measures the amount of noise present in the voice signal.",
                "HNR": "Measures the ratio of clear voice sound to background noise.",
                "RPDE": "Measures the complexity and irregularity of voice patterns.",
                "DFA": "Measures the stability and self-similarity of voice signals.",
                "Spread1": "A feature that measures variations in voice frequency patterns.",
                "Spread2": "A feature that measures the spread of voice frequency variations.",
                "D2": "Measures the complexity of vocal cord movements.",
                "PPE": "Measures irregularities in voice pitch patterns."
            }
    
    search_term = st.text_input(
        "Enter a medical term",
        placeholder="Example: BMI, Glucose, Albumin"
    )
    
    if st.button("Search Meaning"):
        # Create a lowercase version of the dictionary
        medical_terms_lower = {key.lower(): value for key, value in medical_terms.items()}
        term = search_term.strip().lower()
        if term in medical_terms_lower:
            st.success(f"{search_term}: {medical_terms_lower[term]}")
        else:
            st.warning("Term not found. Please try another medical term.")

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if disease == "🏠 Home":
    st.markdown('<div class="main-title">🏥 MediPredict</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Driven Multi-Disease Prediction</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diseases Covered", "5")
    with col2:
        st.metric("ML Models", "5 Trained")
    with col3:
        st.metric("Algorithms Used", "RF, SVM, XGBoost")

    st.markdown("---")
    st.markdown("### 🔬 How to Use")
    st.markdown("""
    1. Select a disease from the **left sidebar**
    2. Fill in the patient's medical values
    3. Click **Predict** to get an instant result
    4. The model shows **risk level** and **probability**
    """)

    st.markdown("### 🧬 Diseases & Models")
    data = {
        "Disease": ["Diabetes", "Heart Disease", "Parkinson's", "Kidney Disease", "Liver Disease"],
        "Algorithm": ["Random Forest", "Random Forest", "SVM (RBF)", "Random Forest", "XGBoost"],
        "Dataset Size": ["768 rows", "918 rows", "195 rows", "400 rows", "583 rows"],
        "Expected Accuracy": ["~80%", "~87%", "~93%", "~98%", "~76%"]
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# DIABETES
# ─────────────────────────────────────────────
elif disease == "🩸 Diabetes":
    st.markdown("## 🩸 Diabetes Prediction")
    st.markdown("Diabetes: A chronic condition in which the body cannot properly regulate blood sugar (glucose) levels.")
    st.markdown('<div class="info-card">Enter the patient\'s medical measurements below. All values should reflect fasting state where applicable.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes Risk"):
        model = load_model('diabetes_model.pkl')
        scaler = load_model('diabetes_scaler.pkl')
        if model is None:
            st.error("Model not found. Please run `python train_all_models.py` first.")
        else:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="result-box-danger">⚠️ High Risk of Diabetes &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-safe">✅ Low Risk of Diabetes &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)

            st.progress(float(probability))
            st.caption(f"Model confidence: {probability*100:.1f}% likelihood of diabetes")
            
    medical_term()
# ─────────────────────────────────────────────
# HEART DISEASE
# ─────────────────────────────────────────────
elif disease == "❤️ Heart Disease":
    st.markdown("## ❤️ Heart Disease Prediction")
    st.markdown("Heart Disease: A group of conditions that affect the heart's structure and function, reducing its ability to pump blood effectively.")
    st.markdown('<div class="info-card">Enter cardiovascular measurements. Chest pain type and ECG results require clinical assessment.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 130)
    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
    with col3:
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", -5.0, 10.0, 0.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    sex_enc = 1 if sex == "Male" else 0
    cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
    angina_enc = 1 if exercise_angina == "Y" else 0
    slope_map = {"Up": 0, "Flat": 1, "Down": 2}

    if st.button("🔍 Predict Heart Disease Risk"):
        model = load_model('heart_model.pkl')
        scaler = load_model('heart_scaler.pkl')
        if model is None:
            st.error("Model not found. Please run `python train_all_models.py` first.")
        else:
            input_data = np.array([[age, sex_enc, cp_map[chest_pain], resting_bp, cholesterol,
                                    fasting_bs, ecg_map[resting_ecg], max_hr, angina_enc,
                                    oldpeak, slope_map[st_slope]]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="result-box-danger">⚠️ High Risk of Heart Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-safe">✅ Low Risk of Heart Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)

            st.progress(float(probability))
            st.caption(f"Model confidence: {probability*100:.1f}% likelihood of heart disease")
    medical_term()
            
# ─────────────────────────────────────────────
# PARKINSON'S
# ─────────────────────────────────────────────
elif disease == "🧠 Parkinson's Disease":
    st.markdown("## 🧠 Parkinson's Disease Prediction")
    st.markdown("Parkinson's Disease: A progressive neurological disorder that affects movement, balance, and coordination due to the loss of dopamine-producing brain cells.")
    st.markdown('<div class="info-card">This model uses voice measurement biomarkers. Values are typically obtained from voice recording analysis software.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz) — Avg vocal freq", 80.0, 270.0, 154.2, step=0.1)
        fhi = st.number_input("MDVP:Fhi(Hz) — Max vocal freq", 100.0, 600.0, 197.1, step=0.1)
        flo = st.number_input("MDVP:Flo(Hz) — Min vocal freq", 60.0, 240.0, 116.3, step=0.1)
        jitter_pct = st.number_input("MDVP:Jitter(%)", 0.0, 0.1, 0.00622, step=0.0001, format="%.5f")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001, 0.0000440, step=0.00001, format="%.6f")
        rap = st.number_input("MDVP:RAP", 0.0, 0.1, 0.00317, step=0.0001, format="%.5f")
        ppq = st.number_input("MDVP:PPQ", 0.0, 0.1, 0.00345, step=0.0001, format="%.5f")
        ddp = st.number_input("Jitter:DDP", 0.0, 0.1, 0.00951, step=0.0001, format="%.5f")
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.02971, step=0.0001, format="%.5f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 2.0, 0.282, step=0.001)
        apq3 = st.number_input("Shimmer:APQ3", 0.0, 0.1, 0.01457, step=0.0001, format="%.5f")
        apq5 = st.number_input("Shimmer:APQ5", 0.0, 0.2, 0.01963, step=0.0001, format="%.5f")
        apq = st.number_input("MDVP:APQ", 0.0, 0.2, 0.02460, step=0.0001, format="%.5f")
        dda = st.number_input("Shimmer:DDA", 0.0, 0.2, 0.04371, step=0.0001, format="%.5f")
    with col3:
        nhr = st.number_input("NHR — Noise-to-Harmonics Ratio", 0.0, 0.5, 0.01491, step=0.0001, format="%.5f")
        hnr = st.number_input("HNR — Harmonics-to-Noise Ratio", 5.0, 40.0, 21.9, step=0.1)
        rpde = st.number_input("RPDE", 0.2, 0.7, 0.498, step=0.001)
        dfa = st.number_input("DFA", 0.5, 0.9, 0.718, step=0.001)
        spread1 = st.number_input("Spread1", -8.0, -2.0, -5.684, step=0.001)
        spread2 = st.number_input("Spread2", 0.0, 0.5, 0.227, step=0.001)
        d2 = st.number_input("D2", 1.0, 4.0, 2.382, step=0.001)
        ppe = st.number_input("PPE", 0.0, 0.5, 0.206, step=0.001)

    if st.button("🔍 Predict Parkinson's Risk"):
        model = load_model('parkinsons_model.pkl')
        scaler = load_model('parkinsons_scaler.pkl')
        if model is None:
            st.error("Model not found. Please run `python train_all_models.py` first.")
        else:
            input_data = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp,
                                    shimmer, shimmer_db, apq3, apq5, apq, dda,
                                    nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="result-box-danger">⚠️ High Risk of Parkinson\'s &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-safe">✅ Low Risk of Parkinson\'s &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)

            st.progress(float(probability))
    medical_term()

# ─────────────────────────────────────────────
# KIDNEY DISEASE
# ─────────────────────────────────────────────
elif disease == "🫘 Kidney Disease":
    st.markdown("## 🫘 Chronic Kidney Disease Prediction")
    st.markdown("Kidney Disease: A condition in which the kidneys become damaged and are unable to effectively filter waste and excess fluids from the blood.")
    st.markdown('<div class="info-card">Enter lab test values. Leave unknown values at 0 — the model handles missing data.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 0, 100, 45)
        bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
        sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, step=0.001, format="%.3f")
        al = st.number_input("Albumin (0-5)", 0, 5, 0)
        su = st.number_input("Sugar (0-5)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    with col2:
        ba = st.selectbox("Bacteria", ["present", "notpresent"])
        bgr = st.number_input("Blood Glucose Random (mgs/dl)", 0, 500, 120)
        bu = st.number_input("Blood Urea (mgs/dl)", 0, 400, 40)
        sc = st.number_input("Serum Creatinine (mgs/dl)", 0.0, 20.0, 1.2, step=0.1)
        sod = st.number_input("Sodium (mEq/L)", 100, 200, 137)
        pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, step=0.1)
        hemo = st.number_input("Hemoglobin (gms)", 3.0, 20.0, 14.0, step=0.1)
    with col3:
        pcv = st.number_input("Packed Cell Volume", 10, 60, 44)
        wc = st.number_input("White Blood Cell Count (cells/cumm)", 3000, 20000, 8000)
        rc = st.number_input("Red Blood Cell Count (millions/cmm)", 2.0, 8.0, 5.0, step=0.1)
        htn = st.selectbox("Hypertension", ["yes", "no"])
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
        cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["yes", "no"])
        ane = st.selectbox("Anemia", ["yes", "no"])

    enc = {"normal": 0, "abnormal": 1, "present": 1, "notpresent": 0,
           "yes": 1, "no": 0, "good": 0, "poor": 1}

    if st.button("🔍 Predict Kidney Disease Risk"):
        model = load_model('kidney_model.pkl')
        scaler = load_model('kidney_scaler.pkl')
        imputer = load_model('kidney_imputer.pkl')
        if model is None:
            st.error("Model not found. Please run `python train_all_models.py` first.")
        else:
            input_data = np.array([[age, bp, sg, al, su, enc[rbc], enc[pc], enc[pcc],
                                    enc[ba], bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                                    enc[htn], enc[dm], enc[cad], enc[appet], enc[pe], enc[ane]]])
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="result-box-danger">⚠️ High Risk of Chronic Kidney Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-safe">✅ Low Risk of Chronic Kidney Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)

            st.progress(float(probability))

    medical_term()

# ─────────────────────────────────────────────
# LIVER DISEASE
# ─────────────────────────────────────────────
elif disease == "🫀 Liver Disease":
    st.markdown("## 🫀 Liver Disease Prediction")
    st.markdown("Liver Disease: A disorder that impairs the liver's ability to perform vital functions such as detoxification, metabolism, and nutrient processing.")
    st.markdown('<div class="info-card">Enter liver function test (LFT) values from the patient\'s blood report.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.0, 80.0, 1.0, step=0.1)
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, 0.3, step=0.1)
    with col2:
        alk_phos = st.number_input("Alkaline Phosphotase (IU/L)", 50, 2200, 200)
        alamine = st.number_input("Alamine Aminotransferase (IU/L)", 5, 2100, 30)
        aspartate = st.number_input("Aspartate Aminotransferase (IU/L)", 5, 5000, 35)
    with col3:
        total_proteins = st.number_input("Total Proteins (g/dL)", 2.0, 10.0, 6.8, step=0.1)
        albumin = st.number_input("Albumin (g/dL)", 0.0, 6.0, 3.5, step=0.1)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 3.0, 1.0, step=0.01)

    gender_enc = 1 if gender == "Male" else 0

    if st.button("🔍 Predict Liver Disease Risk"):
        model = load_model('liver_model.pkl')
        scaler = load_model('liver_scaler.pkl')
        if model is None:
            st.error("Model not found. Please run `python train_all_models.py` first.")
        else:
            input_data = np.array([[age, gender_enc, total_bilirubin, direct_bilirubin,
                                    alk_phos, alamine, aspartate, total_proteins,
                                    albumin, ag_ratio]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="result-box-danger">⚠️ High Risk of Liver Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box-safe">✅ Low Risk of Liver Disease &nbsp;|&nbsp; Probability: {probability*100:.1f}%</div>', unsafe_allow_html=True)

            st.progress(float(probability))
            st.caption(f"Model confidence: {probability*100:.1f}% likelihood of liver disease")
    
    medical_term()
    
# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "MediPredict — AI-Driven Multi-Disease Prediction | For Educational Use Only"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "Copyright © - Narendra®. All Rights Reserved "
    "</div>",
    unsafe_allow_html=True
)
