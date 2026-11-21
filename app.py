import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CardioPredict | Heart Disease Risk Prediction System",
    page_icon=":material/cardiology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_model.joblib")
    preprocessor = joblib.load("heart_disease_preprocessor.joblib")
    return model, preprocessor


model, preprocessor = load_artifacts()


# ============================================================
# DESIGN SYSTEM (CSS)
# Palette: ink #12263A | clinical teal #0F766E | mist #F2F6F8
#          line #D9E2E7 | alert #B42318 | caution #B54708 | ok #067647
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #12263A;
    --muted: #5B6B7A;
    --teal: #0F766E;
    --teal-dark: #0B5C56;
    --teal-soft: #E3F2F0;
    --mist: #F2F6F8;
    --line: #D9E2E7;
    --alert: #B42318;
    --alert-soft: #FDECEA;
    --caution: #B54708;
    --caution-soft: #FEF3E2;
    --ok: #067647;
    --ok-soft: #E6F4EC;
}

html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp input, .stApp textarea, .stApp select,
.stApp button, .stApp [data-baseweb="select"] div {
    font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
}

/* Keep Streamlit's icon font so icons render as icons, not text */
.stApp [data-testid="stIconMaterial"],
.stApp .material-symbols-rounded,
.stApp [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

.stApp { background-color: var(--mist); }

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Typography ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: var(--ink) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.stApp p, .stApp li { color: var(--muted); }

/* ---------- Top banner ---------- */
.app-banner {
    gap: 20px;
    background: var(--ink);
    border-radius: 12px;
    padding: 26px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    margin-bottom: 22px;
    border-left: 6px solid var(--teal);
}
.app-banner h1 {
    color: #FFFFFF !important;
    font-size: 1.7rem;
    margin: 0 0 4px 0;
    padding: 0;
}
.app-banner p {
    color: #B9C7D3 !important;
    margin: 0;
    font-size: 0.98rem;
}
.app-banner .badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #E6EEF4;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
    white-space: nowrap;
}

/* ---------- Section cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid var(--line) !important;
    border-radius: 12px;
    box-shadow: 0 1px 2px rgba(18, 38, 58, 0.04);
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2px 0 2px 0;
    color: var(--ink);
    font-size: 1.08rem;
    font-weight: 600;
}
.section-title .ico {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    border-radius: 8px;
    background: var(--teal-soft);
    color: var(--teal);
}
.section-title .ico svg { width: 18px; height: 18px; }

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 54px; height: 54px;
    border-radius: 12px;
    background: var(--teal);
    color: #FFFFFF;
    flex: none;
}
.side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.side-brand .brand-mark { width: 38px; height: 38px; border-radius: 9px; }
.side-brand .brand-mark svg { width: 22px; height: 22px; }
.side-brand .name { color: #FFFFFF !important; font-size: 1.3rem; font-weight: 700; }
.idle-box .icon { color: var(--teal); }
.section-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin-bottom: 10px;
}

/* ---------- Inputs ---------- */
[data-testid="stWidgetLabel"] p {
    color: var(--ink) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background-color: #FBFCFD !important;
    border-radius: 8px !important;
    border-color: var(--line) !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.18) !important;
}

/* ---------- Button ---------- */
.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    background-color: var(--teal);
    color: #FFFFFF;
    border: none;
    transition: background-color 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--teal-dark);
    color: #FFFFFF;
    border: none;
}
.stButton > button:focus-visible {
    outline: 3px solid rgba(15, 118, 110, 0.35);
    outline-offset: 2px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: var(--ink);
}
section[data-testid="stSidebar"] * { color: #E6EEF4; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

.side-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}
.side-card .k { color: #9FB2C2 !important; font-size: 0.78rem; display: block; }
.side-card .v { color: #FFFFFF !important; font-weight: 600; }

/* ---------- Result panel ---------- */
.result-card {
    border-radius: 12px;
    padding: 22px 22px 20px 22px;
    border: 1px solid var(--line);
    background: #FFFFFF;
}
.result-card.low  { border-top: 6px solid var(--ok); }
.result-card.mod  { border-top: 6px solid var(--caution); }
.result-card.high { border-top: 6px solid var(--alert); }
.result-card.idle { border-top: 6px solid var(--line); }

.result-label { color: var(--muted); font-size: 0.85rem; margin-bottom: 2px; }
.result-big {
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.05;
    color: var(--ink);
}
.result-status {
    display: inline-block;
    margin-top: 10px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.low  .result-status { background: var(--ok-soft);      color: var(--ok); }
.mod  .result-status { background: var(--caution-soft); color: var(--caution); }
.high .result-status { background: var(--alert-soft);   color: var(--alert); }

.result-text { margin-top: 14px; color: var(--muted); font-size: 0.93rem; line-height: 1.5; }

/* risk scale bar */
.gauge { margin-top: 18px; }
.gauge-track {
    position: relative;
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg,
        #2E9E6B 0%, #2E9E6B 35%,
        #E3A03C 35%, #E3A03C 65%,
        #D0453A 65%, #D0453A 100%);
}
.gauge-marker {
    position: absolute;
    top: -5px;
    width: 4px;
    height: 22px;
    background: var(--ink);
    border-radius: 2px;
    transform: translateX(-2px);
    box-shadow: 0 0 0 2px #FFFFFF;
}
.gauge-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.78rem;
}

.idle-box { text-align: center; padding: 26px 10px; }
.idle-box .icon { font-size: 2.4rem; }
.idle-box p { margin: 8px 0 0 0; font-size: 0.93rem; }

/* ---------- Summary table ---------- */
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.summary-table td {
    padding: 9px 4px;
    border-bottom: 1px solid var(--line);
    color: var(--ink);
}
.summary-table tr:last-child td { border-bottom: none; }
.summary-table td.k { color: var(--muted); }
.summary-table td.v { text-align: right; font-weight: 600; }
.flag {
    display: inline-block;
    margin-left: 8px;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.flag.ok   { background: var(--ok-soft);      color: var(--ok); }
.flag.warn { background: var(--caution-soft); color: var(--caution); }
.flag.bad  { background: var(--alert-soft);   color: var(--alert); }

/* ---------- Disclaimer ---------- */
.disclaimer {
    background: var(--caution-soft);
    border: 1px solid #F3D9AE;
    border-left: 5px solid var(--caution);
    border-radius: 10px;
    padding: 14px 18px;
    color: #6B3A0B;
    font-size: 0.9rem;
    line-height: 1.5;
}
.disclaimer b { color: #6B3A0B; }

hr { border-color: var(--line); }

@media (max-width: 800px) {
    .app-banner { flex-direction: column; align-items: flex-start; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

ICONS = {
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
    "chart": '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
}


def icon(name, size=20):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def section_header(title, subtitle, icon_name="activity"):
    st.markdown(
        f'<div class="section-title"><span class="ico">{icon(icon_name)}</span>{title}</div>'
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def flag(text, kind):
    return f'<span class="flag {kind}">{text}</span>'


def bp_flag(v):
    if v < 120:
        return flag("Normal", "ok")
    if v < 140:
        return flag("Elevated", "warn")
    return flag("High", "bad")


def chol_flag(v):
    if v < 200:
        return flag("Desirable", "ok")
    if v < 240:
        return flag("Borderline", "warn")
    return flag("High", "bad")


def risk_level(pct):
    if pct < 35:
        return "low", "Lower risk"
    if pct < 65:
        return "mod", "Moderate risk"
    return "high", "Higher risk"


def gauge_html(pct):
    pct = max(0.0, min(100.0, pct))
    return (
        '<div class="gauge">'
        '<div class="gauge-track">'
        f'<div class="gauge-marker" style="left:{pct}%"></div>'
        '</div>'
        '<div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>'
        '</div>'
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f'<div class="side-brand"><span class="brand-mark">{icon("heart", 22)}</span>'
        '<span class="name">CardioPredict</span></div>',
        unsafe_allow_html=True,
    )
    

    st.divider()

    st.subheader("About the system")
    st.write(
        "CardioPredict is a machine learning system that estimates the "
        "likelihood of heart disease from a patient's clinical data."
    )

    st.divider()

    st.subheader("Model information")
    st.markdown(
        '<div class="side-card"><span class="k">Algorithm</span>'
        '<span class="v">Random Forest Classifier</span></div>'
        '<div class="side-card"><span class="k">Task</span>'
        '<span class="v">Binary classification</span></div>'
        '<div class="side-card"><span class="k">Evaluation metrics</span>'
        '<span class="v">Accuracy, Precision, Recall, F1</span></div>'
        '<div class="side-card"><span class="k">Test accuracy</span>'
        '<span class="v">≈ 86%</span></div>'
        '<div class="side-card"><span class="k">Deployment</span>'
        '<span class="v">Streamlit</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    


# ============================================================
# BANNER
# ============================================================

st.markdown(
    '<div class="app-banner">'
    f'<div class="brand-mark">{icon("heart", 30)}</div>'
    '<div style="flex:1">'
    '<h1>Heart Disease Risk Assessment</h1>'
    '<p>Enter the patient\'s clinical information to generate a machine learning risk estimate.</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT: FORM (left)  |  RESULT (right)
# ============================================================

form_col, result_col = st.columns([2.1, 1], gap="large")


with form_col:

    # ---------------- Patient information ----------------
    with st.container(border=True):
        section_header("Patient information", "Basic demographic details and presenting symptoms.", "user")

        c0, c1 = st.columns(2)
        with c0:
            patient_id = st.text_input("Patient name / ID (optional)", placeholder="e.g. P-0001")
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)

        c2, c3 = st.columns(2)
        with c2:
            sex = st.selectbox("Sex", ["Male", "Female"])
        with c3:
            cp = st.selectbox(
                "Chest pain type",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            )

    # ---------------- Cardiac measurements ----------------
    with st.container(border=True):
        section_header("Cardiac measurements", "Vital signs and laboratory values.", "heart")

        c1, c2, c3 = st.columns(3)
        with c1:
            trestbps = st.number_input(
                "Resting blood pressure (mmHg)",
                min_value=50.0, max_value=250.0, value=120.0,
            )
        with c2:
            chol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=50.0, max_value=700.0, value=200.0,
            )
        with c3:
            thalch = st.number_input(
                "Maximum heart rate (bpm)",
                min_value=50.0, max_value=250.0, value=150.0,
            )

    # ---------------- Clinical information ----------------
    with st.container(border=True):
        section_header("Clinical information", "Diagnostic tests and exercise-related findings.", "stethoscope")

        c1, c2, c3 = st.columns(3)
        with c1:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", ["True", "False"])
        with c2:
            restecg = st.selectbox(
                "Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"]
            )
        with c3:
            exang = st.selectbox("Exercise-induced angina", ["True", "False"])

    # ---------------- Additional measurements ----------------
    with st.container(border=True):
        section_header("Additional measurements", "Stress-test and imaging results.", "chart")

        c1, c2 = st.columns(2)
        with c1:
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0, max_value=10.0, value=1.0,
            )
        with c2:
            slope = st.selectbox(
                "ST segment slope", ["upsloping", "flat", "downsloping"]
            )

        c3, c4 = st.columns(2)
        with c3:
            ca = st.number_input(
                "Major vessels coloured by fluoroscopy (0–4)",
                min_value=0.0, max_value=4.0, value=0.0,
            )
        with c4:
            thal = st.selectbox(
                "Thalassemia", ["normal", "fixed defect", "reversible defect"]
            )

    st.caption("All fields except patient name / ID are used for the prediction.")
    predict_clicked = st.button("Run risk assessment", icon=":material/monitor_heart:")


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalch": [thalch],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal],
    })

    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]

    st.session_state["result"] = {
        "prediction": int(prediction[0]),
        "risk_pct": float(probability * 100),
        "patient_id": patient_id.strip(),
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
    }


# ============================================================
# RESULT PANEL
# ============================================================

with result_col:

    st.markdown("### Assessment result")

    result = st.session_state.get("result")

    if result is None:

        st.markdown(
            '<div class="result-card idle"><div class="idle-box">'
            f'<div class="icon">{icon("stethoscope", 40)}</div>'
            '<p>No assessment yet.<br>Complete the patient form and select '
            '<b>Run risk assessment</b>.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    else:

        pct = result["risk_pct"]
        level_class, level_text = risk_level(pct)

        if result["prediction"] == 1:
            headline = "Heart disease risk detected"
            guidance = (
                "The model classifies this profile as positive for heart disease. "
                "Findings like this should be reviewed with a qualified clinician."
            )
        else:
            headline = "Lower heart disease risk"
            guidance = (
                "The model classifies this profile as negative for heart disease. "
                "A negative result does not rule out the condition."
            )

        patient_line = (
            f'<div class="result-label">Patient: {result["patient_id"]}</div>'
            if result["patient_id"] else ""
        )

        st.markdown(
            f'<div class="result-card {level_class}">'
            f'{patient_line}'
            '<div class="result-label">Estimated probability of heart disease</div>'
            f'<div class="result-big">{pct:.1f}%</div>'
            f'<span class="result-status">{level_text}</span>'
            f'{gauge_html(pct)}'
            f'<div class="result-text"><b style="color:#12263A">{headline}.</b> {guidance}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        with st.container(border=True):
            section_header("Input summary", "Values used, with general reference ranges.", "clipboard")
            st.markdown(
                '<table class="summary-table">'
                f'<tr><td class="k">Age / Sex</td><td class="v">{result["age"]} · {result["sex"]}</td></tr>'
                f'<tr><td class="k">Chest pain</td><td class="v">{result["cp"]}</td></tr>'
                f'<tr><td class="k">Blood pressure</td>'
                f'<td class="v">{result["trestbps"]:.0f} mmHg{bp_flag(result["trestbps"])}</td></tr>'
                f'<tr><td class="k">Cholesterol</td>'
                f'<td class="v">{result["chol"]:.0f} mg/dL{chol_flag(result["chol"])}</td></tr>'
                f'<tr><td class="k">Max heart rate</td><td class="v">{result["thalch"]:.0f} bpm</td></tr>'
                f'<tr><td class="k">ST depression</td><td class="v">{result["oldpeak"]:.1f}</td></tr>'
                '</table>',
                unsafe_allow_html=True,
            )


# ============================================================
# MODEL INFO + DISCLAIMER
# ============================================================

st.write("")
st.divider()

st.markdown("### About the prediction model")
st.write(
    "This application uses a Random Forest classification model trained on "
    "heart disease patient data. The model achieved approximately 86% test "
    "accuracy in the project evaluation."
)