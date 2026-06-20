import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model and scaler ──────────────────────────────
model  = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ── Load dataset for dropdown options ─────────────────
df_raw = pd.read_excel('crop_yield_india^.csv.xlsx')
df_raw['Season'] = df_raw['Season'].str.strip()

states   = sorted(df_raw['State_Name'].dropna().unique().tolist())
crops    = sorted(df_raw['Crop'].dropna().unique().tolist())
seasons  = sorted(df_raw['Season'].dropna().unique().tolist())
districts = sorted(df_raw['District_Name'].dropna().unique().tolist())

# ── Label encoding maps (must match training) ──────────
from sklearn.preprocessing import LabelEncoder

df_enc = df_raw.copy()
le_state    = LabelEncoder()
le_district = LabelEncoder()
le_crop     = LabelEncoder()
le_season   = LabelEncoder()

df_enc['State_Name']    = le_state.fit_transform(df_enc['State_Name'].astype(str))
df_enc['District_Name'] = le_district.fit_transform(df_enc['District_Name'].astype(str))
df_enc['Crop']          = le_crop.fit_transform(df_enc['Crop'].astype(str))
df_enc['Season']        = le_season.fit_transform(df_enc['Season'].astype(str))

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="centered"
)

# ── Header ─────────────────────────────────────────────
st.title("🌾 Crop Yield Prediction System")
st.markdown("#### Predict crop yield based on farming conditions across India")
st.markdown("---")

# ── Input form ─────────────────────────────────────────
st.subheader("Enter Crop Details")

col1, col2 = st.columns(2)

with col1:
    state    = st.selectbox("State", states)
    crop     = st.selectbox("Crop", crops)
    season   = st.selectbox("Season", seasons)
    district = st.selectbox("District", districts)

with col2:
    crop_year    = st.number_input("Crop Year",        min_value=1990, max_value=2030, value=2024)
    area         = st.number_input("Area (hectares)",  min_value=0.1,  max_value=500000.0, value=100.0)
    temperature  = st.number_input("Temperature (°C)", min_value=0.0,  max_value=50.0,     value=28.0)
    humidity     = st.number_input("Humidity (%)",     min_value=0.0,  max_value=100.0,    value=60.0)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0,   value=40.0)

st.markdown("---")

# ── Predict button ─────────────────────────────────────
if st.button("🔍 Predict Yield", use_container_width=True):

    # Encode inputs using the same label encoders
    try:
        state_enc    = le_state.transform([state])[0]
        district_enc = le_district.transform([district])[0]
        crop_enc     = le_crop.transform([crop])[0]
        season_enc   = le_season.transform([season])[0]
    except Exception:
        st.error("Encoding error — please select valid options.")
        st.stop()

    # Build input row (must match X column order from training)
    input_data = pd.DataFrame([[
        state_enc, district_enc, crop_year, season_enc,
        crop_enc, temperature, humidity, soil_moisture, area
    ]], columns=[
        'State_Name', 'District_Name', 'Crop_Year', 'Season',
        'Crop', 'Temperature', 'Humidity', 'Soil_Moisture', 'Area'
    ])

    # Predict (model was trained on unscaled data)
    log_pred  = model.predict(input_data)[0]
    yield_pred = np.expm1(log_pred)  # reverse log1p transform

    # ── Display result ──────────────────────────────────
    st.markdown("### 🌱 Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Yield",  f"{yield_pred:,.1f} kg/ha")
    col_b.metric("Crop",             crop)
    col_c.metric("State",            state)

    # Yield category
    if yield_pred < 500:
        st.warning("⚠️ Low yield predicted. Consider better irrigation or fertilizer.")
    elif yield_pred < 2000:
        st.info("ℹ️ Moderate yield predicted. Conditions are average.")
    else:
        st.success("✅ High yield predicted. Excellent farming conditions!")

    st.markdown("---")
    st.caption("Note: Prediction is based on historical Indian agriculture data. "
               "Actual yield may vary based on local conditions.")