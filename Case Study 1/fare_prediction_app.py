import streamlit as st
import pandas as pd
import joblib
from datetime import date, datetime
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Flight Fare Predictor | KUL Hub",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AirAsia Flight Fare Prediction System")
st.markdown("Predict flight ticket prices originating from **Kuala Lumpur (KUL)** based on route, schedule, and flight timing dynamics.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_fare_model.joblib"

# Route metadata used by the input controls.
# Destination metadata
domestic_airports = ['PEN', 'JHB', 'LGK', 'KTA', 'TGG', 'BKI', 'KCH', 'MYY', 'SDK', 'TWU']
international_airports = ['SIN', 'CGK', 'DPS', 'KNO', 'SUB']
all_airports = sorted(domestic_airports + international_airports)

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


model_bundle = load_model_bundle()
model = model_bundle["model"]
feature_cols = model_bundle["feature_cols"]
flight_num_map = model_bundle["flight_num_map"]

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("📋 Flight Parameters")

departure_station = st.sidebar.text_input("Departure Station", value="KUL (Kuala Lumpur)", disabled=True)

arrival_station = st.sidebar.selectbox(
    "Arrival Station", 
    options=all_airports,
    index=all_airports.index('DPS')
)

flight_number = st.sidebar.selectbox(
    "Flight Number",
    options=list(flight_num_map.keys())
)

flight_date = st.sidebar.date_input(
    "Flight Date",
    value=date(2026, 10, 15)
)

flight_time = st.sidebar.time_input(
    "Departure Time",
    value=datetime.strptime("12:00", "%H:%M").time()
)

# ---------------------------------------------------------
# Feature Processing Logic
# ---------------------------------------------------------
is_international = 1 if arrival_station in international_airports else 0
route_desc = "International" if is_international == 1 else "Domestic"

day_code = flight_date.weekday()
is_weekend = 1 if day_code in [4, 5, 6] else 0
day_of_month = flight_date.day

hour = flight_time.hour
minute = flight_time.minute

flight_num_encoded = flight_num_map.get(flight_number, 250.0)

# Build one-hot encoded vector for Arrival Station
# Excluding first alphabetical dummy column ('Arrival_BKI' for drop_first)
dummy_stations = [column for column in feature_cols if column.startswith("Arrival Station_")]

station_dummies = {col: 0.0 for col in dummy_stations}
target_col = f"Arrival Station_{arrival_station}"
if target_col in station_dummies:
    station_dummies[target_col] = 1.0

# Assemble model feature input row
input_features = {
    'Is_International': float(is_international),
    'Is_Weekend': float(is_weekend),
    'Day_Code': float(day_code),
    'Day_of_Month': float(day_of_month),
    'Hour': float(hour),
    'Minute': float(minute),
    'Flight_Num_Encoded': float(flight_num_encoded),
    **station_dummies
}

input_df = pd.DataFrame([input_features]).reindex(columns=feature_cols, fill_value=0.0)

# ---------------------------------------------------------
# Main UI Display & Prediction
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Origin", "KUL")
col2.metric("Destination", arrival_station)
col3.metric("Route Type", route_desc)

st.divider()

if st.button("🔮 Predict Flight Fare", use_container_width=True, type="primary"):
    predicted_fare = model.predict(input_df)[0]
    
    st.balloons()
    st.success("Fare Calculated Successfully!")
    
    st.markdown(
        f"""
        <div style="text-align: center; background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 10px;">
            <h3 style="margin: 0; color: #333;">Estimated One-Way Fare</h3>
            <h1 style="margin: 10px 0; color: #1E88E5; font-size: 42px;">MYR {predicted_fare:.2f}</h1>
            <p style="margin: 0; color: #666; font-size: 14px;">Flight: {flight_number} | Date: {flight_date.strftime('%d %b %Y')} at {flight_time.strftime('%H:%M')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )