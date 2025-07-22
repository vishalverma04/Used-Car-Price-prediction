import streamlit as st
import pandas as pd
import pickle
import json

# Load model and encoders
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('car_dict.json', 'r') as f:
    car_dict = json.load(f)

# Transmission options from your training data
transmission_options = [
    '6-Speed A/T', '8-Speed A/T', 'A/T', '7-Speed A/T', 'F',
    'Transmission w/Dual Shift Mode', '6-Speed M/T', '10-Speed A/T',
    '9-Speed A/T', '5-Speed A/T', 'A/T CVT', 'CVT-F',
    '6-Speed A/T with Auto-Shift', 'M/T', 'CVT Transmission', '4-Speed A/T',
    '8-Speed A/T with Auto-Shift', '8-SPEED AT', '5-Speed M/T', 'Variable', '2',
    'A/T, 9-Speed 9G-Tronic', 'A/T, 8-Speed',
    'A/T, 8-Speed Sport w/Sport & M/T Modes', 'Auto, 6-Speed w/CmdShft',
    'Transmission Overdrive Switch',
    'A/T, 8-Speed M STEPTRONIC w/Drivelogic, Sport & M/T Modes',
    '7-Speed A/T with Auto-Shift', '6-Speed', 'A/T, 10-Speed', '1-Speed A/T',
    '7-Speed M/T', 'A/T, 7-Speed S tronic Dual-Clutch', 'M/T, 6-Speed',
    '6-Speed Electronically Controlled A/T with O', '6 Speed At/Mt',
    'SCHEDULED FOR OR IN PRODUCTION', '8-Speed M/T'
]

# Pre-grouped transmission options
manual_transmissions = [x for x in transmission_options if 'M/T' in x]
automatic_transmissions = [x for x in transmission_options if 'A/T' in x or 'Auto' in x]
cvt_transmissions = [x for x in transmission_options if 'CVT' in x or 'Variable' in x]
other_transmissions = list(set(transmission_options) - set(manual_transmissions) - set(automatic_transmissions) - set(cvt_transmissions))

# Now define tm_dict after categorizing
tm_dict = {
    'Manual': {
        '5-Speed': [x for x in manual_transmissions if '5-Speed' in x],
        '6-Speed': [x for x in manual_transmissions if '6-Speed' in x],
        '7-Speed': [x for x in manual_transmissions if '7-Speed' in x],
        '8-Speed': [x for x in manual_transmissions if '8-Speed' in x],
    },
    'Automatic': {
        '4-Speed': [x for x in automatic_transmissions if '4-Speed' in x],
        '5-Speed': [x for x in automatic_transmissions if '5-Speed' in x],
        '6-Speed': [x for x in automatic_transmissions if '6-Speed' in x],
        '7-Speed': [x for x in automatic_transmissions if '7-Speed' in x],
        '8-Speed': [x for x in automatic_transmissions if '8-Speed' in x],
        '9-Speed': [x for x in automatic_transmissions if '9-Speed' in x],
        '10-Speed': [x for x in automatic_transmissions if '10-Speed' in x],
        '1-Speed': [x for x in automatic_transmissions if '1-Speed' in x],
    },
    'CVT': {
        'CVT': cvt_transmissions
    },
    'Other': {
        'Other': other_transmissions
    }
}

# Streamlit UI
st.title("🚗 Used Car Price Predictor")

# Step 1: Brand selection
brand = st.selectbox("Select Brand", sorted(car_dict.keys()))

# Step 2: Fuel Type
fuel_type = st.selectbox("Select Fuel Type", sorted(car_dict[brand].keys()))

# Step 3: Model
model_name = st.selectbox("Select Model", sorted(car_dict[brand][fuel_type]))

# Step 4: Year, Mileage
year = st.number_input("Enter Model Year", min_value=1990, max_value=2025, step=1)
milage_km = st.number_input("Enter Mileage (in kilometers)", min_value=0.0, step=100.0)
milage_meters = milage_km * 1000  # Convert to meters

# Step 5: Transmission (via grouped UI)
trans_cat = st.selectbox("Select Transmission Type", list(tm_dict.keys()))
trans_speed = st.selectbox("Select Gear Speed", list(tm_dict[trans_cat].keys()))
transmission = st.selectbox("Select Transmission Option", tm_dict[trans_cat][trans_speed])

accident_map = {
    "No": "None reported",
    "Yes": "At least 1 accident or damage reported"
}

accident_input = st.selectbox("Accident History", ["No", "Yes"])
accident = accident_map[accident_input]

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'brand': [brand],
        'model': [model_name],
        'model_year': [year],
        'milage': [milage_meters],
        'fuel_type': [fuel_type],
        'transmission': [transmission],
        'accident': [accident]
    })

    # Label Encoding
    for col in ['brand', 'model', 'fuel_type', 'transmission', 'accident']:
        le = label_encoders[col]
        if input_df[col][0] not in le.classes_:
            st.error(f"{col.capitalize()} '{input_df[col][0]}' not found in training data.")
            st.stop()
        input_df[col] = le.transform(input_df[col])

    # Prediction
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Selling Price: ₹{predicted_price:,.2f}")
