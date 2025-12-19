import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
print('Libraries Imported Successfully')


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained XGBoost model
model_path = os.path.join(script_dir, "flood_xgboost_model.pkl")
model = joblib.load(model_path)
print('Model Loaded Successfully')

# Load background image (optional — add your own image named "flood.jpg")
image_path = os.path.join(script_dir, "flood_image.jpg")
if os.path.exists(image_path):
    image = Image.open(image_path)
    print('Image Loaded Successfully')
else:
    print('No image found — skipping display')

# Title & Header
st.title("Flood Probability Prediction Model")
st.markdown("Real-time flood risk assessment")

if os.path.exists(image_path):
    st.image(image, use_column_width=True)

st.write("Enter the values below to predict **Flood Probability (0 = Safe, 1 = Certain Flood)**")

# Documentation
with st.expander("Documentation: Input Feature Descriptions (Scale 0-15)"):
    st.write("""
    **MonsoonIntensity** — Strength of monsoon rains  
    **TopographyDrainage** — How well water drains from land  
    **RiverManagement** — Quality of river control systems  
    **Deforestation** — Level of forest loss  
    **Urbanization** — City expansion rate  
    **ClimateChange** — Impact of global warming  
    **DamsQuality** — Condition of dams and reservoirs  
    **Siltation** — Soil buildup in rivers/lakes  
    **AgriculturalPractices** — Farming methods affecting runoff  
    **Encroachments** — Illegal building in flood zones  
    **IneffectiveDisasterPreparedness** — Poor planning/response  
    **DrainageSystems** — Urban drainage efficiency  
    **CoastalVulnerability** — Risk from sea level/tides  
    **Landslides** — Slope instability risk  
    **Watersheds** — Water catchment health  
    **DeterioratingInfrastructure** — Aging bridges/roads  
    **PopulationScore** — Population density in risk areas  
    **WetlandLoss** — Destruction of natural flood buffers  
    **InadequatePlanning** — Poor land-use policies  
    **PoliticalFactors** — Corruption or neglect in governance  
    """)
print('Documentation Set Successfully')

# Input Features
st.sidebar.header("Adjust Flood Risk Factors (0 = Low, 15 = Extreme)")

features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds', 'DeterioratingInfrastructure',
    'PopulationScore', 'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
]
print('Feature Values Created Successfully')

inputs = {}
for feature in features:
    value = st.sidebar.slider(
        feature.replace('_', ' '),
        min_value=0,
        max_value=15,
        value=5,
        step=1
    )
    inputs[feature] = value
print('Input Features Set Successfully')

# Prediction Button
if st.sidebar.button("Predict Flood Risk"):
    # Create input DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Predict
    probability = model.predict(input_df)[0]
    risk_percent = probability * 100

    # Display Results
    st.success(f"Predicted Flood Probability: **{probability:.4f}**")
    st.metric("Flood Risk", f"{risk_percent:.1f}%", delta=None)

    # Risk Level Summary
    if probability >= 0.75:
        st.error("EXTREME DANGER — Evacuation strongly recommended!")
    elif probability >= 0.6:
        st.error("HIGH RISK — Emergency preparations needed immediately")
    elif probability >= 0.5:
        st.warning("MODERATE RISK — Monitor closely and prepare")
    elif probability >= 0.3:
        st.info("ELEVATED RISK — Stay alert")
    else:
        st.info("LOW RISK — Normal conditions")

    # Show input summary
    with st.expander("View Your Input Values"):
        st.dataframe(input_df.T.rename(columns={0: "Score (0–15)"}))
print('Model Prediction Done Successfully')

# Footer
st.markdown("---")
st.caption("Powered by XGBoost | R² ≈ 0.977 | Trained on 500 thousand flood records")