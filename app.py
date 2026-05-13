import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
model = joblib.load("models/insurance_model.pkl")

# Page config
st.set_page_config(
    page_title="Insurance Predictor",
    layout="wide"
)

st.title("💊 Medical Insurance Cost Prediction")

# Inputs
age = st.slider("Age", 18, 100, 25)

bmi = st.slider("BMI", 10.0, 50.0, 25.0)

children = st.slider("Children", 0, 10, 0)

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Insurance Cost: ${prediction:,.2f}"
    )

    # Simple chart
    chart_df = pd.DataFrame({
        "Factor": ["Age", "BMI", "Children"],
        "Value": [age, bmi, children]
    })

    fig = px.bar(
        chart_df,
        x="Factor",
        y="Value",
        title="User Health Factors"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )