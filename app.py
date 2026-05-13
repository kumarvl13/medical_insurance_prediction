import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="MediPredict",
    page_icon="💊",
    layout="wide"
)

# Load Model
model = joblib.load("models/insurance_model.pkl")

rf_model = model.named_steps["model"]

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.stButton>button {
    background: linear-gradient(to right, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💊 MediPredict")
st.subheader("AI-Based Medical Insurance Cost Predictor")

st.write("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 25)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Children", 0, 10, 0)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

# Prediction Button
if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_data)[0]

    st.write("## Prediction Result")

    st.markdown(f"""
    <div class="metric-card">
        <h2>Estimated Insurance Cost</h2>
        <h1>${prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    feature_importance = model.named_steps[
        "model"
    ].feature_importances_

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    # Plotly Bar Chart
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation='h',
        title="Feature Importance Analysis",
        text_auto=".3f"
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    #Insurance Cost Meter
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Insurance Cost"},
        gauge={
            'axis': {'range': [0, 50000]},
            'bar': {'thickness': 0.3},
            'steps': [
                {'range': [0, 10000], 'color': "green"},
                {'range': [10000, 25000], 'color': "orange"},
                {'range': [25000, 50000], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Risk Category
    if prediction < 5000:
        risk = "Low Risk"
    elif prediction < 15000:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.success(f"Risk Category: {risk}")


    # Insights
    st.write("## AI Insights")

    insights = []

    if smoker == "yes":
        insights.append(
            "Smoking heavily increases insurance premiums."
        )

    if bmi > 30:
        insights.append(
            "High BMI contributes to increased health risk."
        )

    if age > 50:
        insights.append(
            "Higher age may increase expected medical expenses."
        )

    if children > 3:
        insights.append(
            "More dependents may influence insurance planning."
        )

    for insight in insights:
        st.info(insight)