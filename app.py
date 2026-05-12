import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="MediPredict",
    page_icon="💊",
    layout="wide"
)

# Load Model
model = joblib.load("models/insurance_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

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

    # Risk Category
    if prediction < 5000:
        risk = "Low Risk"
    elif prediction < 15000:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.success(f"Risk Category: {risk}")

    # =========================
    # SHAP Explainability
    # =========================

    st.write("## AI Explainability using SHAP")

    # Preprocess input
    processed_input = model.named_steps[
        "preprocessor"
    ].transform(input_data)

    # SHAP Values
    shap_values = explainer.shap_values(processed_input)

    # Feature Names
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    # SHAP DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    })

    shap_df["AbsImpact"] = np.abs(shap_df["Impact"])

    shap_df = shap_df.sort_values(
        by="AbsImpact",
        ascending=False
    )

    st.write("### Feature Contribution")

    # Plotly Bar Chart
    import plotly.express as px

    fig = px.bar(
        shap_df.head(10),
        x="Impact",
        y="Feature",
        orientation='h',
        title="Top Features Affecting Prediction"
    )

    st.plotly_chart(fig, use_container_width=True)


    # SHAP Waterfall Plot
    st.write("### SHAP Waterfall Explanation")

    # Create Explanation Object
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value[0],
        data=processed_input[0],
        feature_names=feature_names
    )

    # Plot
    fig2, ax = plt.subplots(figsize=(10, 5))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    st.pyplot(fig2)

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