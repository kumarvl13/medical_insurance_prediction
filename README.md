# 💊 Explainable AI-Based Medical Insurance Premium Prediction System

## 📌 Project Overview

The Explainable AI-Based Medical Insurance Premium Prediction System is a machine learning-powered web application that predicts medical insurance costs based on various personal and health-related factors such as:

- Age
- BMI (Body Mass Index)
- Smoking habits
- Gender
- Number of children
- Residential region

This project helps users understand how different health and lifestyle factors influence insurance premiums while assisting insurance companies in analyzing customer risk profiles.

The system uses Machine Learning algorithms for accurate prediction and integrates Explainable AI using SHAP (SHapley Additive Explanations) to provide transparency and interpretability of model predictions.

The web application is developed using Streamlit with an interactive and modern dashboard UI.

---

# 🚀 Features

- ✅ Medical insurance cost prediction
- ✅ Interactive Streamlit web application
- ✅ Explainable AI using SHAP
- ✅ Feature importance visualization
- ✅ Risk analysis and premium insights
- ✅ Interactive charts and graphs
- ✅ Real-time prediction system
- ✅ User-friendly modern dashboard

---

# 🧠 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Explainable AI Integration
8. Web Application Development
9. Deployment

---

# 📂 Project Structure

```text
medical_insurance_project/
│
├── dataset/
│   └── insurance.csv
│
├── models/
│   ├── insurance_model.pkl
│   └── shap_explainer.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── style.css
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| SHAP | Explainable AI |
| Plotly | Interactive Visualization |
| Matplotlib | Graph Plotting |
| Streamlit | Web Application Framework |
| Joblib | Model Serialization |

---

# 📊 Dataset Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance cost (Target Variable) |

---

# ⚙️ Installation

## Step 1 — Clone the Repository

```bash
git clone <repository_link>
```

---

## Step 2 — Navigate to Project Folder

```bash
cd medical_insurance_project
```

---

## Step 3 — Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train the Machine Learning Model

```bash
python train_model.py
```

This generates:
- `insurance_model.pkl`
- `shap_explainer.pkl`

---

## Run Streamlit Application

```bash
streamlit run app.py
```

Then open the local URL displayed in the terminal.

---

# 📈 Model Used

The project uses:

## 🌲 Random Forest Regressor

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Advantages
- High accuracy
- Handles nonlinear relationships
- Robust against overfitting
- Works well on tabular datasets

---

# 🔍 Explainable AI using SHAP

The project integrates SHAP (SHapley Additive Explanations) to explain model predictions.

SHAP helps users understand:
- Which features increase insurance cost
- Which features reduce insurance cost
- Feature contribution impact
- Model decision transparency

### SHAP Visualizations
- Feature Importance Plot
- SHAP Waterfall Plot
- Prediction Explanation

---

# 📊 Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- R² Score

---

# 💡 Example Prediction Factors

The insurance premium prediction depends on factors such as:

- Smoking habits
- BMI level
- Age
- Number of children
- Region
- Gender

For example:
- Smokers generally receive higher insurance premiums.
- Higher BMI may increase predicted medical expenses.
- Older individuals may have higher insurance costs.

---

# 🌐 Streamlit Dashboard Features

- Interactive user inputs
- Real-time prediction
- SHAP explainability
- Risk category analysis
- Interactive charts
- Responsive modern UI

---

# 📷 Future Enhancements

- User Authentication System
- PDF Report Generation
- Database Integration
- Advanced ML Models (XGBoost)
- Cloud Deployment
- Insurance Plan Recommendation System
- Chatbot Integration
- Health Score Analysis

---

# 🎯 Applications

- Insurance Premium Estimation
- Healthcare Risk Assessment
- Predictive Analytics Research
- Explainable AI Demonstration
- Educational Data Science Projects

---

# 📚 Learning Outcomes

Through this project, the following concepts were implemented:

- Machine Learning Regression
- Data Preprocessing
- Feature Engineering
- Explainable AI (XAI)
- Model Deployment
- Data Visualization
- Streamlit Web Development

---

# 👨‍💻 Author

Kumar

Bachelor of Engineering (B.E)

Data Science & Machine Learning Enthusiast

---

# 📄 License

This project is developed for educational and academic purposes.
