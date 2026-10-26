"""
AI Heart Disease Prediction Dashboard
Streamlit application for predicting heart disease risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI HEART DISEASE PREDICTION DASHBOARD",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    model_path = Path('best_model.pkl')
    if not model_path.exists():
        st.error("Model not found! Please run `python train_models.py` first.")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


@st.cache_data
def load_dataset():
    """Load dataset for visualizations."""
    try:
        df = pd.read_csv('heart_disease_dataset.csv')
        return df
    except Exception:
        return None


def get_user_input():
    """Collect user input for prediction."""
    st.sidebar.header("📋 Patient Information")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", ["Female (0)", "Male (1)"], index=1)
        cp = st.selectbox(
            "Chest Pain Type",
            ["Typical angina (1)", "Atypical angina (2)", "Non-anginal (3)", "Asymptomatic (4)"],
            index=0
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"], index=0)
        restecg = st.selectbox(
            "Resting ECG",
            ["Normal (0)", "ST-T wave abnormality (1)", "Left ventricular hypertrophy (2)"],
            index=0
        )
    
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"], index=0)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "ST Segment Slope",
            ["Upsloping (1)", "Flat (2)", "Downsloping (3)"],
            index=1
        )
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], index=0)
        thal = st.selectbox(
            "Thalassemia",
            ["Normal (3)", "Fixed defect (6)", "Reversible defect (7)"],
            index=0
        )
    
    # Convert to numeric
    sex_val = 1 if "Male" in sex else 0
    cp_val = int(cp.split("(")[1].replace(")", ""))
    fbs_val = 1 if "Yes" in fbs else 0
    restecg_val = int(restecg.split("(")[1].replace(")", ""))
    exang_val = 1 if "Yes" in exang else 0
    slope_val = int(slope.split("(")[1].replace(")", ""))
    thal_val = int(thal.split("(")[1].replace(")", ""))
    
    return [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
            thalach, exang_val, oldpeak, slope_val, ca, thal_val]


def predict(model_data, user_input):
    """Make prediction and return result with probability."""
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Scale input
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Get probability (for classifiers that support it)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        prob_heart_disease = proba[1]  # Probability of class 1 (disease)
    else:
        prob_heart_disease = float(prediction)
    
    return prediction, prob_heart_disease


def create_visualizations(df):
    """Create and display data visualizations."""
    if df is None or len(df) < 10:
        return
    
    st.subheader("📊 Data Exploratory Analysis")
    
    # Target distribution
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        if 'heart_disease' in df.columns:
            df['heart_disease'].value_counts().plot(kind='bar', ax=ax, color=['#4caf50', '#f44336'])
            ax.set_title('Heart Disease Distribution')
            ax.set_xlabel('Heart Disease (0=No, 1=Yes)')
            ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if 'age' in df.columns and 'heart_disease' in df.columns:
            sns.boxplot(data=df, x='heart_disease', y='age', ax=ax, palette=['#4caf50', '#f44336'])
            ax.set_title('Age vs Heart Disease')
        st.pyplot(fig)
        plt.close()
    
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        if 'chol' in df.columns and 'heart_disease' in df.columns:
            sns.boxplot(data=df, x='heart_disease', y='chol', ax=ax, palette=['#4caf50', '#f44336'])
            ax.set_title('Cholesterol vs Heart Disease')
        st.pyplot(fig)
        plt.close()
    
    # Correlation heatmap
    st.markdown("#### Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        plt.close()


def main():
    """Main application entry point."""
    # Header
    st.markdown('<p class="main-header">❤️ AI Heart Disease Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.info("Run `python train_models.py` in the terminal to train and save the model.")
        return
    
    # Get user input
    user_input = get_user_input()
    
    # Prediction section
    st.sidebar.markdown("---")
    if st.sidebar.button("🔮 Predict", use_container_width=True):
        with st.spinner("Analyzing..."):
            prediction, prob_heart_disease = predict(model_data, user_input)
        
        st.balloons()
        
        risk_class = "high-risk" if prediction == 1 else "low-risk"
        result_text = "⚠️ **High Risk** - Heart disease detected" if prediction == 1 else "✅ **Low Risk** - No heart disease detected"
        
        st.markdown(f'''
        <div class="prediction-box {risk_class}">
            <h3>Prediction Result</h3>
            <p>{result_text}</p>
            <p><strong>Probability of Heart Disease:</strong> {prob_heart_disease*100:.2f}%</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Main content area - tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Visualizations", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        ### Welcome to the AI Heart Disease Prediction Dashboard
        
        This application uses machine learning to predict the risk of heart disease based on patient clinical parameters.
        
        **How to use:**
        1. Enter patient information in the sidebar
        2. Click **Predict** to get the risk assessment
        3. View probability and explore visualizations
        
        **Features used for prediction:**
        - **Age** - Patient age in years
        - **Sex** - Male/Female
        - **Chest Pain Type** - Type of chest pain
        - **Blood Pressure** - Resting blood pressure
        - **Cholesterol** - Serum cholesterol level
        - **Fasting Blood Sugar** - Blood sugar fasting status
        - **Resting ECG** - Electrocardiogram results
        - **Max Heart Rate** - Maximum heart rate achieved
        - **Exercise Angina** - Exercise induced angina
        - **ST Depression** - ST depression value
        - **Slope** - Slope of peak exercise ST segment
        - **Major Vessels** - Number of major vessels
        - **Thalassemia** - Thalassemia test result
        """)
    
    with tab2:
        df = load_dataset()
        create_visualizations(df)
    
    with tab3:
        st.markdown("""
        ### About This Project
        
        **AI Heart Disease Prediction Dashboard** is a machine learning project built with:
        - **Python** - Core programming language
        - **Streamlit** - Web application framework
        - **Scikit-learn** - Machine learning models (Logistic Regression, Decision Tree, Random Forest)
        - **Matplotlib & Seaborn** - Data visualizations
        
        **Dataset:** UCI Heart Disease Dataset (Cleveland)
        
        **Models Trained:**
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier
        
        The best performing model is automatically selected and saved for predictions.
        
        ⚠️ **Disclaimer:** This tool is for educational and research purposes only. 
        Always consult healthcare professionals for medical decisions.
        """)


if __name__ == '__main__':
    main()
