# ❤️ AI Heart Disease Prediction Dashboard

A professional Machine Learning project for predicting heart disease risk using clinical parameters. Built with Python, Streamlit, and Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📁 Project Structure

```
AI Health Risk Prediction Dashboard/
├── app.py                    # Streamlit dashboard application
├── train_models.py           # ML model training script
├── preprocess.py             # Data cleaning and preprocessing
├── download_data.py          # Dataset download utility
├── heart_disease_dataset.csv # Heart disease dataset
├── best_model.pkl            # Saved best model (generated after training)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone or Download

```bash
git clone <repository-url>
cd "AI Health Risk Prediction Dashboard"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

If you have `heart_disease_dataset.csv` in the project folder, skip this step. Otherwise, run:

```bash
python download_data.py
```

This downloads the UCI Heart Disease dataset and saves it as `heart_disease_dataset.csv`.

### 4. Train Models

```bash
python train_models.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression, Decision Tree, and Random Forest models
- Compare accuracy and select the best model
- Save the best model to `best_model.pkl`

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the dashboard.

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)

### Features (13 input columns)

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Sex (1 = male, 0 = female) | Binary |
| cp | Chest pain type (1-4) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl (1 = yes) | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina (1 = yes) | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment (1-3) | Categorical |
| ca | Number of major vessels (0-3) | Categorical |
| thal | Thalassemia (3, 6, 7) | Categorical |

### Target

- **heart_disease**: 0 = No heart disease, 1 = Heart disease present

---

## 🤖 Machine Learning Pipeline

1. **Data Cleaning**: Handle missing values, outliers, invalid entries
2. **Preprocessing**: Standard scaling for numeric features
3. **Train-Test Split**: 80-20 stratified split
4. **Models Trained**:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Model Selection**: Best model by test accuracy
6. **Persistence**: Best model saved with pickle

---

## 🖥️ Dashboard Features

- **Patient Input Form**: Enter all 13 clinical parameters
- **Real-time Prediction**: Get heart disease risk prediction
- **Probability Display**: See probability of heart disease (%)
- **Visualizations**:
  - Target distribution
  - Age vs Heart Disease
  - Cholesterol vs Heart Disease
  - Feature correlation heatmap

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

---

## 📄 License

MIT License - feel free to use and modify for your projects.

---

## 👤 Author

AI Heart Disease Prediction Dashboard - GitHub Ready Project
