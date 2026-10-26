"""
Preprocessing Module for AI Heart Disease Prediction Dashboard
Handles data cleaning, preprocessing, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Column definitions for UCI Heart Disease dataset
FEATURE_COLUMNS = [
    'age',           # Age in years
    'sex',           # Sex (1 = male; 0 = female)
    'cp',            # Chest pain type (1-4)
    'trestbps',      # Resting blood pressure (mm Hg)
    'chol',          # Serum cholesterol (mg/dl)
    'fbs',           # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg',       # Resting electrocardiographic results (0-2)
    'thalach',       # Maximum heart rate achieved
    'exang',         # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak',       # ST depression induced by exercise
    'slope',         # Slope of peak exercise ST segment (1-3)
    'ca',            # Number of major vessels colored by fluoroscopy (0-3)
    'thal'           # Thalassemia: 3 = normal; 6 = fixed defect; 7 = reversible defect
]

TARGET_COLUMN = 'heart_disease'


def load_data(filepath: str = 'heart_disease_dataset.csv') -> pd.DataFrame:
    """
    Load the heart disease dataset from CSV.
    Supports both raw UCI format and preprocessed format.
    """
    df = pd.read_csv(filepath)
    
    # Handle raw UCI format (no headers or different column names)
    if 'num' in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={'num': TARGET_COLUMN})
    
    # Handle column naming if using raw UCI numeric indices
    if df.columns[0] == '63' or not any(c in df.columns for c in FEATURE_COLUMNS[:3]):
        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: handle missing values, outliers, and data types.
    """
    df = df.copy()
    
    # Replace '?' with NaN for missing values
    df = df.replace('?', np.nan)
    
    # Convert numeric columns
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Convert target to binary: 0 = no disease, 1 = disease (1,2,3,4 -> 1)
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
    
    # Handle invalid categorical values
    # thal: valid values 3, 6, 7 - replace invalid with mode
    if 'thal' in df.columns:
        valid_thal = [3, 6, 7]
        df.loc[~df['thal'].isin(valid_thal), 'thal'] = df['thal'].mode()[0]
    
    # ca: valid values 0, 1, 2, 3
    if 'ca' in df.columns:
        valid_ca = [0, 1, 2, 3]
        df.loc[~df['ca'].isin(valid_ca), 'ca'] = df['ca'].mode()[0]
    
    # Remove impossible values
    df = df[df['age'] > 0]
    df = df[df['age'] <= 120]
    df = df[df['trestbps'] >= 80]
    df = df[df['trestbps'] <= 250]
    df = df[df['chol'] >= 100]
    df = df[df['chol'] <= 600]
    df = df[df['thalach'] >= 60]
    df = df[df['thalach'] <= 220]
    
    return df.reset_index(drop=True)


def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    """
    Preprocess features: scale numerical values and prepare for ML.
    Returns: X, y, scaler (fitted), feature_names
    """
    df = df.copy()
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, scaler, FEATURE_COLUMNS


def get_train_test_split(filepath: str = 'heart_disease_dataset.csv', 
                         test_size: float = 0.2, 
                         random_state: int = 42):
    """
    Load, clean, preprocess data and return train-test split.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_columns
    """
    # Load and clean data
    df = load_data(filepath)
    df_clean = clean_data(df)
    
    # Preprocess
    X, y, scaler, feature_columns = preprocess_data(df_clean)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns


def get_preprocessing_pipeline(filepath: str = 'heart_disease_dataset.csv'):
    """
    Returns cleaned dataframe and fitted scaler for use in training and app.
    """
    df = load_data(filepath)
    df_clean = clean_data(df)
    _, _, scaler, feature_columns = preprocess_data(df_clean)
    return df_clean, scaler, feature_columns


if __name__ == '__main__':
    # Test preprocessing pipeline
    print("Testing preprocessing pipeline...")
    X_train, X_test, y_train, y_test, scaler, features = get_train_test_split()
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features: {features}")
    print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
    print("Preprocessing complete!")
