"""
Data Download Module - Fetches UCI Heart Disease dataset and saves as heart_disease_dataset.csv
Run this script if you don't have the dataset or need to refresh it.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# UCI Heart Disease column names (14 attributes)
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heart_disease'
]

UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'


def download_heart_disease_data():
    """Download and prepare the UCI Heart Disease dataset."""
    print("Downloading UCI Heart Disease dataset...")
    
    try:
        df = pd.read_csv(
            UCI_URL,
            names=COLUMNS,
            na_values='?',
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Using fallback: creating sample data...")
        return create_sample_data()
    
    # Convert to numeric
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Binary target: 0 = no disease, 1 = disease (1,2,3,4 -> 1)
    df['heart_disease'] = (df['heart_disease'] > 0).astype(int)
    
    # Drop rows with missing values
    df = df.dropna()
    
    output_path = Path('heart_disease_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} ({len(df)} rows)")
    return df


def create_sample_data():
    """Create sample data when download fails."""
    np.random.seed(42)
    n = 250
    
    data = {
        'age': np.random.randint(29, 78, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(1, 5, n),
        'trestbps': np.random.randint(94, 200, n),
        'chol': np.random.randint(126, 564, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(71, 202, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
        'slope': np.random.randint(1, 4, n),
        'ca': np.random.randint(0, 4, n),
        'thal': np.random.choice([3, 6, 7], n),
    }
    
    # Create target with some correlation to features
    risk_score = (data['age'] / 70 + data['chol'] / 300 + data['trestbps'] / 150) / 3
    data['heart_disease'] = (risk_score + np.random.randn(n) * 0.2 > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    output_path = Path('heart_disease_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path} ({len(df)} rows)")
    return df


if __name__ == '__main__':
    download_heart_disease_data()
