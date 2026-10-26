"""
Model Training Module for AI Heart Disease Prediction Dashboard
Trains Logistic Regression, Decision Tree, and Random Forest models.
Compares accuracy and saves the best model using pickle.
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import get_train_test_split, FEATURE_COLUMNS, TARGET_COLUMN


def train_models(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train multiple ML models and return trained models with their metrics.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state, max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=random_state, max_depth=10
        )
    }
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Confusion Matrix:\n{results[name]['confusion_matrix']}")
    
    return results


def get_best_model(results):
    """
    Select the model with highest accuracy.
    """
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    return best_name, results[best_name]['model']


def save_model(model, scaler, filepath='best_model.pkl'):
    """
    Save the best model and scaler using pickle.
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {filepath}")


def load_model(filepath='best_model.pkl'):
    """
    Load saved model and scaler from pickle file.
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def main():
    """
    Main training pipeline: load data, train models, compare, save best.
    """
    print("=" * 50)
    print("AI Heart Disease Prediction - Model Training")
    print("=" * 50)
    
    # Load and split data
    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_columns = get_train_test_split()
    
    # Train models
    print("\nTraining models...")
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Compare and select best
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    for name, res in results.items():
        print(f"{name}: {res['accuracy']:.4f}")
    
    best_name, best_model = get_best_model(results)
    print(f"\nBest Model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")
    
    # Save best model
    save_model(best_model, scaler)
    
    return results, best_name, best_model


if __name__ == '__main__':
    main()
