"""
MedExplain - Model Training

This module handles the training of disease prediction models using various ML algorithms
like XGBoost, RandomForest, etc. It also tracks experiments using MLflow.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

def load_and_preprocess_data(data_path):
    """
    Load the preprocessed data from .npy files.
    
    Args:
        data_path (str): Path to the processed data directory
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # Load the preprocessed data
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    # Load feature names
    feature_names = joblib.load(os.path.join(data_path, 'feature_names.joblib'))
    
    return X_train, X_test, y_train, y_test, feature_names

def train_model(X_train, y_train, model_type="xgboost", params=None, 
                cv=5, random_state=42):
    """
    Train a machine learning model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): Type of model to train ("xgboost" or "random_forest")
        params (dict): Model hyperparameters
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        object: Trained model
    """
    if params is None:
        # Default parameters
        if model_type == "xgboost":
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': random_state
            }
        else:  # random_forest
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': random_state
            }
    
    # Create and train model
    if model_type == "xgboost":
        model = xgb.XGBClassifier(**params)
    else:  # random_forest
        model = RandomForestClassifier(**params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

def save_model(model, model_path, feature_names=None):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path (str): Path to save the model
        feature_names: List of feature names
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    
    # If feature names are provided, save them as well
    if feature_names is not None:
        feature_path = os.path.join(os.path.dirname(model_path), 'feature_names.joblib')
        joblib.dump(feature_names, feature_path)
    
    print(f"Model saved to {model_path}")

def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data', required=True, help='Directory containing processed data')
    parser.add_argument('--output', required=True, help='Output directory for model')
    
    args = parser.parse_args()
    
    # Set up logging
    mlflow.set_experiment("diabetes_prediction")
    
    # Define paths
    model_path = os.path.join(args.output, 'model.joblib')
    
    # Model parameters
    model_type = "xgboost"  # or "random_forest"
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(args.data)
        
        # Train model
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, feature_names)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save model
        save_model(model, model_path, feature_names)
        
        # Log model in MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save metrics to JSON file
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model training completed with metrics: {metrics}")

if __name__ == "__main__":
    # This will only run if the script is executed directly
    main()

