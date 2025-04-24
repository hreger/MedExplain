"""
MedExplain - Model Evaluation

This module handles model evaluation and performance metrics calculation.
It generates evaluation reports and visualizations for model performance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

def load_model_and_data(model_path, data_path):
    """
    Load the trained model and test data.
    
    Args:
        model_path (str): Path to the saved model
        data_path (str): Path to the processed data directory
        
    Returns:
        tuple: (model, X_test, y_test, feature_names)
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load test data
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    # Load feature names
    feature_names = joblib.load(os.path.join(data_path, 'feature_names.joblib'))
    
    return model, X_test, y_test, feature_names

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate various performance metrics.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        y_prob (numpy.ndarray): Predicted probabilities
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        output_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

def generate_report(model, X_test, y_test, output_dir):
    """
    Generate evaluation report with metrics and visualizations.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        output_dir (str): Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(output_dir, 'figures', 'confusion_matrix.png')
    )
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

def main():
    """Main function to run the evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', required=True, help='Path to the trained model')
    parser.add_argument('--data', required=True, help='Path to the processed data')
    parser.add_argument('--output', required=True, help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data(args.model, args.data)
    
    # Generate evaluation report
    generate_report(model, X_test, y_test, args.output)

if __name__ == '__main__':
    main() 