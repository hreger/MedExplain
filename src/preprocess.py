"""
MedExplain - Data Preprocessing

This module handles data preprocessing for the MedExplain application.
It includes functions for data cleaning, feature engineering, and data splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """
    Load the dataset from the specified path.
    
    Args:
        data_path (str): Path to the dataset file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    return pd.read_csv(data_path)

def preprocess_data(df):
    """
    Preprocess the dataset.
    
    Args:
        df (pandas.DataFrame): Input dataset
        
    Returns:
        tuple: (X, y, feature_names)
    """
    # Basic preprocessing
    # Remove any rows with missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    return X, y, feature_names

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def scale_features(X_train, X_test):
    """
    Scale the features using StandardScaler.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def main():
    """Main function to run the preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess medical data')
    parser.add_argument('--input', required=True, help='Input data directory')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess data
    df = load_data(os.path.join(args.input, 'diabetes.csv'))
    X, y, feature_names = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save processed data
    np.save(os.path.join(args.output, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(args.output, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(args.output, 'y_train.npy'), y_train)
    np.save(os.path.join(args.output, 'y_test.npy'), y_test)
    
    # Save feature names
    import joblib
    joblib.dump(feature_names, os.path.join(args.output, 'feature_names.joblib'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(args.output, 'scaler.joblib'))

if __name__ == '__main__':
    main() 