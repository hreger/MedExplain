"""
MedExplain - Data Download

This module handles downloading and preparing the Pima Diabetes dataset.
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve

def download_pima_dataset():
    """Download and prepare the Pima Diabetes dataset."""
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Dataset URL from UCI Machine Learning Repository
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    
    # Column names for the dataset
    column_names = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age',
        'Outcome'
    ]
    
    # Download the dataset
    print("Downloading Pima Diabetes dataset...")
    file_path = os.path.join('data/raw', 'diabetes.csv')
    urlretrieve(url, file_path)
    
    # Read and save with proper column names
    df = pd.read_csv(file_path, header=None, names=column_names)
    df.to_csv(file_path, index=False)
    
    print(f"Dataset downloaded and saved to {file_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset columns:")
    for col in column_names:
        print(f"- {col}")
    
    return df

if __name__ == "__main__":
    download_pima_dataset() 