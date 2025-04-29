import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_model():
    # Load data
    data = pd.read_csv('data/diabetes.csv')
    
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/diabetes_model.joblib')

if __name__ == "__main__":
    train_and_save_model()