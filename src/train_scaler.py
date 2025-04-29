from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

def train_and_save_scaler():
    # Load your training data
    # Using PIMA dataset as mentioned in README
    data = pd.read_csv('data/diabetes.csv')  # Adjust path as needed
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler
    scaler.fit(data.drop('Outcome', axis=1))
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.joblib')

if __name__ == "__main__":
    train_and_save_scaler()