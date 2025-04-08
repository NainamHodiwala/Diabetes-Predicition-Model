import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filename='diabetes.csv'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    data_path = os.path.join(base_dir, 'data', filename)

    print(f" Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" Dataset not found at path: {data_path}")
    
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

    return X_train_scaled, X_test_scaled, y_train, y_test
