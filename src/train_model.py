from sklearn.linear_model import LogisticRegression
import joblib
from preprocessing import load_data, preprocess_data
import os

def train():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/diabetes_model.pkl')
    print(" Model trained and saved successfully.")

if __name__ == "__main__":
    train()
