from src.preprocessing import load_data, preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

def main():
    print(" Starting Diabetes Prediction Pipeline...")

    print(" Loading and preprocessing data...")
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print(" Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print(" Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n Accuracy: {accuracy:.4f}")
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/diabetes_model.pkl")
    print("\n Model saved to 'models/diabetes_model.pkl'.")

if __name__ == "__main__":
    main()
