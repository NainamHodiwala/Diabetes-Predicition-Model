import joblib
from preprocessing import load_data, preprocess_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    model = joblib.load('../models/diabetes_model.pkl')

    y_pred = model.predict(X_test)

    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
