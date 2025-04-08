## Diabetes Prediction using Machine Learning

This project is a machine learning pipeline that predicts whether a patient has diabetes using health-related attributes such as Glucose level, Blood Pressure, BMI, and more. It follows a modular and production-ready structure.

---

##  Project Structure

Diabetes-Prediction-Model
│ ├── Dashboard.py # Main Streamlit app for diabetes prediction
├── requirements.txt # Python dependencies
├── README.md # Project overview and documentation
├── .gitignore # Git ignored files (if any) │
├── data / └── diabetes.csv │ 
├── docs /Diabetes_Prediction_Report
├── models/ # (Optional) For saved model files (pkl, etc.) │ └── logistic_model.pkl │ └── assets/ 
dashboard_screenshot.png




##  Dataset

The dataset contains the following features:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (1: Diabetic, 0: Non-diabetic)

You can download the dataset from [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or place it in the `data/` folder as `diabetes.csv`.

---

##  How to Run

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd ML_Project


# Diabetes-Prediction-Modelgit init
