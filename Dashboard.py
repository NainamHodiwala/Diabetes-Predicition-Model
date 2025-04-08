import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("Diabetes Prediction App")
st.write("This app uses **Logistic Regression** to predict whether a person is likely to have diabetes.")

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

st.subheader("Dataset Overview")
st.dataframe(df.head())

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

acc = accuracy_score(y_test, model.predict(X_test_scaled))
st.markdown(f"**Model Used:** Logistic Regression  \n**Accuracy on test data:** {acc:.2f}")

st.subheader("Predict Diabetes")

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=float(df[col].mean()), step=1.0 if df[col].dtype != 'int' else 1)
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The person is likely to have diabetes.")
    else:
        st.success("The person is unlikely to have diabetes.")
    
    st.markdown(f"**Probability of having diabetes:** `{probability:.2f}`")
