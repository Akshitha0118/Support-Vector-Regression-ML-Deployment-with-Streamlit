import streamlit as st
import pandas as pd
from sklearn.svm import SVR

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Salary Predictor", layout="centered")

# ---------- CSS ----------
st.markdown("""
<style>
body {background-color:#f6f8fc;}
.main {
    background:white;
    padding:2rem;
    border-radius:12px;
    box-shadow:0 8px 20px rgba(0,0,0,0.1);
}
h1 {color:#4B6BFB; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
data = pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# ---------- MODEL ----------
model = SVR(kernel="poly", degree=4, gamma="auto", C=6, epsilon=1.8)
model.fit(X, y)

# ---------- USER INPUT ----------
level = st.slider("Select Experience Level", 1.0, 10.0, 6.5, 0.1)

# ---------- PREDICTION ----------
if st.button("Predict Salary"):
    salary = model.predict([[level]])[0]
    st.success(f"💰 Predicted Salary: ₹ {salary:,.2f}")
