import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Relationship Probability", layout="centered")

# ---------------- SAFE MODEL LOADER ---------------- #

def safe_load(path, name):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"❌ File not found: '{path}'. Upload this file into the SAME folder (or correct subfolder).")
        st.stop()
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing Python package while loading {name}: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading {name}: {e}")
        st.stop()

# --------- CHANGE THE PATH HERE (IMPORTANT) ---------- #
MODEL_XGB_PATH = "models/xgb_model.pkl"
MODEL_CAT_PATH = "models/cat_model.pkl"

xgb = safe_load(MODEL_XGB_PATH, "XGBoost model")
cat = safe_load(MODEL_CAT_PATH, "CatBoost model")

# ----------------------------------------------------- #

BRANCH_MAP = {
    "BIOTECH": 0,
    "CE": 1,
    "CSE": 2,
    "ECE": 3,
    "IT": 4,
    "ME": 5
}

DEFAULTS = {f"F{i}": 5.0 for i in range(1, 34)}
DEFAULTS["F1"] = 20.0
DEFAULTS["F2"] = 170.0
DEFAULTS["F3"] = 60.0
DEFAULTS["F4"] = 2
DEFAULTS["F5"] = 0
DEFAULTS["F6"] = 5.0
DEFAULTS["F26"] = 500.0
DEFAULTS["F27"] = 100.0
DEFAULTS["F28"] = 3.0

# ----------------- STYLING ----------------- #

base_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: #020617;
    color: #e5e7eb;
    transition: background 0.6s ease;
}
[data-testid="stSidebar"] {
    background: #020617;
}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>❤️ Relationship Probability Predictor</h1>", unsafe_allow_html=True)

# ---------------- INPUTS ---------------- #

name = st.text_input("Enter your Name", placeholder="Kartik, Priya, etc.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (F1)", 16, 35, 20)
    height = st.slider("Height in cm (F2)", 140, 200, 170)
    weight = st.slider("Weight in kg (F3)", 40, 100, 60)

with col2:
    gym = st.slider("Gym frequency per week (F4)", 0, 7, 2)
    branch = st.selectbox("Branch (F5)", list(BRANCH_MAP.keys()))
    social = st.slider("Social Score (F6)", 0, 10, 5)

# Fill defaults
row = DEFAULTS.copy()
row["F1"] = float(age)
row["F2"] = float(height)
row["F3"] = float(weight)
row["F4"] = int(gym)
row["F5"] = BRANCH_MAP[branch]
row["F6"] = float(social)

df = pd.DataFrame([row])

if st.button("Calculate my relationship probability 💘"):

    try:
        p1 = float(xgb.predict(df)[0])
        p2 = float(cat.predict(df)[0])
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

    prob = np.clip(0.6 * p1 + 0.4 * p2, 0, 100)

    st.markdown(f"<h2 style='text-align:center;'>Probability: {prob:.2f}%</h2>", unsafe_allow_html=True)
