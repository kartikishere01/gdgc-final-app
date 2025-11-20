import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1) Page config
st.set_page_config(page_title="Relationship Probability", layout="centered")

# 2) Find the folder where THIS app.py is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Debug info - you can remove later
st.write("📂 Running from folder:", BASE_DIR)
st.write("📄 Files in this folder:", os.listdir(BASE_DIR))

# 3) Safe model loader that uses absolute path
def safe_load_model(filename, name):
    full_path = os.path.join(BASE_DIR, filename)
    try:
        model = joblib.load(full_path)
        return model
    except FileNotFoundError:
        st.error(
            f"❌ File for {name} not found.\n\n"
            f"Tried to load: {full_path}\n\n"
            f"Make sure `{filename}` is in the SAME folder as `app.py` in your GitHub repo."
        )
        st.stop()
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing Python package while loading {name}: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading {name} from {full_path}: {e}")
        st.stop()

# 4) Load models from same folder as app.py
xgb = safe_load_model("xgb_model.pkl", "XGBoost model")
cat = safe_load_model("cat_model.pkl", "CatBoost model")

# ---------------- CONSTANTS ---------------- #

BRANCH_MAP = {
    "BIOTECH": 0,
    "CE": 1,
    "CSE": 2,
    "ECE": 3,
    "IT": 4,
    "ME": 5
}

DEFAULTS = {
    "F1": 20.0,
    "F2": 170.0,
    "F3": 60.0,
    "F4": 2,
    "F5": 0,
    "F6": 5.0,
    "F7": 0,
    "F8": 0,
    "F9": 0,
    "F10": 0,
    "F11": 5.0,
    "F12": 5.0,
    "F13": 0,
    "F14": 5.0,
    "F15": 5.0,
    "F16": 5.0,
    "F17": 5.0,
    "F18": 5.0,
    "F19": 5.0,
    "F20": 5.0,
    "F21": 5.0,
    "F22": 5.0,
    "F23": 5.0,
    "F24": 5.0,
    "F25": 5.0,
    "F26": 500.0,
    "F27": 100.0,
    "F28": 3.0,
    "F29": 5.0,
    "F30": 5.0,
    "F31": 5.0,
    "F32": 5.0,
    "F33": 5.0,
}

# ---------------- STYLING ---------------- #

base_css = """
<style>
body {
    margin: 0;
    padding: 0;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617, #020617);
    color: #e5e7eb;
    transition: background 0.6s ease;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: #020617;
}
h1, h2, h3, h4, h5, h6 {
    color: #f9fafb;
}
.block-container {
    padding-top: 2rem;
}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <span style="font-size: 0.9rem; letter-spacing: 0.15em; text-transform: uppercase; color:#f97373;">
            GDGC NITJ • Love Analytics Lab
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="text-align:center; font-size:2.6rem;">
        ❤️ Relationship Probability Predictor
    </h1>
    <p style="text-align:center; font-size:1rem; color:#9ca3af;">
        Minimal inputs, maximum emotional damage.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------- INPUTS ---------------- #

name = st.text_input("Enter your Name", placeholder="Kartik, Priya, etc.")

st.markdown(
    "<h3 style='margin-top:1.5rem;'>Tell us just the basics</h3>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (F1)", min_value=16, max_value=35, value=20)
    height = st.slider("Height in cm (F2)", min_value=140, max_value=200, value=170)
    weight = st.slider("Weight in kg (F3)", min_value=40, max_value=100, value=60)

with col2:
    gym_freq = st.slider("Gym frequency per week (F4)", min_value=0, max_value=7, value=2)
    branch_name = st.selectbox("Branch (F5)", list(BRANCH_MAP.keys()))
    social_score = st.slider("Social vibe score (F6)", min_value=0, max_value=10, value=5)

# prepare input row
input_row = DEFAULTS.copy()
input_row["F1"] = float(age)
input_row["F2"] = float(height)
input_row["F3"] = float(weight)
input_row["F4"] = int(gym_freq)
input_row["F5"] = BRANCH_MAP[branch_name]
input_row["F6"] = float(social_score)

input_df = pd.DataFrame([input_row])

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Calculate my relationship probability 💘")

# ---------------- PREDICTION ---------------- #

if predict_btn:
    name_display = name.strip() if name and name.strip() else "You"

    try:
        p1 = xgb.predict(input_df)[0]
        p2 = cat.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error while predicting with the models: {e}")
        st.stop()

    prob = 0.6 * float(p1) + 0.4 * float(p2)
    prob = float(np.clip(prob, 0, 100))

    # background based on probability
    if prob < 20:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #7f1d1d, #020617);
        }
        </style>
        """
    elif prob < 40:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #9a3412, #020617);
        }
        </style>
        """
    elif prob < 60:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #ca8a04, #020617);
        }
        </style>
        """
    elif prob < 80:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #15803d, #020617);
        }
        </style>
        """
    else:
        bg_css = """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #0e7490, #020617);
        }
        </style>
        """

    st.markdown(bg_css, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='text-align:center;'>Probability for {name_display}: {prob:.2f}%</h2>",
        unsafe_allow_html=True,
    )

# legend and footer as earlier can stay same if you want
