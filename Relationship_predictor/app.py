import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Relationship Probability", layout="centered")

# Get absolute path of this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- SAFE MODEL LOADER ---------------- #

def safe_load_model(filename, name):
    path = os.path.join(BASE_DIR, filename)
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(
            f"File for {name} not found.\n\n"
            f"Looked for: {path}\n\n"
            f"Make sure `{filename}` is in the SAME folder as `app.py` in your repo."
        )
        st.stop()
    except ModuleNotFoundError as e:
        st.error(f"Missing Python package while loading {name}: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {name} from {path}: {e}")
        st.stop()

# Load models from the same folder as app.py
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

    # message based on prob
    if prob >= 80:
        st.markdown(
            f"""
            <p style='text-align:center; font-size:1.1rem; color:#e0f2fe; margin-top:0.5rem;'>
                {name_display}, yeh toh elite tier scene hai.<br>
                <b>“Cupid ne bhi bola, OP life!”</b> 💘
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align:center; font-size:0.95rem; color:#cbd5f5;'>
                Catch line: <i>“Tumhara graph bhi smooth hai, aur vibe bhi.”</i>
            </p>
            """,
            unsafe_allow_html=True,
        )
    elif prob >= 60:
        st.markdown(
            f"""
            <p style='text-align:center; font-size:1.05rem; color:#bbf7d0; margin-top:0.5rem;'>
                {name_display}, strong chances hain, bas thoda sa right time, right person. 😉
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align:center; font-size:0.95rem; color:#d1fae5;'>
                Catch line: <i>“Scene bana hua hai, bas confirm hone ka wait hai.”</i>
            </p>
            """,
            unsafe_allow_html=True,
        )
    elif prob >= 40:
        st.markdown(
            f"""
            <p style='text-align:center; font-size:1.05rem; color:#fef9c3; margin-top:0.5rem;'>
                {name_display}, 50–50 ka scene hai, thoda effort doge toh story ban sakti hai. 🙂
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align:center; font-size:0.95rem; color:#fef3c7;'>
                Catch line: <i>“Chances hain, bas tum confident rehna.”</i>
            </p>
            """,
            unsafe_allow_html=True,
        )
    elif prob >= 20:
        st.markdown(
            f"""
            <p style='text-align:center; font-size:1.05rem; color:#fed7aa; margin-top:0.5rem;'>
                {name_display}, scene weak hai, par hopeless nahi. Work on yourself, baaki life dekh legi. 🙂
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align:center; font-size:0.95rem; color:#fed7aa;'>
                Catch line: <i>“Abhi story filler episode pe hai, climax baad mein aayega.”</i>
            </p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <p style='text-align:center; font-size:1.05rem; color:#fecaca; margin-top:0.5rem;'>
                {name_display}, iss time relationship graph se zyada tumhari self-growth ka chart important hai. 🌧️
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style='text-align:center; font-size:0.95rem; color:#fecaca;'>
                Catch line: <i>“Dil tumhara sahi hai, bas timing thodi galat chal rahi hai.”</i>
            </p>
            """,
            unsafe_allow_html=True,
        )

# legend
st.markdown(
    """
    <div style="margin-top:2.5rem;">
        <div style="font-size:0.9rem; color:#9ca3af; text-align:center; margin-bottom:0.3rem;">
            Color legend
        </div>
        <div style="display:flex; height:18px; border-radius:999px; overflow:hidden; border:1px solid #374151;">
            <div style="flex:1; background:#7f1d1d;"></div>
            <div style="flex:1; background:#9a3412;"></div>
            <div style="flex:1; background:#ca8a04;"></div>
            <div style="flex:1; background:#15803d;"></div>
            <div style="flex:1; background:#0e7490;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#9ca3af; margin-top:4px;">
            <span>0–20%: Dead scene</span>
            <span>20–40%: Weak</span>
            <span>40–60%: Maybe</span>
            <span>60–80%: Strong</span>
            <span>80–100%: Elite</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <hr style="margin-top:1.5rem; margin-bottom:0.5rem; border-color:#1f2937;">
    <p style='text-align:center; font-size:0.8rem; color:#6b7280;'>
        This is a fun ML demo, not real relationship advice. But haan, gym, GPA aur personality pe kaam karna kabhi waste nahi jaata 😌
    </p>
    """,
    unsafe_allow_html=True,
)
