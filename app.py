import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI FormFG | Failure Prediction", layout="wide")
st.title("🔧 AI FormFG — Engine Failure Prediction System")

# ----------------------------
# LOAD MODELS
# ----------------------------
try:
    with open("model/ml_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    dl_model = load_model("model/dl_model.keras")
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ----------------------------
# FEATURE LIST — ensure it matches DL model training
# ----------------------------
features = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "Ps30", "phi",
    "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31",
    "W32", "T48", "SmFan", "SmLPC", "SmHPC", "X_cycles"
    # ❗ remove or add "σ (sigma)" ONLY if your DL model input_shape says (None, 26)
]

# ----------------------------
# SIDEBAR INPUT FORM
# ----------------------------
with st.sidebar.form("input_form"):
    st.header("📥 Enter Engine Parameters")
    inputs = {}
    for ftr in features:
        inputs[ftr] = st.number_input(ftr, value=0.0, step=0.01, format="%.4f")
    submitted = st.form_submit_button("🚀 Predict")

# ----------------------------
# PREDICTION LOGIC
# ----------------------------
if submitted:
    try:
        # Convert inputs to numpy array
        input_values = np.array([[inputs[f] for f in features]])

        # --- ML Model Prediction ---
        ml_pred = ml_model.predict(input_values)[0]

        # --- DL Model Prediction ---
        dl_pred = float(dl_model.predict(input_values, verbose=0)[0][0])

        # Display results
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("LightGBM Prediction", round(ml_pred, 4))

        with col2:
            st.metric("Deep Learning Prediction", round(dl_pred, 4))

        st.success("✅ Prediction completed successfully!")

        # Optional: show model input shape for debugging
        st.write("🧩 DL Model Expected Input Shape:", dl_model.input_shape)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | AI FormFG 2025")
