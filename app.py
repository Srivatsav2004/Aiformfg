import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ==========================================================
#  PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="AI FormFG | Engine Health Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI FormFG — Engine Health & Failure Prediction Dashboard")
st.markdown(
    "This web app compares **Deep Learning** and **Machine Learning** models "
    "for predicting engine health based on input parameters."
)

# ==========================================================
#  LOAD MODELS
# ==========================================================
try:
    with open(os.path.join("model", "ml_model.pkl"), "rb") as f:
        ml_model = pickle.load(f)
    with open(os.path.join("model", "dl_model.pkl"), "rb") as f:
        dl_model = pickle.load(f)
    st.sidebar.success("✅ Both models loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ==========================================================
#  FEATURE LISTS
# ==========================================================
dl_features = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "Ps30", "phi",
    "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd",
    "W31", "W32", "T48", "SmFan", "SmLPC", "SmHPC", "X_cycles", "σ (sigma)"
]

ml_features = [
    'cycle_time', 'T24', 'T30', 'T50', 'P30', 'Nf',
    'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32'
]

# ==========================================================
#  SIDEBAR INPUTS
# ==========================================================
st.sidebar.header("📥 Input Engine Parameters")
st.sidebar.markdown("Enter the values for each feature:")

with st.sidebar.form("input_form"):
    user_inputs = {}
    for f in dl_features:
        user_inputs[f] = st.number_input(f, value=0.0, step=0.01, format="%.4f")
    acc1 = st.number_input("DL Model Accuracy (acc1)", value=0.85, step=0.01)
    acc2 = st.number_input("ML Model Accuracy (acc2)", value=0.80, step=0.01)
    submitted = st.form_submit_button("🚀 Run Predictions")

# ==========================================================
#  PREDICTION LOGIC
# ==========================================================
if submitted:
    try:
        # Prepare DL input (all features)
        dl_input = np.array([[user_inputs[f] for f in dl_features]], dtype=float)

        # Prepare ML input (only subset)
        ml_input = np.array([[user_inputs.get(f, 0) for f in ml_features]], dtype=float)

        # ---------------------------
        # Handle DL Model Input Shape
        # ---------------------------
        expected_shape = dl_model.input_shape  # e.g. (None, 30, 14)
        st.write("🧩 DL Model Expected Input Shape:", expected_shape)

        if len(expected_shape) == 3:
            # e.g. reshape (1, 26) → (1, 30, 14)
            time_steps = expected_shape[1]
            features_per_step = expected_shape[2]

            # Create a padded or repeated array to match shape
            dl_input = np.resize(dl_input, (1, time_steps, features_per_step))

        # Run predictions
        dl_pred = float(dl_model.predict(dl_input, verbose=0)[0][0])
        ml_pred = float(ml_model.predict(ml_input)[0])

        # Compute final weighted prediction
        total = acc1 + acc2
        weight_dl = acc1 / total
        weight_ml = acc2 / total
        final_pred = (dl_pred * weight_dl) + (ml_pred * weight_ml)

        # ===============================
        #  DISPLAY RESULTS
        # ===============================
        st.subheader("📊 Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Deep Learning Prediction", round(dl_pred, 4))
        col2.metric("ML (LightGBM) Prediction", round(ml_pred, 4))
        col3.metric("Weighted Final Prediction", round(final_pred, 4))
        st.success("✅ Prediction completed successfully!")

        # Charts below
        st.markdown("### 📈 Model Comparison")
        fig, ax = plt.subplots()
        models = ["Deep Learning", "ML Model", "Weighted Final"]
        values = [dl_pred, ml_pred, final_pred]
        ax.bar(models, values, color=["#2E86DE", "#F39C12", "#27AE60"])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

        # ======================================================
        #  VISUALIZATIONS
        # ======================================================
        st.markdown("### 📈 Model Output Comparison")

        fig, ax = plt.subplots()
        models = ["Deep Learning", "ML Model", "Weighted Final"]
        values = [dl_pred, ml_pred, final_pred]
        ax.bar(models, values, color=["#2E86DE", "#F39C12", "#27AE60"])
        ax.set_ylabel("Prediction Value")
        ax.set_title("Model Prediction Comparison")
        st.pyplot(fig)

        st.markdown("### ⚖️ Accuracy Weight Distribution")
        fig2, ax2 = plt.subplots()
        weights = [weight_dl, weight_ml]
        labels = ["DL Model Weight", "ML Model Weight"]
        ax2.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

        st.markdown("### 🧾 Input Summary")
        st.dataframe({
            "Feature": list(user_inputs.keys()),
            "Value": [round(v, 4) for v in user_inputs.values()]
        })

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ==========================================================
#  FOOTER
# ==========================================================
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | AI FormFG 2025 | All rights reserved")
