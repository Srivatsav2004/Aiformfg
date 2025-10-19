import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ==========================================================
#  PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="AI For MFG | Engine Health Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI For MFG — Engine Health & Failure Prediction Dashboard")
st.markdown(
    "This app predicts engine health metrics using both Deep Learning and Machine Learning models."
)

# ==========================================================
#  FIXED RMSE VALUES
# ==========================================================
DL_RMSE = 17.70599
ML_RMSE = 51.072

inv_dl = 1 / DL_RMSE
inv_ml = 1 / ML_RMSE
weight_dl = inv_dl / (inv_dl + inv_ml)
weight_ml = inv_ml / (inv_dl + inv_ml)

# ==========================================================
#  LOAD MODELS
# ==========================================================
try:
    with open(os.path.join("model", "ml_model.pkl"), "rb") as f:
        ml_model = pickle.load(f)
    with open(os.path.join("model", "dl_model.pkl"), "rb") as f:
        dl_model = pickle.load(f)
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ==========================================================
#  FEATURE LISTS
# ==========================================================
dl_features = [
    "unit_ID", "cycles", "setting_1", "setting_2", "setting_3", "T2", "T24", "T30", "T50",
    "P2", "P15", "P30", "Nf", "Nc", "Ps30", "phi", "NRf", "NRc", "BPR", "farB",
    "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"
]

ml_features = [
    'cycle_time', 'T24', 'T30', 'T50', 'P30', 'Nf',
    'Nc', 'Ps30', 'phi', 'NRf', 'BPR', 'htBleed', 'W31', 'W32'
]

# ==========================================================
#  PREFILLED SAMPLE ROW
# ==========================================================
prefilled_row = {
    "unit_ID": 1,
    "cycles": 1,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0,
    "T2": 518.67,
    "T24": 641.82,
    "T30": 1589.70,
    "T50": 1400.60,
    "P2": 14.62,
    "P15": 21.61,
    "P30": 556.24,
    "Nf": 2388.00,
    "Nc": 8138.62,
    "Ps30": 8.4195,
    "phi": 0.03,
    "NRf": 392.0,
    "NRc": 2388.0,
    "BPR": 100.0,
    "farB": 39.06,
    "htBleed": 23.4190,
    "Nf_dmd": 7000.0,
    "PCNfR_dmd": 0.9,
    "W31": 150.0,
    "W32": 140.0,
    "cycle_time": 1  # Added for ML feature
}

# ==========================================================
#  SIDEBAR INPUT FORM
# ==========================================================
st.sidebar.header("📥 Input Engine Parameters")
st.sidebar.markdown("Adjust the values below if needed:")

with st.sidebar.form("input_form"):
    user_inputs = {}
    for feature in dl_features + ['cycle_time']:
        user_inputs[feature] = st.number_input(
            feature,
            value=float(prefilled_row.get(feature, 0.0)),
            step=0.1,
            format="%.4f"
        )
    submitted = st.form_submit_button("🚀 Run Predictions")

# ==========================================================
#  PREDICTION LOGIC
# ==========================================================
if submitted:
    try:
        # DL model input
        input_array = np.array([[user_inputs[f] for f in dl_features]], dtype=float)

        # Adjust input shape for DL model if needed
        expected_shape = dl_model.input_shape
        if len(expected_shape) == 3:
            time_steps = expected_shape[1]
            features_per_step = expected_shape[2]
            input_array = np.resize(input_array, (1, time_steps, features_per_step))

        dl_pred = float(dl_model.predict(input_array, verbose=0)[0][0])

        # ML model input
        ml_input = np.array([[user_inputs[f] for f in ml_features]], dtype=float)
        ml_pred = float(ml_model.predict(ml_input)[0])

        # Weighted final prediction
        final_pred = (dl_pred * weight_dl) + (ml_pred * weight_ml)

        # =============================
        # Display results
        # =============================
        st.subheader("📊 Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Deep Learning Prediction", round(dl_pred, 4))
        col2.metric("Machine Learning Prediction", round(ml_pred, 4))
        col3.metric("Weighted Final Prediction", round(final_pred, 4))

        st.success("✅ Prediction successful!")

        # =============================
        # Visualizations
        # =============================
        st.markdown("### 📈 Model Comparison")
        fig, ax = plt.subplots()
        models = ["Deep Learning", "Machine Learning", "Weighted Final"]
        values = [dl_pred, ml_pred, final_pred]
        ax.bar(models, values, color=["#2E86DE", "#F39C12", "#27AE60"])
        ax.set_ylabel("Prediction Value")
        st.pyplot(fig)

        st.markdown("### ⚖️ RMSE Weight Distribution")
        fig2, ax2 = plt.subplots()
        weights = [weight_dl, weight_ml]
        labels = [
            f"DL Model (RMSE={DL_RMSE:.2f})",
            f"ML Model (RMSE={ML_RMSE:.2f})"
        ]
        ax2.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90)
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
st.caption("AI For MFG | RMSE-weighted prediction system")
