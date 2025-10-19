import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Dual Model Dashboard", page_icon="🤖", layout="wide")

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/17/Google-flask-logo.png", width=120)
st.sidebar.title("⚙️ Model Parameters")
st.sidebar.write("Provide the required input values for predictions below.")

# --- LOAD MODELS ---
with open("models/dl_model.pkl", "rb") as f:
    dl_model = pickle.load(f)

with open("models/ml_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# --- INPUT FIELDS ---
inputs = {}
features = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "Ps30", "phi",
    "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31",
    "W32", "T48", "SmFan", "SmLPC", "SmHPC", "X_cycles", "σ (sigma)"
]

for ftr in features:
    inputs[ftr] = st.sidebar.number_input(ftr, value=0.0)

if st.sidebar.button("🚀 Predict"):
    # --- PREPARE INPUTS ---
    dl_features = np.array([[inputs[f] for f in features]])
    ml_subset = ["X_cycles", "T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30",
                 "phi", "NRf", "BPR", "htBleed", "W31", "W32"]
    ml_features = np.array([[inputs[f] for f in ml_subset]])

    # --- MODEL PREDICTIONS ---
    dl_pred = dl_model.predict(dl_features)[0]
    ml_pred = ml_model.predict(ml_features)[0]
    acc_dl, acc_ml = 0.89, 0.84  # Replace with your actual accuracies

    # --- WEIGHTED FINAL ---
    w1 = acc_dl / (acc_dl + acc_ml)
    w2 = acc_ml / (acc_dl + acc_ml)
    final_pred = (w1 * dl_pred) + (w2 * ml_pred)

    # --- LAYOUT ---
    st.title("🤖 Dual Model Prediction Dashboard")
    st.markdown("### Deep Learning vs Machine Learning Comparative Analysis")

    # --- METRICS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("DL Model Prediction", f"{dl_pred:.4f}")
    c2.metric("ML Model Prediction", f"{ml_pred:.4f}")
    c3.metric("Weighted Final Prediction", f"{final_pred:.4f}")

    st.markdown("---")
    st.subheader("🎯 Model Accuracy Comparison")

    # --- ACCURACY CHART ---
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Bar(
        x=["Deep Learning", "Machine Learning"],
        y=[acc_dl * 100, acc_ml * 100],
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{acc_dl*100:.1f}%", f"{acc_ml*100:.1f}%"],
        textposition="auto"
    ))
    acc_fig.update_layout(
        title="Model Accuracy (%)",
        yaxis_title="Accuracy (%)",
        xaxis_title="Model Type",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(acc_fig, use_container_width=True)

    # --- PREDICTION COMPARISON CHART ---
    st.subheader("📈 Prediction Comparison")
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=final_pred,
        delta={'reference': ml_pred, 'increasing': {'color': "green"}},
        title={'text': "Final Weighted Prediction"},
        gauge={'axis': {'range': [0, max(dl_pred, ml_pred, final_pred)*1.2]}}
    ))
    st.plotly_chart(pred_fig, use_container_width=True)

    # --- SUMMARY ---
    st.success("✅ Prediction and visual analysis complete!")
    st.markdown(f"**Weight (DL):** {w1:.2f}  **Weight (ML):** {w2:.2f}")
    st.progress(int((acc_dl + acc_ml) / 2 * 100))

else:
    st.title("🧠 Dual Model Deployment Interface")
    st.markdown("""
    Welcome to the **Dual Model Prediction & Analysis Dashboard**.

    This app allows you to:
    - Input feature parameters for both Deep Learning and Machine Learning models  
    - Compare predictions and accuracies  
    - View a final weighted prediction  

    Click **'Predict'** in the sidebar to begin.
    """)
