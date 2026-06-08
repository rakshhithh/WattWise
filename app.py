import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Electricity Forecast", layout="wide")

st.title("⚡ Electricity Demand Forecasting (48 Hours)")
st.write("Forecast electricity demand using Prophet, LSTM, or an ensemble model.")

lstm = load_model("lstm_model.keras")

with open("prophet_model.pkl", "rb") as f:
    prophet = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

last_window = np.load("last_window.npy")

model_choice = st.selectbox(
    "Select Forecasting Model",
    ["Prophet", "LSTM", "Ensemble"]
)

if st.button("🔮 Generate 48-Hour Forecast"):

    timestamps = pd.date_range(
        start=pd.Timestamp.now(),
        periods=48,
        freq="H"
    )

    if model_choice == "Prophet":
        future = prophet.make_future_dataframe(periods=48, freq="H")
        forecast = prophet.predict(future)
        preds = forecast["yhat"].tail(48).values

        title = "Prophet Forecast"

    elif model_choice == "LSTM":
        window = last_window.copy()
        preds_scaled = []

        for _ in range(48):
            pred = lstm.predict(window.reshape(1, *window.shape), verbose=0)
            preds_scaled.append(pred[0, 0])

            window = np.roll(window, -1, axis=0)
            window[-1, 0] = pred[0, 0]
        
        n_features = last_window.shape[1]

        dummy = np.zeros((48, n_features))
        dummy[:, 0] = np.array(preds_scaled)

        preds = scaler.inverse_transform(dummy)[:, 0]

        title = "LSTM Forecast"

    else:
        future = prophet.make_future_dataframe(periods=48, freq="H")
        forecast = prophet.predict(future)
        prophet_48 = forecast["yhat"].tail(48).values

        window = last_window.copy()
        lstm_scaled = []

        for _ in range(48):
            pred = lstm.predict(window.reshape(1, *window.shape), verbose=0)
            lstm_scaled.append(pred[0, 0])
            window = np.roll(window, -1, axis=0)
            window[-1, 0] = pred[0, 0]

        dummy = np.zeros((48, last_window.shape[1]))
        dummy[:, 0] = np.array(lstm_scaled)

        lstm_48 = scaler.inverse_transform(dummy)[:, 0]

        preds = 0.5 * prophet_48 + 0.5 * lstm_48
        title = "Ensemble Forecast"

    st.subheader(f"📈 {title}")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(timestamps, preds, label=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Electricity Load")
    ax.legend()
    st.pyplot(fig)

    df_out = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_load": preds
    })

    st.download_button(
        "⬇️ Download Forecast CSV",
        df_out.to_csv(index=False),
        file_name="forecast.csv"
    )