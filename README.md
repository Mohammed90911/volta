# ⚡ VOLTA — Day-Ahead Electricity Price Forecasting

VOLTA is an end-to-end Deep Learning project designed to forecast hourly electricity prices in Spain 24 hours in advance, allowing users to optimize their energy consumption.

## 📌 Project Overview & Problem
Electricity prices fluctuate significantly every hour. Without clear visibility into future price spikes, consumers risk using heavy energy appliances during peak, expensive hours. VOLTA predicts prices to enable smart energy scheduling.

## 📊 Dataset
- **Source:** Kaggle (Hourly energy consumption, generation, and prices in Spain 2015–2018)[cite: 3].
- **Volume:** ~35,000 hourly records[cite: 4].
- **Key Discovery:** Prices show clear daily peaks and a ~11% price drop during weekends.

## 🧠 Model Architecture & Strategy
- **Sequence Input:** 168 hours (7-day historical window).
- **Target Output:** 24-hour forecasted prices.
- **Model:** LSTM Neural Network (Keras / TensorFlow).
- **Feature Engineering:** Added `is_weekend` to capture weekly cyclicality.

## 📈 Results & Evaluation
Evaluating on unseen test data (2018):
- **Test MAE:** `4.03 €/MWh`
- **Baseline Comparison:** **22.5% more accurate** than the Lag-24 baseline (yesterday's price).

## 🚀 Live Demo
Access the interactive Streamlit application here: [https://etr7kubn6necxskpivxubs.streamlit.app/]

## 🛠️ Tech Stack
`Python` | `TensorFlow` | `Keras` | `Pandas` | `NumPy` | `Scikit-Learn` | `Streamlit`
