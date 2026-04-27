# 🚚 Supply Chain Disruption Prediction System (India)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://supply-chain-disruption.streamlit.app/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, multi-model AI system designed to predict and mitigate supply chain disruptions across the Indian logistics network.

**🔗 Live App:** [supply-chain-disruption.streamlit.app](https://supply-chain-disruption.streamlit.app/)
 By integrating real-time weather data, regional news sentiment, and historical performance metrics, the system provides proactive visibility into potential delays.

---

## 🌟 Key Features

*   **Hybrid AI Architecture:** Combines Anomaly Detection (Isolation Forest), Deep Learning (LSTM), and Gradient Boosting (XGBoost).
*   **Real-time Intelligence:** Integrates live 7-day weather forecasts via Open-Meteo API.
*   **Sentiment Analysis:** Factors in regional disruption signals from news data.
*   **Interactive Dashboard:** A sleek Streamlit interface for predicting shipment risks and delivery delays.
*   **Explainable AI:** Provides top "Risk Causes" for every prediction.

---

## 🤖 The Multi-Model Approach

This system utilizes a **Stacked Hybrid Architecture** to capture different facets of risk:

1.  **Isolation Forest (Anomaly Detection):** Monitors environmental inputs (Rainfall, Wind, Temp) to flag statistically unusual weather patterns that might bypass standard severity thresholds.
2.  **LSTM (Long Short-Term Memory):** A Deep Learning model that analyzes sequences of 21 features to understand temporal dependencies and how a "build-up" of minor issues leads to major disruptions.
3.  **XGBoost (Meta-Classifier):** The final decision-maker that weighs the LSTM insights against categorical data (Transport Mode, Product Type) to deliver the final probability score.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.12 (Recommended)
*   Git

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sourabhsp23/supply_chain_disruption_lstm.git
    cd supply_chain_disruption_lstm
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```text
├── datasets/             # Historical shipment, news, and supplier data
├── models/               # Saved .pkl and .h5 model files & encoders
├── notebooks/            # Step-by-step EDA and Model Training
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_LSTM_Model.ipynb
│   ├── 04_XGBoost_Model.ipynb
│   └── 05_Future_Prediction.ipynb
├── app.py                # Streamlit Interface
├── requirements.txt      # Project dependencies
└── runtime.txt           # Python version for cloud deployment
```

---

## 📈 System Objectives
*   **Proactive Visibility:** Move from reactive to predictive logistics.
*   **Risk Mitigation:** Identify high-risk shipments before they leave the origin.
*   **Optimized Planning:** Adjust buffer stocks and delivery promises based on data-driven confidence scores.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.
