import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import json
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
import difflib
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Supply Chain Disruption Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants & Paths ---
DATA_DIR   = 'datasets/'
MODELS_DIR = 'models/'

# --- Load Models & Encoders ---
@st.cache_resource
def load_all_assets():
    xgb_model   = joblib.load(MODELS_DIR + 'xgboost_model.pkl')
    lstm_model  = load_model(MODELS_DIR + 'lstm_model.h5')
    iso_forest  = joblib.load(MODELS_DIR + 'isolation_forest.pkl')
    scaler      = joblib.load(MODELS_DIR + 'scaler.pkl')
    le_transport = joblib.load(MODELS_DIR + 'le_transport.pkl')
    le_product   = joblib.load(MODELS_DIR + 'le_product.pkl')
    le_city      = joblib.load(MODELS_DIR + 'le_city.pkl')
    le_dest      = joblib.load(MODELS_DIR + 'le_dest.pkl')
    
    with open(MODELS_DIR + 'feature_list.json') as f:
        XGB_FEATURES = json.load(f)
        
    return {
        'xgb': xgb_model, 'lstm': lstm_model, 'iso': iso_forest,
        'scaler': scaler, 'le_transport': le_transport, 'le_product': le_product,
        'le_city': le_city, 'le_dest': le_dest, 'xgb_features': XGB_FEATURES
    }

# --- Load Data & Lookups ---
@st.cache_data
def load_lookups():
    d3 = pd.read_csv(DATA_DIR + 'dataset3_news_sentiment_india.csv', parse_dates=['date'])
    d4 = pd.read_csv(DATA_DIR + 'dataset4_supplier_risk_india.csv')
    merged = pd.read_csv(DATA_DIR + 'merged_dataset.csv')
    
    d3['month'] = d3['date'].dt.month
    sentiment_lookup = d3.groupby(['affected_city','month']).agg(
        avg_sentiment=('sentiment_score','mean'),
        avg_disruption_signal=('disruption_signal','mean')
    ).reset_index()
    
    city_hist_avg = merged.groupby('supplier_city')['historical_disruption_count'].mean().to_dict()
    
    city_rolling = merged.groupby('supplier_city').agg(
        avg_rolling7 =('rolling_7d_delay','mean'),
        avg_rolling14=('rolling_14d_delay','mean'),
        avg_lag1     =('lag1_delay','mean'),
        avg_lag2     =('lag2_delay','mean')
    ).to_dict('index')
    
    return sentiment_lookup, d4, merged, city_hist_avg, city_rolling

# --- Helper Functions ---
CITY_COORDS = {
    'Mumbai':    (19.08, 72.88), 'Delhi':     (28.61, 77.21),
    'Chennai':   (13.08, 80.27), 'Kolkata':   (22.57, 88.36),
    'Bangalore': (12.97, 77.59), 'Hyderabad': (17.38, 78.49),
    'Pune':      (18.52, 73.86), 'Ahmedabad': (23.03, 72.58),
    'Jaipur':    (26.91, 75.79), 'Lucknow':   (26.85, 80.95),
    'Surat':     (21.17, 72.83), 'Kanpur':    (26.46, 80.33),
    'Nagpur':    (21.15, 79.08), 'Indore':    (22.72, 75.86),
    'Bhopal':    (23.26, 77.41),
}

LSTM_FEATURES = [
    'distance_km','promised_delivery_days','quantity_units',
    'rainfall_mm','temperature_celsius','wind_speed_kmh',
    'severity_score','weather_anomaly_score',
    'sentiment_score','disruption_signal',
    'composite_risk_score','delivery_reliability_score',
    'regional_risk_index','historical_disruption_count',
    'rolling_7d_delay','rolling_14d_delay','lag1_delay','lag2_delay',
    'transport_mode_enc','product_category_enc','supplier_city_enc'
]
SEQ_LEN = 14

def fetch_forecast(city):
    try:
        lat, lon = CITY_COORDS[city]
        resp = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lat, 'longitude': lon,
            'daily': 'precipitation_sum,temperature_2m_max,wind_speed_10m_max',
            'forecast_days': 7, 'timezone': 'Asia/Kolkata'
        }, timeout=10)
        resp.raise_for_status()
        wf   = pd.DataFrame(resp.json()['daily'])
        rain = float(wf['precipitation_sum'].fillna(0).mean())
        temp = float(wf['temperature_2m_max'].fillna(28).mean())
        wind = float(wf['wind_speed_10m_max'].fillna(10).mean())
        sev  = 2 if (rain > 50 or wind > 60) else (1 if (rain > 15 or wind > 35) else 0)
        return rain, temp, wind, sev
    except:
        return 0.0, 28.0, 10.0, 0

def get_sentiment(city, month, sentiment_lookup):
    row = sentiment_lookup[
        (sentiment_lookup['affected_city'] == city) &
        (sentiment_lookup['month'] == month)
    ]
    if len(row) == 0:
        return 0.0, 0.0
    return float(row['avg_sentiment'].values[0]), float(row['avg_disruption_signal'].values[0])

def get_supplier_risk(supplier_name, d4):
    row = d4[d4['supplier_name'] == supplier_name]
    if len(row) == 0:
        return {c: float(d4[c].median()) for c in
                ['composite_risk_score','delivery_reliability_score',
                 'regional_risk_index','strike_incidents_3yr',
                 'financial_stability_score']} | {'risk_category': 'Medium'}
    return row[['composite_risk_score','delivery_reliability_score',
                'regional_risk_index','strike_incidents_3yr',
                'financial_stability_score','risk_category']].iloc[0].to_dict()

def get_top_causes(origin_city, transport_mode, weather_sev, sentiment, supplier_risk_cat):
    causes = []
    if weather_sev == 2:
        causes.append(f'Severe weather alert in {origin_city}')
    elif weather_sev == 1:
        causes.append(f'Moderate rainfall/wind in {origin_city}')
    if sentiment < -0.4:
        causes.append(f'Negative news sentiment in {origin_city}')
    if supplier_risk_cat in ('High', 'Medium'):
        causes.append(f'Supplier rated {supplier_risk_cat} risk')
    if transport_mode == 'Road':
        causes.append('Road transport vulnerability')
    if len(causes) == 0:
        causes.append('Combined low-level risk factors')
    return causes[:3]

# --- Main App ---

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Disruption", "About the System"])

    assets = load_all_assets()
    sentiment_lookup, d4, merged, city_hist_avg, city_rolling = load_lookups()

    if page == "Predict Disruption":
        render_predictor(assets, sentiment_lookup, d4, merged, city_hist_avg, city_rolling)
    else:
        render_about()

def render_predictor(assets, sentiment_lookup, d4, merged, city_hist_avg, city_rolling):
    st.title("🚚 Supply Chain Disruption Predictor")
    st.markdown("Enter shipment details to predict the risk of disruption and estimated delivery delay.")

    col1, col2 = st.columns(2)

    with col1:
        origin_city = st.selectbox("Origin City", list(CITY_COORDS.keys()))
        destination_city = st.selectbox("Destination City", list(CITY_COORDS.keys()), index=1)
        product_category = st.selectbox("Product Category", list(assets['le_product'].classes_))
        transport_mode = st.selectbox("Transport Mode", list(assets['le_transport'].classes_))

    with col2:
        quantity_units = st.number_input("Quantity (Units)", min_value=1, value=500)
        order_date = st.date_input("Order Date", datetime.now())
        supplier_name = st.selectbox("Supplier Name", list(d4['supplier_name'].unique()))

    if st.button("Predict Disruption Risk", type="primary"):
        with st.spinner("Analyzing risk factors and running models..."):
            # Logic from predict_shipment
            order_dt = pd.to_datetime(order_date)
            shipment_month = order_dt.month

            rain, temp, wind, sev = fetch_forecast(origin_city)
            
            weather_vec = np.array([[rain, temp, wind, sev]])
            iso_pred = assets['iso'].predict(weather_vec)
            weather_anomaly = int(iso_pred[0] == -1)

            sentiment, disruption_signal = get_sentiment(origin_city, shipment_month, sentiment_lookup)
            risk = get_supplier_risk(supplier_name, d4)
            hist_count = city_hist_avg.get(origin_city, 0.0)
            cr = city_rolling.get(origin_city, {'avg_rolling7':0,'avg_rolling14':0,'avg_lag1':0,'avg_lag2':0})

            transport_enc = int(assets['le_transport'].transform([transport_mode])[0])
            product_enc = int(assets['le_product'].transform([product_category])[0])
            city_enc = int(assets['le_city'].transform([origin_city])[0])
            dest_enc = int(assets['le_dest'].transform([destination_city])[0])

            pair = merged[
                (merged['supplier_city'] == origin_city) &
                (merged['destination_city'] == destination_city)
            ]['distance_km']
            distance_km = float(pair.mean()) if len(pair) > 0 else float(merged['distance_km'].mean())

            promised_days_est = float(merged[
                (merged['transport_mode'] == transport_mode)
            ]['promised_delivery_days'].mean())

            feat = {
                'distance_km': distance_km,
                'promised_delivery_days': promised_days_est,
                'quantity_units': float(quantity_units),
                'rainfall_mm': rain,
                'temperature_celsius': temp,
                'wind_speed_kmh': wind,
                'severity_score': float(sev),
                'weather_anomaly_score': float(weather_anomaly),
                'sentiment_score': sentiment,
                'disruption_signal': disruption_signal,
                'composite_risk_score': float(risk['composite_risk_score']),
                'delivery_reliability_score': float(risk['delivery_reliability_score']),
                'regional_risk_index': float(risk['regional_risk_index']),
                'strike_incidents_3yr': float(risk['strike_incidents_3yr']),
                'financial_stability_score': float(risk['financial_stability_score']),
                'historical_disruption_count': hist_count,
                'rolling_7d_delay': cr['avg_rolling7'],
                'rolling_14d_delay': cr['avg_rolling14'],
                'lag1_delay': cr['avg_lag1'],
                'lag2_delay': cr['avg_lag2'],
                'transport_mode_enc': float(transport_enc),
                'product_category_enc': float(product_enc),
                'supplier_city_enc': float(city_enc),
                'destination_city_enc': float(dest_enc),
            }

            # LSTM
            lstm_vec = np.array([feat[f] for f in LSTM_FEATURES], dtype=np.float32)
            lstm_vec_scaled = assets['scaler'].transform(lstm_vec.reshape(1, -1))
            lstm_seq = np.tile(lstm_vec_scaled, (SEQ_LEN, 1)).reshape(1, SEQ_LEN, len(LSTM_FEATURES))
            lstm_prob = float(assets['lstm'].predict(lstm_seq, verbose=0)[0][0])

            # XGBoost
            feat['lstm_disruption_prob'] = lstm_prob
            xgb_df = pd.DataFrame([[feat[f] for f in assets['xgb_features']]], columns=assets['xgb_features'])
            xgb_prob = float(assets['xgb'].predict_proba(xgb_df)[0][1])

            final_pct = round(xgb_prob * 100, 1)
            
            # Output Layout
            st.divider()
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if final_pct >= 70:
                    st.error(f"### High Risk: {final_pct}%")
                elif final_pct >= 40:
                    st.warning(f"### Medium Risk: {final_pct}%")
                else:
                    st.success(f"### Low Risk: {final_pct}%")
                st.write(f"**Confidence:** {round((abs(xgb_prob - 0.5) / 0.5) * 100, 1)}%")

            with res_col2:
                base_travel_days = int(round(distance_km / 400))
                if transport_mode == 'Air': base_travel_days = max(1, int(round(distance_km / 800)))
                elif transport_mode == 'Rail': base_travel_days = int(round(distance_km / 500))
                
                avg_delay = float(merged[merged['supplier_city'] == origin_city]['delay_days'].mean())
                predicted_delay = int(round(avg_delay * xgb_prob * 1.5))
                expected_actual = order_dt + pd.Timedelta(days=base_travel_days + predicted_delay)
                
                st.metric("Predicted Delay", f"{predicted_delay} Days")
                st.write(f"**Exp. Delivery:** {expected_actual.strftime('%Y-%m-%d')}")

            with res_col3:
                st.write("**Top Risk Factors:**")
                causes = get_top_causes(origin_city, transport_mode, sev, sentiment, risk['risk_category'])
                for c in causes:
                    st.write(f"- {c}")

            st.divider()
            
            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.write("### Environmental Factors")
                st.write(f"- **Weather Forecast:** {round(rain,1)}mm rain, {round(wind,1)}km/h wind")
                st.write(f"- **Anomaly Detected:** {'Yes' if weather_anomaly else 'No'}")
                st.write(f"- **News Sentiment:** {round(sentiment, 2)} (Signal: {round(disruption_signal, 2)})")
            
            with det_col2:
                st.write("### Supplier & Logistics")
                st.write(f"- **Supplier Risk:** {risk['risk_category']} (Score: {round(risk['composite_risk_score'], 2)})")
                st.write(f"- **Historical Disruptions:** {round(hist_count, 1)} in {origin_city}")
                st.write(f"- **LSTM Prob:** {round(lstm_prob*100, 1)}% | **XGB Prob:** {round(xgb_prob*100, 1)}%")

def render_about():
    st.title("📖 About the System")
    
    st.markdown("""
    ### 🎯 End Goal
    The primary objective of this system is to provide **proactive visibility** into supply chain disruptions within the Indian logistics network. By integrating real-time environmental data, news sentiment, and historical patterns, it enables businesses to:
    1. **Anticipate Delays:** Move from reactive to predictive logistics management.
    2. **Mitigate Risk:** Identify high-risk shipments before they leave the warehouse.
    3. **Optimize Planning:** Adjust buffer stocks and delivery promises based on data-driven insights.

    ---

    ### 🤖 The Multi-Model Architecture
    This system employs a "Stacked Hybrid" approach, using three distinct types of Machine Learning models to capture different facets of risk.

    #### 1. Isolation Forest (Anomaly Detection)
    *   **Role:** Identifies unusual weather patterns.
    *   **How it works:** Unlike standard models that learn "normal" behavior, Isolation Forest explicitly isolates anomalies. In this system, it looks at rainfall, wind speed, and temperature to flag weather conditions that are statistically 'weird' compared to the training data.
    *   **Impact:** Even if total rainfall isn't 'severe', an anomalous combination of factors can trigger a risk signal.

    #### 2. LSTM - Long Short-Term Memory (Deep Learning)
    *   **Role:** Captures temporal dependencies and complex non-linear relationships.
    *   **How it works:** LSTM is a type of Recurrent Neural Network (RNN) designed to remember patterns over time. Here, it processes a sequence of 21 features (weather, sentiment, supplier risk, rolling delays).
    *   **Impact:** It excels at understanding how a 'build-up' of minor issues (like three days of negative news followed by rising supplier risk) contributes to an eventual disruption.

    #### 3. XGBoost (Gradient Boosted Trees)
    *   **Role:** The "Final Arbiter" or Meta-Classifier.
    *   **How it works:** XGBoost is a powerful ensemble method that builds decision trees sequentially, each one correcting the errors of the previous one. It takes all the raw inputs **plus** the probability output from the LSTM model.
    *   **Impact:** It provides the final probability score by weighing the deep learning insights against categorical factors (like Transport Mode or Product Category) that are highly predictive of disruption.

    ---

    ### 📊 Data Sources
    - **Historical Supply Chain Data:** 2,000+ shipment records across 15 Indian cities.
    - **Live Weather Forecast:** Real-time data from Open-Meteo API.
    - **News & Sentiment:** Analyzed disruption signals from regional news.
    - **Supplier Risk Profiles:** Comprehensive scoring of 60+ major Indian suppliers.
    """)

if __name__ == "__main__":
    main()
