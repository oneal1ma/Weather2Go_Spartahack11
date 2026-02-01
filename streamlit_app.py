import streamlit as st
import requests
import joblib
import numpy as np

def load_model():
    return joblib.load("model_artifacts/rf_model.joblib")

# Get weather data from OpenWeather API
def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None
    
# Process weather data and make prediction
def predict_safety(weather_data, model, scaler):
    if not weather_data:
        return None, None

    # Extract relevant weather features
    features = {
        'temperature': weather_data['main']['temp'],
        'humidity': weather_data['main']['humidity'],
        'wind_speed': weather_data['wind']['speed'],
        'precipitation': weather_data.get('rain', {}).get('1h', 0)
    }

    # Scale features
    features_scaled = scaler.transform(pd.DataFrame([features]))

    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    return prediction, probability

def main():
    st.title("🚗 Driving Safety Analyzer")

    # Load models
    try:
        model, scaler = load_models()
    except FileNotFoundError:
        st.error("Please run model_training.py first to create the model!")
        return

    # Get API key from secrets
    api_key = st.secrets["openweather"]["api_key"]

    # Create input form
    with st.form("route_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_location = st.text_input("Starting Location")
        with col2:
            end_location = st.text_input("Destination")

        submitted = st.form_submit_button("Analyze Route")

    if submitted and start_location and end_location:
        # Get weather data for both locations
        start_weather = get_weather(start_location, api_key)
        end_weather = get_weather(end_location, api_key)

        if start_weather and end_weather:
            # Analyze both locations
            start_score, start_prob = predict_safety(start_weather, model, scaler)
            end_score, end_prob = predict_safety(end_weather, model, scaler)

            # Display results
            st.header("Route Analysis")

            # Create two columns for start and end locations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📍 {start_location}")
                st.write(f"Temperature: {start_weather['main']['temp']}°C")
                st.write(f"Humidity: {start_weather['main']['humidity']}%")
                st.write(f"Wind Speed: {start_weather['wind']['speed']} m/s")

                # Display safety level
                if start_score == 1:
                    st.success("Safe driving conditions")
                elif start_score == 2:
                    st.warning("Moderate risk")
                else:
                    st.error("High risk")

            with col2:
                st.subheader(f"🏁 {end_location}")
                st.write(f"Temperature: {end_weather['main']['temp']}°C")

