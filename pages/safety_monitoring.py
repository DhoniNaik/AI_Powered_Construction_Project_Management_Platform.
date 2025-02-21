import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.data_generator import generate_sample_data

st.title("🚨 Safety Monitoring & Hazard Detection")

if 'hazard_detections' not in st.session_state:
    st.session_state.hazard_detections = []
if 'safety_alerts' not in st.session_state:
    st.session_state.safety_alerts = []

def detect_hazards(image):
    """Simulate hazard detection using computer vision"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hazards = []

    if np.mean(gray) < 100:
        hazards.append({
            'type': 'Poor Lighting',
            'confidence': 0.85,
            'location': 'Area requires additional lighting'
        })

    edges = cv2.Canny(gray, 100, 200)
    if np.mean(edges) > 50:
        hazards.append({
            'type': 'Trip Hazard',
            'confidence': 0.75,
            'location': 'Uneven surface detected'
        })

    return hazards

class SafetyPredictor:
    def predict_risk_areas(self, historical_data):
        """Predict high-risk areas based on historical incident data"""
        risk_areas = []

        if len(historical_data) > 0:
            incident_count = pd.Series(
                [incident['type'] for incident in historical_data]
            ).value_counts()

            for incident_type, count in incident_count.items():
                if count >= 2:  # Threshold for high-risk classification
                    risk_areas.append({
                        'area_type': incident_type,
                        'risk_level': min(0.9, count * 0.2),  # Scale risk with frequency
                        'suggestion': f"Increase monitoring for {incident_type.lower()} incidents"
                    })

        return risk_areas

st.header("👁️ Real-Time Hazard Detection")

uploaded_file = st.file_uploader(
    "Upload site image for hazard detection",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(uploaded_file, caption="Site Image", use_column_width=True)
    hazards = detect_hazards(image)

    if hazards:
        st.subheader("⚠️ Detected Hazards")
        for hazard in hazards:
            st.warning(
                f"**{hazard['type']}** (Confidence: {hazard['confidence']:.2%})\n\n"
                f"Location: {hazard['location']}"
            )

            # Store detection in session state
            detection = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': hazard['type'],
                'confidence': hazard['confidence'],
                'location': hazard['location']
            }
            st.session_state.hazard_detections.append(detection)
    else:
        st.success("No immediate hazards detected")

st.header("🔮 Predictive Safety Analytics")

safety_predictor = SafetyPredictor()

risk_areas = safety_predictor.predict_risk_areas(st.session_state.hazard_detections)

if risk_areas:
    st.subheader("Predicted High-Risk Areas")
    for area in risk_areas:
        st.error(
            f"**{area['area_type']}**\n\n"
            f"Risk Level: {area['risk_level']:.2%}\n\n"
            f"Suggestion: {area['suggestion']}"
        )

        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': area['area_type'],
            'risk_level': area['risk_level'],
            'recommendation': area['suggestion']
        }
        st.session_state.safety_alerts.append(alert)

st.header("📊 Safety Analytics")

if st.session_state.hazard_detections:
    detections_df = pd.DataFrame(st.session_state.hazard_detections)

    fig_hazards = px.pie(
        detections_df,
        names='type',
        title='Hazard Type Distribution'
    )
    st.plotly_chart(fig_hazards)

    detections_df['timestamp'] = pd.to_datetime(detections_df['timestamp'])
    fig_timeline = px.line(
        detections_df,
        x='timestamp',
        y='confidence',
        color='type',
        title='Hazard Detection Timeline'
    )
    st.plotly_chart(fig_timeline)

st.header("🚨 Recent Safety Alerts")
if st.session_state.safety_alerts:
    alerts_df = pd.DataFrame(st.session_state.safety_alerts)
    alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    alerts_df = alerts_df.sort_values('timestamp', ascending=False)

    for _, alert in alerts_df.head(5).iterrows():
        st.error(
            f"**{alert['type']}** - {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Risk Level: {alert['risk_level']:.2%}\n\n"
            f"Recommendation: {alert['recommendation']}"
        )
else:
    st.info("No active safety alerts")

st.markdown("---")
st.markdown("Safety monitoring system is actively analyzing site conditions and predicting potential hazards.")