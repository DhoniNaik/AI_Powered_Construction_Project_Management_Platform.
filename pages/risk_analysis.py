import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import generate_sample_data
from utils.ml_models import RiskPredictor

st.title("🎯 Risk Analysis Dashboard")

data = generate_sample_data()
risk_data = data['risks']

risk_predictor = RiskPredictor()

if 'weather_condition' not in risk_data.columns:
    risk_data['weather_condition'] = 'Clear'  # Default value

st.subheader("Risk Overview")
col1, col2 = st.columns(2)

with col1:
    fig_trend = px.line(
        risk_data, 
        x='date', 
        y='risk_score',
        title='Risk Score Trend'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    fig_dist = px.box(
        risk_data, 
        x='category', 
        y='risk_score',
        title='Risk Distribution by Category'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Risk Predictions")

risk_data['weather_severity_score'] = risk_data['weather_condition'].map({
    'Clear': 0.2,
    'Moderate Rain': 0.5,
    'Heavy Rain': 0.8,
    'Storm Warning': 0.9,
    'Cyclone Alert': 1.0,
    'Flood Warning': 0.9
}).fillna(0.2)  

if 'site_condition_score' not in risk_data.columns:
    risk_data['site_condition_score'] = 0.5  # Default moderate risk

try:
    risk_predictor.train(risk_data)
    predictions = risk_predictor.predict(risk_data)
    risk_data['predicted_risk'] = predictions

    fig_predictions = go.Figure()
    fig_predictions.add_trace(go.Scatter(
        x=risk_data['date'],
        y=risk_data['risk_score'],
        name='Actual Risk',
        line=dict(color='blue')
    ))
    fig_predictions.add_trace(go.Scatter(
        x=risk_data['date'],
        y=risk_data['predicted_risk'],
        name='Predicted Risk',
        line=dict(color='red', dash='dash')
    ))
    fig_predictions.update_layout(
        title='Risk Prediction Model Performance',
        xaxis_title='Date',
        yaxis_title='Risk Score'
    )
    st.plotly_chart(fig_predictions, use_container_width=True)

    st.subheader("Risk Factors Analysis")

    risk_factors = risk_data.groupby('category')['risk_score'].mean().reset_index()
    fig_factors = px.bar(
        risk_factors,
        x='category',
        y='risk_score',
        title='Average Risk Score by Category'
    )
    st.plotly_chart(fig_factors, use_container_width=True)

    st.subheader("Weather Impact Analysis")
    weather_impact = risk_data.groupby('weather_condition')['risk_score'].mean().reset_index()
    fig_weather = px.bar(
        weather_impact,
        x='weather_condition',
        y='risk_score',
        title='Average Risk Score by Weather Condition'
    )
    st.plotly_chart(fig_weather, use_container_width=True)

    st.subheader("Risk Summary")
    col3, col4, col5 = st.columns(3)

    with col3:
        avg_risk = risk_data['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.2f}")

    with col4:
        high_risk_count = len(risk_data[risk_data['risk_score'] > 70])
        st.metric("High Risk Incidents", high_risk_count)

    with col5:
        prediction_accuracy = 1 - abs(risk_data['risk_score'] - risk_data['predicted_risk']).mean() / 100
        st.metric("Model Accuracy", f"{prediction_accuracy:.1%}")

except Exception as e:
    st.error(f"Error in risk prediction: {str(e)}")
    st.info("Please ensure your data includes required columns: date, risk_score, category, and weather_condition")

st.markdown("---")
st.markdown("Risk analysis is updated in real-time based on project data and environmental conditions.")