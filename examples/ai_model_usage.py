import streamlit as st
from utils.ai_models import RiskAssessmentModel, TimelinePredictor, ResourceOptimizer, SafetyHazardDetector
from utils.ai_helpers import prepare_project_data, analyze_project, analyze_safety_image
import cv2
import numpy as np

def demo_risk_assessment():
    """Demonstrate risk assessment model usage"""
    project_data = {
        'weather_severity': 0.3,        
        'resource_availability': 0.9,    
        'historical_performance': 0.85,  
    }
    
    risk_model = RiskAssessmentModel()
    risk_assessment = risk_model.predict_risk(project_data)
    
    return risk_assessment

def demo_timeline_prediction():
    """Demonstrate timeline prediction model usage"""
    project_data = {
        'planned_duration': 90,        
        'complexity_score': 0.6,      
        'resource_availability': 0.8,   
    }
    
    timeline_model = TimelinePredictor()
    timeline_prediction = timeline_model.predict_timeline(project_data)
    
    return timeline_prediction

def demo_resource_optimization():
    """Demonstrate resource optimization model usage"""
    project_data = {
        'scope_size': 100,            
        'complexity_score': 0.5,      
        'weather_severity_score': 0.2 
    }
    
    resource_model = ResourceOptimizer()
    resource_prediction = resource_model.predict_resource_needs(project_data)
    
    return resource_prediction

def demo_safety_detection(image):
    """Demonstrate safety hazard detection model usage"""
    safety_model = SafetyHazardDetector()
    
    hazards = safety_model.analyze_image(image)
    
    site_assessment = safety_model.assess_site_conditions(hazards)
    
    return hazards, site_assessment

st.title("Construction AI Models Demo")

st.header("1. Risk Assessment Model")
if st.button("Run Risk Assessment Demo"):
    risk_results = demo_risk_assessment()
    st.json(risk_results)

st.header("2. Timeline Prediction Model")
if st.button("Run Timeline Prediction Demo"):
    timeline_results = demo_timeline_prediction()
    st.json(timeline_results)

st.header("3. Resource Optimization Model")
if st.button("Run Resource Optimization Demo"):
    resource_results = demo_resource_optimization()
    st.json(resource_results)

st.header("4. Safety Hazard Detection Model")
uploaded_file = st.file_uploader("Upload a construction site image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Safety Hazards"):
        hazards, assessment = demo_safety_detection(image)
        
        st.subheader("Detected Hazards")
        st.json(hazards)
        
        st.subheader("Site Safety Assessment")
        st.json(assessment)
