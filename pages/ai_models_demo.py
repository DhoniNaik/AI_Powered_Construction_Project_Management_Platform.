import streamlit as st
from utils.ai_models import RiskAssessmentModel, TimelinePredictor, ResourceOptimizer, SafetyHazardDetector
from utils.ai_helpers import prepare_project_data, analyze_project, analyze_safety_image
import cv2
import numpy as np

st.title("🤖 AI Models Demo")
st.markdown("""
This page demonstrates the AI capabilities of our construction project management platform:
- Risk Assessment Model
- Timeline Prediction Model
- Resource Optimization Model
- Safety Hazard Detection Model
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Assessment",
    "Timeline Prediction",
    "Resource Optimization",
    "Safety Detection"
])

with tab1:
    st.header("📊 Risk Assessment Model")
    st.markdown("""
    This model evaluates project risks based on multiple factors including:
    - Weather conditions
    - Resource availability
    - Historical performance
    """)

    # Input controls
    weather_risk = st.slider("Weather Severity (0-1)", 0.0, 1.0, 0.3, 0.1)
    resource_avail = st.slider("Resource Availability (0-1)", 0.0, 1.0, 0.9, 0.1)
    hist_performance = st.slider("Historical Performance (0-1)", 0.0, 1.0, 0.85, 0.1)

    if st.button("Analyze Risk"):
        try:
            project_data = {
                'weather_severity_score': weather_risk,
                'resource_availability_score': resource_avail,
                'historical_performance_score': hist_performance
            }

            risk_model = RiskAssessmentModel()
            risk_assessment = risk_model.predict_risk(project_data)

            # Display results in a formatted way
            st.subheader("Risk Assessment Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Risk Level", risk_assessment['risk_level'])
                st.metric("Confidence", f"{risk_assessment['confidence']:.2%}")

            with col2:
                st.write("Risk Factors:")
                for factor, value in risk_assessment['risk_factors'].items():
                    st.progress(value, text=f"{factor.title()}: {value:.2f}")

        except Exception as e:
            st.error(f"Error in risk assessment: {str(e)}")
            st.info("Please try adjusting the input values and try again.")

with tab2:
    st.header("⏱️ Timeline Prediction Model")
    st.markdown("""
    This model predicts project timelines and potential delays based on:
    - Planned duration
    - Project complexity
    - Resource availability
    """)

    # Input controls
    planned_days = st.number_input("Planned Duration (days)", 30, 365, 90)
    complexity = st.slider("Project Complexity (0-1)", 0.0, 1.0, 0.6, 0.1)
    resource_score = st.slider("Resource Availability Score (0-1)", 0.0, 1.0, 0.8, 0.1)

    if st.button("Predict Timeline"):
        try:
            project_data = {
                'planned_duration': planned_days,
                'complexity_score': complexity,
                'resource_availability_score': resource_score
            }

            timeline_model = TimelinePredictor()
            timeline_prediction = timeline_model.predict_timeline(project_data)

            # Display results in a formatted way
            st.subheader("Timeline Prediction Results")

            # Metrics overview
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Duration",
                    f"{timeline_prediction['predicted_duration']} days",
                    f"{timeline_prediction['delay_days']:+.1f} days"
                )

            with col2:
                delay_risk_color = (
                    "red" if timeline_prediction['delay_risk'] > 0.2
                    else "orange" if timeline_prediction['delay_risk'] > 0.1
                    else "green"
                )
                st.metric(
                    "Delay Risk",
                    f"{timeline_prediction['delay_risk']:.1%}",
                    delta_color=delay_risk_color
                )

            with col3:
                st.metric(
                    "Confidence Score",
                    f"{timeline_prediction['confidence_score']:.1%}"
                )

            # Impact factors
            st.subheader("Impact Factors")
            for factor, value in timeline_prediction['factors'].items():
                st.progress(
                    value,
                    text=f"{factor.replace('_', ' ').title()}: {value:.2f}"
                )

            # Recommendations
            st.subheader("Recommendations")
            if timeline_prediction['delay_risk'] > 0.2:
                st.warning("""
                    ⚠️ High delay risk detected. Consider:
                    - Increasing resource allocation
                    - Breaking down complex tasks
                    - Adding buffer time to the schedule
                """)
            elif timeline_prediction['delay_risk'] > 0.1:
                st.info("""
                    ℹ️ Moderate delay risk. Monitor:
                    - Resource utilization
                    - Task dependencies
                    - Critical path activities
                """)
            else:
                st.success("""
                    ✅ Low delay risk. Continue:
                    - Regular progress tracking
                    - Resource optimization
                    - Risk monitoring
                """)

        except Exception as e:
            st.error(f"Error in timeline prediction: {str(e)}")
            st.info("Please try adjusting the input values and try again.")

with tab3:
    st.header("🔄 Resource Optimization Model")
    st.markdown("""
    This model optimizes resource allocation based on:
    - Project scope
    - Complexity
    - Weather impact
    """)

    # Input controls
    scope_size = st.number_input("Project Scope (units)", 10, 1000, 100)
    complexity_score = st.slider("Project Complexity Score (0-1)", 0.0, 1.0, 0.5, 0.1)
    weather_impact = st.slider("Weather Impact Score (0-1)", 0.0, 1.0, 0.2, 0.1)

    if st.button("Optimize Resources"):
        try:
            project_data = {
                'scope_size': scope_size,
                'complexity_score': complexity_score,
                'weather_severity': weather_impact
            }

            resource_model = ResourceOptimizer()
            resource_prediction = resource_model.predict_resource_needs(prepare_project_data(project_data))
            st.json(resource_prediction)
        except Exception as e:
            st.error(f"Error in resource optimization: {str(e)}")

with tab4:
    st.header("👁️ Safety Hazard Detection Model")
    st.markdown("""
    This model uses computer vision to detect safety hazards in construction site images:
    - Fall hazards
    - Equipment hazards
    - Lighting conditions
    - PPE violations
    """)

    uploaded_file = st.file_uploader("Upload a construction site image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        try:
            # Convert uploaded file to opencv format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if st.button("Analyze Safety Hazards"):
                # Analyze image for safety hazards
                analysis_results = analyze_safety_image(image)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Detected Hazards")
                    st.json(analysis_results['hazards'])

                with col2:
                    st.subheader("Site Safety Assessment")
                    st.json(analysis_results['site_assessment'])
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure the uploaded image is in a valid format (JPG, JPEG, or PNG).")

# Footer
st.markdown("---")
st.markdown("These AI models are powered by machine learning algorithms trained on construction industry data.")