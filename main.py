import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_generator import generate_sample_data
from utils.ml_models import RiskPredictor, ResourceOptimizer
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Construction Project Management Platform",
    page_icon="🏗️",
    layout="wide"
)

if 'projects' not in st.session_state:
    st.session_state.projects = []

st.title("🏗️ AI-Powered Construction Project Management Platform")
st.markdown("""
This platform provides comprehensive project management capabilities with AI-powered insights:
- 📊 Risk Analysis and Prediction
- 👥 Resource Management
- 📈 Project Monitoring
- 🔄 Supply Chain Analytics
""")

st.header("Project Management")

with st.expander("➕ Add New Project", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        new_project_name = st.text_input("Project Name")
        start_date = st.date_input("Start Date", min_value=datetime.now())
        project_type = st.selectbox("Project Type", [
            "Infrastructure",
            "Commercial",
            "Residential",
            "Industrial",
            "Transportation"
        ])

    with col2:
        project_duration = st.number_input("Duration (days)", min_value=1, value=90)
        budget = st.number_input("Budget ($)", min_value=0, value=1000000)
        priority = st.selectbox("Priority", ["High", "Medium", "Low"])

    if st.button("Add Project"):
        if new_project_name:
            new_project = {
                'project_id': f"P{len(st.session_state.projects) + 1:03d}",
                'name': new_project_name,
                'type': project_type,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': (start_date + timedelta(days=project_duration)).strftime('%Y-%m-%d'),
                'duration': project_duration,
                'budget': budget,
                'priority': priority,
                'status': "On Track",
                'completion_percentage': 0,
                'risk_level': "Medium"
            }
            st.session_state.projects.append(new_project)
            st.success(f"Project '{new_project_name}' added successfully!")
        else:
            st.error("Please enter a project name")

if st.session_state.projects:
    st.header("Active Projects")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Projects", len(st.session_state.projects))
    with cols[1]:
        on_track = sum(1 for p in st.session_state.projects if p['status'] == 'On Track')
        st.metric("On Track", on_track)
    with cols[2]:
        high_priority = sum(1 for p in st.session_state.projects if p['priority'] == 'High')
        st.metric("High Priority", high_priority)

    for idx, project in enumerate(st.session_state.projects):
        with st.expander(f"📋 {project['name']} ({project['status']})"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Project ID:** {project['project_id']}")
                st.write(f"**Type:** {project['type']}")
                st.write(f"**Priority:** {project['priority']}")

            with col2:
                st.write(f"**Start Date:** {project['start_date']}")
                st.write(f"**End Date:** {project['end_date']}")
                st.write(f"**Budget:** ${project['budget']:,}")

            with col3:
                new_status = st.selectbox(
                    "Update Status",
                    ["On Track", "Delayed", "Ahead", "Completed"],
                    index=["On Track", "Delayed", "Ahead", "Completed"].index(project['status']),
                    key=f"status_{idx}"
                )

                completion = st.slider(
                    "Completion %",
                    0, 100,
                    value=project['completion_percentage'],
                    key=f"completion_{idx}"
                )

                if st.button("Update", key=f"update_{idx}"):
                    st.session_state.projects[idx]['status'] = new_status
                    st.session_state.projects[idx]['completion_percentage'] = completion
                    st.success("Project updated successfully!")

                if st.button("Delete Project", key=f"delete_{idx}"):
                    st.session_state.projects.pop(idx)
                    st.rerun()

    st.header("Project Timeline")
    if len(st.session_state.projects) > 0:
        df_timeline = pd.DataFrame(st.session_state.projects)
        fig = px.timeline(
            df_timeline,
            x_start="start_date",
            x_end="end_date",
            y="name",
            color="status",
            title="Project Timeline Overview"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No active projects. Add a new project to get started.")

st.sidebar.title("Project Manager Controls")

st.sidebar.subheader("Risk Management")
risk_threshold = st.sidebar.slider(
    "Risk Alert Threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    help="Set the threshold for risk alerts"
)

st.sidebar.subheader("Resource Management")
resource_utilization_target = st.sidebar.slider(
    "Target Resource Utilization (%)",
    min_value=50,
    max_value=100,
    value=80,
    help="Set the target resource utilization rate"
)

st.sidebar.subheader("Project Monitoring")
delay_tolerance = st.sidebar.number_input(
    "Schedule Delay Tolerance (days)",
    min_value=0,
    max_value=30,
    value=5,
    help="Maximum acceptable delay in project timeline"
)


st.markdown("---")
st.markdown("© 2024 AI Construction Project Management Platform")