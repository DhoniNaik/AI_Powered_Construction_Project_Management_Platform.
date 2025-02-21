import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.data_generator import generate_sample_data
from utils.ml_models import RiskPredictor, ResourceOptimizer

LOCATIONS = [
    "Mumbai, Maharashtra",
    "Delhi, Delhi",
    "Bangalore, Karnataka",
    "Hyderabad, Telangana",
    "Chennai, Tamil Nadu",
    "Kolkata, West Bengal",
    "Pune, Maharashtra",
    "Ahmedabad, Gujarat",
    "Jaipur, Rajasthan",
    "Lucknow, Uttar Pradesh",
    "Chandigarh, Punjab",
    "Bhubaneswar, Odisha",
    "Guwahati, Assam",
    "Thiruvananthapuram, Kerala",
    "Indore, Madhya Pradesh",
    "Patna, Bihar",
    "Surat, Gujarat",
    "Nagpur, Maharashtra",
    "Bhopal, Madhya Pradesh",
    "Visakhapatnam, Andhra Pradesh"
]

st.title("🏗️ Project Input & Analysis")

if 'projects' not in st.session_state:
    data = generate_sample_data()
    timeline_data = data['timeline']
    st.session_state.projects = [{
        'project_id': f"P{i+1:03d}",
        'name': row['project'],
        'type': 'Infrastructure',
        'location': LOCATIONS[i % len(LOCATIONS)],  
        'start_date': row['start_date'].strftime('%Y-%m-%d'),
        'end_date': row['end_date'].strftime('%Y-%m-%d'),
        'duration': (row['end_date'] - row['start_date']).days,
        'budget': 1000000,
        'priority': 'Medium',
        'status': row['status'],
        'completion_percentage': 0
    } for i, row in timeline_data.iterrows()]

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = {}

tab1, tab2 = st.tabs(["Project Management", "Data Upload"])

with tab1:
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
            project_location = st.selectbox(
                "Project Location",
                options=LOCATIONS,
                help="Select an Indian city for the project location"
            )

        with col2:
            project_duration = st.number_input("Duration (days)", min_value=1, value=90)
            budget = st.number_input("Budget (₹)", min_value=0, value=1000000)
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])

        if st.button("Add Project"):
            if new_project_name and project_location:
                new_project = {
                    'project_id': f"P{len(st.session_state.projects) + 1:03d}",
                    'name': new_project_name,
                    'type': project_type,
                    'location': project_location,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': (start_date + timedelta(days=project_duration)).strftime('%Y-%m-%d'),
                    'duration': project_duration,
                    'budget': budget,
                    'priority': priority,
                    'status': "On Track",
                    'completion_percentage': 0
                }
                st.session_state.projects.append(new_project)
                st.success(f"Project '{new_project_name}' added successfully!")
            else:
                st.error("Please enter both project name and select a location")

    st.subheader("🚧 Active Projects")

    if st.session_state.projects:
        status_cols = st.columns(3)
        with status_cols[0]:
            total_active = len(st.session_state.projects)
            st.metric("Total Projects", total_active)

        with status_cols[1]:
            on_track = sum(1 for p in st.session_state.projects if p['status'] == 'On Track')
            st.metric("On Track", on_track)

        with status_cols[2]:
            delayed = sum(1 for p in st.session_state.projects if p['status'] == 'Delayed')
            st.metric("Delayed", delayed)

        for idx, project in enumerate(st.session_state.projects):
            with st.expander(f"📋 {project['name']} ({project['status']})"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Project ID:** {project['project_id']}")
                    st.write(f"**Type:** {project['type']}")
                    st.write(f"**Location:** {project['location']}")
                    st.write(f"**Priority:** {project['priority']}")

                with col2:
                    st.write(f"**Start Date:** {project['start_date']}")
                    st.write(f"**End Date:** {project['end_date']}")
                    st.write(f"**Budget:** ₹{project['budget']:,}")

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
        df_timeline = pd.DataFrame(st.session_state.projects)
        fig = px.timeline(
            df_timeline,
            x_start="start_date",
            x_end="end_date",
            y="name",
            color="status",
            title="Project Timeline Overview",
            hover_data=["location", "budget", "priority"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active projects. Add a new project to get started.")

with tab2:
    st.header("📊 Project Data Upload")

    if not st.session_state.projects:
        st.warning("Please create at least one project before uploading data.")
    else:
        selected_project = st.selectbox(
            "Select Project",
            options=[p['name'] for p in st.session_state.projects],
            help="Choose the project to upload data for"
        )

        data_type = st.selectbox(
            "Select Data Type",
            options=[
                "Weather Data",
                "Site Conditions",
                "Resource Data",
                "Labor Force",
                "Work Assignments"
            ]
        )

        upload_file = st.file_uploader(
            f"Upload {data_type} CSV file",
            type="csv",
            help="Upload a CSV file with the required data"
        )

        if upload_file:
            try:
                df = pd.read_csv(upload_file)
                df['project'] = selected_project  # Add project association

                data_key = data_type.lower().replace(" ", "_")
                st.session_state.uploaded_data[data_key] = df

                st.subheader("Data Preview")
                st.dataframe(df.head())

                st.subheader("Data Visualization")

                if data_type == "Weather Data":
                    if 'weather_severity_score' in df.columns:
                        fig = px.line(df, x='date', y='weather_severity_score',
                                    title=f'Weather Severity Trend - {selected_project}')
                        st.plotly_chart(fig)

                elif data_type == "Resource Data":
                    if 'resource_type' in df.columns and 'price_fluctuation' in df.columns:
                        fig = px.bar(df, x='resource_type', y='price_fluctuation',
                                   title=f'Resource Price Analysis - {selected_project}')
                        st.plotly_chart(fig)

                elif data_type == "Labor Force":
                    if 'skill_level' in df.columns:
                        fig = px.pie(df, names='skill_level',
                                   title=f'Workforce Skill Distribution - {selected_project}')
                        st.plotly_chart(fig)

                elif data_type == "Work Assignments":
                    if 'completion_status' in df.columns:
                        fig = px.bar(df['completion_status'].value_counts(),
                                   title=f'Task Completion Status - {selected_project}')
                        st.plotly_chart(fig)

                st.success(f"{data_type} uploaded successfully!")

            except Exception as e:
                st.error(f"Error processing {data_type}: {str(e)}")

st.markdown("---")
st.markdown("Use the tabs above to manage projects and upload project data")