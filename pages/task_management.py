import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from utils.data_generator import generate_sample_data

st.title("Task Management & Worker Productivity")

if 'task_assignments' not in st.session_state:
    st.session_state.task_assignments = []
if 'task_updates' not in st.session_state:
    st.session_state.task_updates = []

data = generate_sample_data()

def calculate_expertise_score(worker_data, assignment_history):
    base_score = 0

    years_exp = float(worker_data['years_experience'])
    exp_score = min(40, years_exp * 3)
    base_score += exp_score

    cert_level = {
        'Basic Safety': 5,
        'OSHA Safety': 15,
        'Equipment Operator': 20,
        'PMP': 25,
        'Electrical License': 25,
        'Architecture License': 30,
        'Structural Engineer': 30,
        'Civil Engineer': 30
    }
    cert_score = cert_level.get(worker_data['certification'], 0)
    base_score += cert_score

    if not assignment_history.empty:
        avg_performance = assignment_history['performance_score'].mean()
        performance_score = (avg_performance / 100) * 30
        base_score += performance_score

    return min(100, base_score)

st.header("🎯 Intelligent Task Assignment")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Available Workers")
    try:
        workers_df = pd.read_csv('data/labor_force_training.csv')
        assignments_df = pd.read_csv('data/work_assignments_training.csv')

        available_workers = []
        for _, worker in workers_df[workers_df['availability_status'] == 'Available'].iterrows():
            worker_history = assignments_df[assignments_df['worker_id'] == worker['worker_id']]
            expertise_score = calculate_expertise_score(worker, worker_history)
            available_workers.append({
                **worker.to_dict(),
                'expertise_score': expertise_score
            })

        available_workers_df = pd.DataFrame(available_workers)
        st.dataframe(available_workers_df[[
            'worker_id', 'name', 'skill_level', 'certification', 
            'specialization', 'expertise_score'
        ]])
    except Exception as e:
        st.error(f"Error loading worker data: {str(e)}")

with col2:
    st.subheader("New Task Assignment")
    task_description = st.text_input("Task Description")
    required_skill = st.selectbox("Required Skill Level", ["Beginner", "Intermediate", "Advanced", "Expert"])
    specialization = st.selectbox("Required Specialization", [
        "Project Management", "Safety Supervision", "Heavy Equipment",
        "Structural Engineering", "Electrical", "Civil Engineering",
        "HVAC Systems", "Architecture"
    ])
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Urgent"])
    deadline = st.date_input("Deadline", min_value=datetime.now())

    if st.button("Assign Task"):
        if task_description and required_skill and priority and deadline:
            matching_workers = pd.DataFrame(available_workers)
            matching_workers = matching_workers[
                (matching_workers['skill_level'] == required_skill) &
                (matching_workers['specialization'] == specialization)
            ]

            if not matching_workers.empty:
                best_worker = matching_workers.nlargest(1, 'expertise_score').iloc[0]
                new_assignment = {
                    'task_id': f"T{len(st.session_state.task_assignments) + 1:03d}",
                    'worker_id': best_worker['worker_id'],
                    'worker_name': best_worker['name'],
                    'task_description': task_description,
                    'specialization': specialization,
                    'priority': priority,
                    'deadline': deadline.strftime('%Y-%m-%d'),
                    'status': 'Assigned',
                    'assigned_date': datetime.now().strftime('%Y-%m-%d'),
                    'expertise_score': best_worker['expertise_score']
                }
                st.session_state.task_assignments.append(new_assignment)
                st.success(f"Task assigned to {best_worker['name']} (Expertise Score: {best_worker['expertise_score']:.1f})")
            else:
                st.warning("No available workers match the required skills and specialization")
        else:
            st.error("Please fill in all task details")

st.header("📊 Task Progress & Updates")
if st.session_state.task_assignments:
    for task in st.session_state.task_assignments:
        with st.expander(f"Task {task['task_id']}: {task['task_description']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Assigned to:** {task['worker_name']}")
                st.write(f"**Deadline:** {task['deadline']}")
                st.write(f"**Current Status:** {task['status']}")

            with col2:
                new_status = st.selectbox(
                    "Update Status",
                    ["Assigned", "In Progress", "Under Review", "Completed"],
                    key=f"status_{task['task_id']}"
                )
                progress = st.slider(
                    "Progress (%)",
                    0, 100,
                    key=f"progress_{task['task_id']}"
                )
                quality_rating = st.slider(
                    "Quality Rating",
                    1, 5,
                    key=f"quality_{task['task_id']}"
                )

                if st.button("Update Task", key=f"update_{task['task_id']}"):
                    task['status'] = new_status
                    task_update = {
                        'task_id': task['task_id'],
                        'update_date': datetime.now().strftime('%Y-%m-%d'),
                        'status': new_status,
                        'progress': progress,
                        'quality_rating': quality_rating
                    }
                    st.session_state.task_updates.append(task_update)
                    st.success("Task updated successfully")

st.header("📈 Productivity Analytics")

if st.session_state.task_updates:
    updates_df = pd.DataFrame(st.session_state.task_updates)

    completion_rate = (updates_df['status'] == 'Completed').mean() * 100
    st.metric("Task Completion Rate", f"{completion_rate:.1f}%")

    avg_quality = updates_df['quality_rating'].mean()
    st.metric("Average Quality Rating", f"{avg_quality:.1f}/5")

    fig_progress = px.line(updates_df, 
                          x='update_date',
                          y='progress',
                          color='task_id',
                          title='Task Progress Over Time')
    st.plotly_chart(fig_progress)

    fig_quality = px.histogram(updates_df,
                              x='quality_rating',
                              title='Quality Ratings Distribution')
    st.plotly_chart(fig_quality)
else:
    st.info("No task updates available yet")

st.markdown("---")
st.markdown("Use the sidebar to access other project management features")


# Worker Productivity Section (from original code, modified to integrate with new features)
st.header("📊 Worker Productivity Tracking")

# Load worker assignments
try:
    assignments_df = pd.read_csv('data/work_assignments_training.csv')
    
    # Performance Overview
    st.subheader("Performance Overview")
    fig_performance = px.box(assignments_df, 
                           x='project_name',
                           y='performance_score',
                           color='completion_status',
                           title='Performance Distribution by Project')
    st.plotly_chart(fig_performance)

    # Productivity Metrics (modified to include expertise score)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_performance = assignments_df['performance_score'].mean()
        st.metric("Average Performance", f"{avg_performance:.1f}%")
    
    with col2:
        completion_rate = (assignments_df['completion_status'] == 'Completed').mean() * 100
        st.metric("Task Completion Rate", f"{completion_rate:.1f}%")
    
    with col3:
        on_time_rate = (assignments_df['completion_status'].isin(['Completed', 'On Track'])).mean() * 100
        st.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%")

    # Productivity Trends (modified to include expertise score)
    st.subheader("Productivity Trends")
    productivity_trend = assignments_df.groupby('worker_id').agg({
        'performance_score': 'mean',
        'expertise_score': 'mean' # Added expertise score
    }).reset_index()
    
    # Merge with worker details
    worker_productivity = pd.merge(productivity_trend, workers_df[['worker_id', 'name', 'skill_level', 'expertise_score']], on='worker_id')
    
    fig_trend = px.bar(worker_productivity,
                      x='name',
                      y='performance_score',
                      color='skill_level',
                      title='Worker Performance Overview',
                      hover_data=['expertise_score']) # Added expertise_score to hover data
    st.plotly_chart(fig_trend)

    # Optimization Suggestions (modified to include expertise score)
    st.subheader("📈 Productivity Optimization Suggestions")
    
    # Generate suggestions based on performance data
    avg_performance = assignments_df['performance_score'].mean()
    low_performers = worker_productivity[worker_productivity['performance_score'] < avg_performance]
    for _, worker in low_performers.iterrows():
        st.warning(f"Suggestion for {worker['name']}: Consider additional training or mentoring to improve performance.  Expertise Score: {worker['expertise_score']:.1f}")
    
    # High performers recognition (modified to include expertise score)
    high_performers = worker_productivity[worker_productivity['performance_score'] > avg_performance + 5]
    for _, worker in high_performers.iterrows():
        st.success(f"Recognition: {worker['name']} is performing above average. Consider for mentoring roles. Expertise Score: {worker['expertise_score']:.1f}")
    
except Exception as e:
    st.error(f"Error analyzing productivity data: {str(e)}")

# Current Task Status
if st.session_state.task_assignments:
    st.header("📋 Current Task Assignments")
    tasks_df = pd.DataFrame(st.session_state.task_assignments)
    st.dataframe(tasks_df)