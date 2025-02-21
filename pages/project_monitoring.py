import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.data_generator import generate_sample_data

st.title("Project Monitoring Dashboard")

data = generate_sample_data()
timeline_data = data['timeline']

st.subheader("Project Timeline")
fig_timeline = px.timeline(timeline_data,
                         x_start="start_date",
                         x_end="end_date",
                         y="project",
                         color="status",
                         title="Project Timeline Overview")
st.plotly_chart(fig_timeline)

st.subheader("Project Status Summary")
status_count = timeline_data['status'].value_counts()
fig_status = px.pie(values=status_count.values,
                    names=status_count.index,
                    title='Project Status Distribution')
st.plotly_chart(fig_status)

st.subheader("Project Progress Tracking")
progress_data = timeline_data[['project', 'completion_percentage']]
progress_data['planned_progress'] = 100  

fig_progress = go.Figure()
fig_progress.add_trace(go.Bar(name='Planned Progress',
                             x=progress_data['project'],
                             y=progress_data['planned_progress']))
fig_progress.add_trace(go.Bar(name='Actual Progress',
                             x=progress_data['project'],
                             y=progress_data['completion_percentage']))
fig_progress.update_layout(barmode='group',
                          title='Planned vs Actual Progress')
st.plotly_chart(fig_progress)

st.subheader("Recent Project Alerts")
alerts = data['alerts'].sort_values('date', ascending=False)
st.dataframe(alerts)