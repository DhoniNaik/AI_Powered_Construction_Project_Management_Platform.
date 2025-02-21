import streamlit as st
import plotly.express as px
from utils.data_generator import generate_sample_data
from utils.ml_models import ResourceOptimizer

st.title("Resource Management Dashboard")

data = generate_sample_data()
resource_data = data['resources']

resource_optimizer = ResourceOptimizer()

st.subheader("Resource Utilization Overview")
col1, col2 = st.columns(2)

with col1:
    fig_util = px.bar(resource_data.groupby('resource_type')['utilization'].mean().reset_index(),
                      x='resource_type', y='utilization',
                      title='Average Resource Utilization by Type')
    st.plotly_chart(fig_util)

with col2:
    fig_trend = px.line(resource_data, x='date', y='utilization',
                        color='resource_type',
                        title='Resource Utilization Trend')
    st.plotly_chart(fig_trend)

st.subheader("Project-wise Resource Allocation")
fig_project = px.sunburst(resource_data,
                         path=['project', 'resource_type'],
                         values='utilization',
                         title='Resource Allocation by Project')
st.plotly_chart(fig_project)

st.subheader("Resource Optimization Recommendations")
recommendations = resource_optimizer.optimize_allocation(resource_data, {})
for rec in recommendations:
    st.info(rec)

st.subheader("Resource Planning")
selected_project = st.selectbox("Select Project", resource_data['project'].unique())
project_resources = resource_data[resource_data['project'] == selected_project]

fig_project_resources = px.bar(project_resources,
                              x='resource_type',
                              y='utilization',
                              title=f'Resource Utilization in {selected_project}')
st.plotly_chart(fig_project_resources)
