import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.data_generator import generate_sample_data

st.title("Supply Chain Analytics Dashboard")

def generate_supply_chain_data():
    suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D']
    materials = ['Cement', 'Steel', 'Concrete', 'Timber']

    supply_data = pd.DataFrame({
        'supplier': np.random.choice(suppliers, 100),
        'material': np.random.choice(materials, 100),
        'delivery_time': np.random.normal(10, 2, 100),
        'cost': np.random.normal(1000, 200, 100),
        'quality_score': np.random.uniform(70, 100, 100),
        'date': pd.date_range(start='2024-01-01', periods=100)
    })
    return supply_data

supply_data = generate_supply_chain_data()

st.subheader("Supplier Performance Matrix")
supplier_performance = supply_data.groupby('supplier').agg({
    'delivery_time': 'mean',
    'quality_score': 'mean',
    'cost': 'mean'
}).reset_index()

fig_performance = px.scatter(supplier_performance,
                           x='delivery_time',
                           y='quality_score',
                           size='cost',
                           color='supplier',
                           title='Supplier Performance Matrix')
st.plotly_chart(fig_performance)

st.subheader("Material Cost Tracking")
material_cost = supply_data.groupby(['date', 'material'])['cost'].mean().reset_index()
fig_cost = px.line(material_cost,
                   x='date',
                   y='cost',
                   color='material',
                   title='Material Cost Trends')
st.plotly_chart(fig_cost)

st.subheader("Supplier Quality Scores")
fig_quality = px.box(supply_data,
                    x='supplier',
                    y='quality_score',
                    title='Supplier Quality Distribution')
st.plotly_chart(fig_quality)

st.subheader("Inventory Forecasting")
forecast_days = 30
material = st.selectbox("Select Material", supply_data['material'].unique())

material_data = supply_data[supply_data['material'] == material]
moving_avg = material_data['cost'].rolling(window=7).mean()

forecast_dates = pd.date_range(start=supply_data['date'].max(),
                             periods=forecast_days)
forecast_values = [moving_avg.iloc[-1]] * forecast_days

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=material_data['date'],
                                 y=material_data['cost'],
                                 name='Actual Cost'))
fig_forecast.add_trace(go.Scatter(x=forecast_dates,
                                 y=forecast_values,
                                 name='Forecast',
                                 line=dict(dash='dash')))
fig_forecast.update_layout(title=f'{material} Cost Forecast')
st.plotly_chart(fig_forecast)