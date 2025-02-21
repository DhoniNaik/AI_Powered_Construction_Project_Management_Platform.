import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

def generate_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Generate mock data for the construction project management platform.

    Training Data Sources:
    1. data/risk_training.csv - Historical risk analysis data
    2. data/resource_training.csv - Resource utilization patterns
    3. data/supply_chain_training.csv - Supply chain metrics
    4. data/project_timeline_training.csv - Project timeline data

    Production Data Sources (Future Implementation):
    1. Project Management Information System (PMIS)
    2. IoT sensors from construction sites
    3. Weather APIs
    4. Labor management systems
    5. Equipment tracking systems

    Returns:
        Dictionary containing DataFrames for different aspects of the project
    """
    projects = ['Metro Line Extension', 'Highway Expansion', 'Smart City Complex', 'Airport Terminal']

    try:
        risk_data = pd.read_csv('data/risk_training.csv')
        resource_data = pd.read_csv('data/resource_training.csv')
        supply_chain_data = pd.read_csv('data/supply_chain_training.csv')
        timeline_data = pd.read_csv('data/project_timeline_training.csv')

        risk_data['date'] = pd.to_datetime(risk_data['date'])
        resource_data['date'] = pd.to_datetime(resource_data['date'])
        supply_chain_data['date'] = pd.to_datetime(supply_chain_data['date'])
        timeline_data['start_date'] = pd.to_datetime(timeline_data['start_date'])
        timeline_data['end_date'] = pd.to_datetime(timeline_data['end_date'])

        timeline_data.columns = timeline_data.columns.str.lower()

    except FileNotFoundError:
        start_dates = [datetime.now() + timedelta(days=x*30) for x in range(len(projects))]
        timeline_data = pd.DataFrame({
            'project': projects,
            'start_date': start_dates,
            'end_date': [date + timedelta(days=90) for date in start_dates],
            'status': np.random.choice(['On Track', 'Delayed', 'Ahead'], len(projects))
        })

        risk_categories = [
            'Weather Risk',
            'Supply Chain Delay',
            'Labor Shortage',
            'Technical Challenge',
            'Regulatory Compliance'
        ]
        risk_data = pd.DataFrame({
            'date': pd.date_range(start=datetime.now()-timedelta(days=180), periods=180),
            'risk_score': np.random.normal(50, 15, 180).clip(0, 100),
            'category': np.random.choice(risk_categories, 180)
        })

        resource_types = ['Heavy Equipment', 'Skilled Labor', 'Construction Materials', 'Engineering Staff']
        resource_data = pd.DataFrame({
            'resource_type': np.repeat(resource_types, 15),
            'utilization': np.random.uniform(60, 95, 60),
            'project': np.random.choice(projects, 60),
            'date': pd.date_range(start=datetime.now()-timedelta(days=60), periods=60)
        })

    alert_types = [
        'Safety Incident Risk',
        'Resource Shortage',
        'Cost Overrun Risk',
        'Schedule Delay Risk',
        'Quality Control Alert'
    ]
    alerts_data = pd.DataFrame({
        'date': pd.date_range(start=datetime.now()-timedelta(days=10), periods=20),
        'type': np.random.choice(alert_types, 20),
        'message': [
            f"Alert {i+1}: {np.random.choice(alert_types)} detected in {np.random.choice(projects)}"
            for i in range(20)
        ]
    })

    return {
        'timeline': timeline_data,
        'alerts': alerts_data,
        'risks': risk_data,
        'resources': resource_data
    }