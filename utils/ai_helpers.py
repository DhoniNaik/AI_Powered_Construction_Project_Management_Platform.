from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from .ai_models import (
    RiskAssessmentModel,
    TimelinePredictor,
    ResourceOptimizer,
    SafetyHazardDetector
)

def prepare_project_data(project: Dict[str, Any]) -> Dict[str, float]:
    """
    Prepare project data for AI model analysis.
    """
    return {
        'weather_severity_score': float(project.get('weather_severity', 0.5)),
        'resource_availability_score': float(project.get('resource_availability', 0.8)),
        'historical_performance_score': float(project.get('performance_score', 0.7)),
        'planned_duration': float(project.get('duration', 90)),
        'complexity_score': float(project.get('complexity', 0.5)),
        'scope_size': float(project.get('scope_size', 100))
    }

def analyze_project(project: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive project analysis using AI models.
    """
    project_data = prepare_project_data(project)

    risk_model = RiskAssessmentModel()
    timeline_model = TimelinePredictor()
    resource_model = ResourceOptimizer()

    risk_assessment = risk_model.predict_risk(project_data)
    timeline_prediction = timeline_model.predict_timeline(project_data)
    resource_needs = resource_model.predict_resource_needs(project_data)

    return {
        'project_id': project.get('project_id'),
        'analysis_timestamp': datetime.now().isoformat(),
        'risk_assessment': risk_assessment,
        'timeline_prediction': timeline_prediction,
        'resource_needs': resource_needs,
    }

def analyze_safety_image(image_data: np.ndarray) -> Dict[str, Any]:
    """
    Analyze construction site image for safety hazards.
    """
    safety_model = SafetyHazardDetector()

    detected_hazards = safety_model.analyze_image(image_data)

    site_assessment = safety_model.assess_site_safety(detected_hazards)

    return {
        'hazards': detected_hazards,
        'site_assessment': site_assessment,
        'analysis_timestamp': datetime.now().isoformat()
    }

def optimize_resources(available_resources: Dict[str, float],
                      projects: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Optimize resource allocation across multiple projects.
    """
    resource_model = ResourceOptimizer()

    project_requirements = []
    for project in projects:
        project_data = prepare_project_data(project)
        needs = resource_model.predict_resource_needs(project_data)
        project_requirements.append({
            'project_id': project['project_id'],
            'resource_units': sum(needs['predicted_resources'].values())
        })

    allocation = resource_model.optimize_allocation(
        available_resources,
        project_requirements
    )

    return {
        'allocation_plan': allocation,
        'optimization_timestamp': datetime.now().isoformat()
    }

def get_project_insights(project_id: str) -> Dict[str, Any]:
    """
    Get comprehensive insights for a specific project.
    """
    project = {
        'project_id': project_id,
        'weather_severity': 0.3,
        'resource_availability': 0.9,
        'performance_score': 0.85,
        'duration': 120,
        'complexity': 0.6,
        'scope_size': 150
    }
    
    analysis = analyze_project(project)
    
    return {
        'project_id': project_id,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'risk_level': analysis['risk_assessment']['risk_level'],
        'predicted_duration': analysis['timeline_prediction']['predicted_duration'],
        'delay_risk': analysis['timeline_prediction']['delay_risk'],
        'resource_requirements': analysis['resource_needs']['predicted_resources'],
        'recommendations': [
            f"Risk Level: {analysis['risk_assessment']['risk_level']}",
            f"Expected Delay: {analysis['timeline_prediction']['delay_days']} days",
            "Monitor resource allocation closely"
        ]
    }