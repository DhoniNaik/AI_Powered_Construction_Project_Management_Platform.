import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
import cv2
from datetime import datetime

class RiskAssessmentModel:
    def __init__(self):
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with sample data to avoid cold start"""
        sample_data = pd.DataFrame({
            'weather_severity_score': np.random.uniform(0, 1, 100),
            'resource_availability_score': np.random.uniform(0, 1, 100),
            'historical_performance_score': np.random.uniform(0, 1, 100)
        })
        sample_risk = np.random.choice(['Low', 'Medium', 'High'], 100)
        self.train(sample_data, sample_risk)

    def train(self, features: pd.DataFrame, risk_levels: np.ndarray) -> None:
        """Train the risk assessment model."""
        try:
            X = self.scaler.fit_transform(features)
            self.risk_classifier.fit(X, risk_levels)
        except Exception as e:
            print(f"Error training model: {str(e)}")
            self._initialize_model()

    def predict_risk(self, project_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict risk level for a project."""
        try:
            features = np.array([[
                project_data['weather_severity_score'],
                project_data['resource_availability_score'],
                project_data['historical_performance_score']
            ]])

            X = self.scaler.transform(features)
            risk_level = self.risk_classifier.predict(X)[0]
            risk_proba = self.risk_classifier.predict_proba(X)[0]

            return {
                'risk_level': risk_level,
                'confidence': float(max(risk_proba)),
                'risk_factors': {
                    'weather': project_data['weather_severity_score'],
                    'resources': project_data['resource_availability_score'],
                    'historical': project_data['historical_performance_score']
                }
            }
        except Exception as e:
            print(f"Error predicting risk: {str(e)}")
            return {
                'risk_level': 'Medium',
                'confidence': 0.5,
                'risk_factors': project_data
            }

class TimelinePredictor:
    """
    AI model for predicting project timelines and potential delays
    based on historical project data and current conditions.
    """
    def __init__(self):
        self.timeline_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with sample data to avoid cold start"""
        sample_data = pd.DataFrame({
            'planned_duration': np.random.uniform(30, 365, 100),
            'complexity_score': np.random.uniform(0, 1, 100),
            'resource_availability_score': np.random.uniform(0, 1, 100),
            'actual_duration': np.random.uniform(30, 400, 100)
        })
        self.train(sample_data)

    def train(self, historical_projects: pd.DataFrame) -> None:
        """Train the timeline prediction model."""
        try:
            # Ensure required columns exist
            required_columns = [
                'planned_duration',
                'complexity_score',
                'resource_availability_score',
                'actual_duration'
            ]

            for col in required_columns:
                if col not in historical_projects.columns:
                    if col == 'actual_duration':
                        historical_projects[col] = historical_projects['planned_duration'] * 1.1
                    else:
                        historical_projects[col] = 0.5  # Default moderate value

            # Extract relevant features
            features = historical_projects[['planned_duration', 'complexity_score', 'resource_availability_score']]

            # Scale features
            self.scaler.fit(features)
            X = self.scaler.transform(features)
            y = historical_projects['actual_duration']

            # Train model
            self.timeline_predictor.fit(X, y)
        except Exception as e:
            print(f"Error training timeline model: {str(e)}")
            self._initialize_model()

    def predict_timeline(self, project_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict project timeline and potential delays."""
        try:
            # Ensure required fields exist
            required_fields = ['planned_duration', 'complexity_score', 'resource_availability_score']
            for field in required_fields:
                if field not in project_data:
                    project_data[field] = 0.5  # Default moderate value

            features = np.array([[
                project_data['planned_duration'],
                project_data['complexity_score'],
                project_data['resource_availability_score']
            ]])

            # Scale and predict
            X = self.scaler.transform(features)
            predicted_duration = float(self.timeline_predictor.predict(X)[0])

            # Calculate delay metrics
            planned_duration = float(project_data['planned_duration'])
            delay_days = max(0, predicted_duration - planned_duration)
            delay_risk = delay_days / planned_duration if planned_duration > 0 else 0

            return {
                'predicted_duration': round(predicted_duration, 1),
                'delay_risk': round(delay_risk, 2),
                'delay_days': round(delay_days, 1),
                'confidence_score': 0.8,  # Could be calculated based on model metrics
                'factors': {
                    'complexity': project_data['complexity_score'],
                    'resource_availability': project_data['resource_availability_score']
                }
            }
        except Exception as e:
            print(f"Error predicting timeline: {str(e)}")
            return {
                'predicted_duration': project_data['planned_duration'],
                'delay_risk': 0.0,
                'delay_days': 0.0,
                'confidence_score': 0.5,
                'factors': project_data
            }

class ResourceOptimizer:
    """
    AI model for optimizing resource allocation and predicting
    resource needs based on project requirements.
    """
    def __init__(self):
        self.resource_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
    
    def predict_resource_needs(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict resource requirements for a project."""
        base_resources = {
            'labor_hours': project_data['scope_size'] * 40,  # 40 hours per scope unit
            'equipment_hours': project_data['scope_size'] * 20,  # 20 hours per scope unit
            'material_units': project_data['scope_size'] * 100  # 100 units per scope unit
        }
        
        complexity_factor = 1 + (project_data.get('complexity_score', 0) * 0.2)
        weather_factor = 1 + (project_data.get('weather_severity_score', 0) * 0.1)
        
        adjusted_resources = {
            key: value * complexity_factor * weather_factor
            for key, value in base_resources.items()
        }
        
        return {
            'predicted_resources': adjusted_resources,
            'adjustment_factors': {
                'complexity': complexity_factor,
                'weather': weather_factor
            }
        }
    
    def optimize_allocation(self, available_resources: Dict[str, float],
                          project_requirements: List[Dict[str, float]]) -> Dict[str, Any]:
        """Optimize resource allocation across multiple projects."""
        total_demand = sum(req['resource_units'] for req in project_requirements)
        allocation_ratio = min(1.0, available_resources['total_units'] / total_demand)
        
        optimized_allocation = []
        for project in project_requirements:
            allocated = project['resource_units'] * allocation_ratio
            optimized_allocation.append({
                'project_id': project['project_id'],
                'allocated_units': allocated,
                'allocation_ratio': allocation_ratio,
                'shortage': project['resource_units'] - allocated if allocation_ratio < 1 else 0
            })
        
        return {
            'allocations': optimized_allocation,
            'total_allocated': sum(alloc['allocated_units'] for alloc in optimized_allocation),
            'shortage_warning': allocation_ratio < 1
        }

class SafetyHazardDetector:
    """
    AI model for detecting safety hazards in construction site images
    and predicting potential safety risks.
    """
    def __init__(self):
        self.hazard_confidence_threshold = 0.6
        self.hazard_categories = {
            'fall_hazard': ['edges', 'openings', 'scaffolding'],
            'equipment_hazard': ['heavy_machinery', 'power_tools'],
            'material_hazard': ['chemicals', 'combustibles'],
            'ppe_violation': ['missing_helmet', 'missing_vest']
        }
    
    def analyze_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze construction site image for safety hazards."""
        detected_hazards = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 50:
            detected_hazards.append({
                'type': 'fall_hazard',
                'confidence': min(1.0, np.mean(edges) / 100),
                'location': 'Edge or height difference detected',
                'severity': 'High'
            })
        
        brightness = np.mean(gray)
        if brightness < 100:
            detected_hazards.append({
                'type': 'lighting_hazard',
                'confidence': min(1.0, (100 - brightness) / 100),
                'location': 'Poor lighting conditions',
                'severity': 'Medium'
            })
        
        return detected_hazards
    
    def assess_site_safety(self, hazard_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall site safety based on hazard history."""
        if not hazard_history:
            return {
                'safety_score': 1.0,
                'risk_level': 'Low',
                'recommendations': ['Insufficient hazard detection history']
            }
        
        hazard_counts = {}
        for hazard in hazard_history:
            hazard_type = hazard['type']
            hazard_counts[hazard_type] = hazard_counts.get(hazard_type, 0) + 1
        
        total_hazards = len(hazard_history)
        safety_score = max(0, 1 - (total_hazards * 0.1))
        
        recommendations = []
        for hazard_type, count in hazard_counts.items():
            if count >= 2:
                recommendations.append(
                    f"Frequent {hazard_type} detections. Implement additional "
                    f"safety measures in affected areas."
                )
        
        return {
            'safety_score': round(safety_score, 2),
            'risk_level': 'High' if safety_score < 0.6 else 'Medium' if safety_score < 0.8 else 'Low',
            'hazard_frequency': hazard_counts,
            'recommendations': recommendations
        }

def predict_project_metrics(project_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze project data and predict various metrics using all AI models.
    """
    risk_model = RiskAssessmentModel()
    timeline_model = TimelinePredictor()
    resource_model = ResourceOptimizer()
    safety_model = SafetyHazardDetector()
    
    risk_assessment = risk_model.predict_risk(project_data)
    timeline_prediction = timeline_model.predict_timeline(project_data)
    resource_prediction = resource_model.predict_resource_needs(project_data)
    
    return {
        'risk_assessment': risk_assessment,
        'timeline_prediction': timeline_prediction,
        'resource_prediction': resource_prediction,
        'timestamp': datetime.now().isoformat()
    }