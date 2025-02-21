import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
import cv2

class RiskPredictor:
    """
    Risk prediction model incorporating weather data and site conditions.
    """
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self._initialize_with_sample_data()

    def _initialize_with_sample_data(self):
        """Initialize the model with sample data to avoid cold start issues."""
        sample_data = pd.DataFrame({
            'weather_severity_score': np.random.uniform(0, 1, 100),
            'site_condition_score': np.random.uniform(0, 1, 100),
            'risk_score': np.random.uniform(0, 100, 100)
        })
        self.train(sample_data)

    def train(self, data: pd.DataFrame) -> None:
        """Train the risk prediction model using weather conditions and site data."""
        try:
            required_columns = ['weather_severity_score', 'site_condition_score', 'risk_score']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0.5  
            features = data[['weather_severity_score', 'site_condition_score']]

            self.scaler.fit(features)
            X = self.scaler.transform(features)
            y = data['risk_score']

            self.model.fit(X, y)
        except Exception as e:
            print(f"Error training model: {str(e)}")
            self._initialize_with_sample_data()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make risk predictions combining weather and site conditions."""
        try:
            for col in ['weather_severity_score', 'site_condition_score']:
                if col not in data.columns:
                    data[col] = 0.5  
            features = data[['weather_severity_score', 'site_condition_score']]

            X = self.scaler.transform(features)
            predictions = self.model.predict(X)

            return np.clip(predictions, 0, 100)
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return np.full(len(data), 50.0)  

class ResourceOptimizer:
    """
    Resource optimization model incorporating economic trends
    and shortage predictions.
    """
    def __init__(self):
        self.price_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )

    def analyze_price_trends(self, resource_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze and predict price trends for resources.
        """
        price_trends = {}
        for resource in resource_data['resource_type'].unique():
            resource_hist = resource_data[resource_data['resource_type'] == resource]
            try:
                if 'price_fluctuation' in resource_hist.columns:
                    price_changes = []
                    for val in resource_hist['price_fluctuation']:
                        try:
                            if isinstance(val, str) and '%' in val:
                                price_changes.append(float(val.rstrip('%')) / 100)
                            elif isinstance(val, (int, float)):
                                price_changes.append(float(val))
                        except (ValueError, TypeError):
                            continue
                    if price_changes:
                        price_trends[resource] = np.mean(price_changes)
                    else:
                        price_trends[resource] = 0.0
            except Exception:
                price_trends[resource] = 0.0
        return price_trends

    def predict_shortages(self, resource_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict potential resource shortages based on historical patterns.
        """
        shortage_predictions = []
        risk_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8, 'Critical': 1.0}

        for resource in resource_data['resource_type'].unique():
            try:
                resource_hist = resource_data[resource_data['resource_type'] == resource]

                shortage_risk = 0.0
                if 'shortage_risk' in resource_hist.columns:
                    risks = []
                    for risk in resource_hist['shortage_risk']:
                        if isinstance(risk, str) and risk in risk_map:
                            risks.append(risk_map[risk])
                        elif isinstance(risk, (int, float)):
                            risks.append(min(1.0, max(0.0, float(risk))))
                    if risks:
                        shortage_risk = np.mean(risks)

                impact_level = 'High' if shortage_risk > 0.7 else 'Medium' if shortage_risk > 0.3 else 'Low'

                shortage_predictions.append({
                    'resource': resource,
                    'shortage_risk': shortage_risk,
                    'impact_level': impact_level
                })
            except Exception:
                shortage_predictions.append({
                    'resource': resource,
                    'shortage_risk': 0.2,
                    'impact_level': 'Low'
                })

        return shortage_predictions

    def optimize_allocation(self, resources: pd.DataFrame, constraints: Dict[str, Any]) -> List[str]:
        """
        Optimize resource allocation based on price trends and shortage risks.
        """
        recommendations = []

        try:
            price_trends = self.analyze_price_trends(resources)
            shortage_risks = self.predict_shortages(resources)

            for resource in resources['resource_type'].unique():
                price_trend = price_trends.get(resource, 0)
                shortage_risk = next(
                    (risk for risk in shortage_risks if risk['resource'] == resource),
                    {'shortage_risk': 0, 'impact_level': 'Low'}
                )

                if shortage_risk['impact_level'] == 'High':
                    recommendations.append(
                        f"URGENT: High shortage risk for {resource}. "
                        f"Consider immediate stockpiling and alternative suppliers."
                    )
                elif price_trend > 0.1:  
                    recommendations.append(
                        f"WARNING: Rising prices for {resource} (+{price_trend*100:.1f}%). "
                        f"Consider advance purchasing or alternative materials."
                    )
        except Exception as e:
            recommendations.append(f"Error analyzing resources: {str(e)}")

        return recommendations

class SafetyHazardDetector:
    """
    Computer vision-based safety hazard detection model.
    """
    def __init__(self):
        self.confidence_threshold = 0.6
        self.hazard_categories = {
            'trip_hazard': ['uneven_surface', 'obstacles'],
            'fall_hazard': ['edges', 'openings'],
            'lighting_hazard': ['dark_areas', 'glare'],
            'equipment_hazard': ['moving_machinery', 'power_tools']
        }

    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process image and detect potential safety hazards.
        """
        hazards = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)
        if np.mean(edges) > 50:
            hazards.append({
                'type': 'Fall Hazard',
                'confidence': min(1.0, np.mean(edges) / 100),
                'location': 'Edge or height difference detected'
            })

        avg_brightness = np.mean(gray)
        if avg_brightness < 100:
            hazards.append({
                'type': 'Lighting Hazard',
                'confidence': min(1.0, (100 - avg_brightness) / 100),
                'location': 'Poor lighting conditions detected'
            })

        return hazards

    def analyze_site_conditions(self, hazard_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze historical hazard data to identify patterns and predict risk areas.
        """
        if not hazard_history:
            return {'risk_level': 'Low', 'recommendations': ['Insufficient historical data']}

        hazard_types = [h['type'] for h in hazard_history]
        hazard_counts = pd.Series(hazard_types).value_counts()

        total_hazards = len(hazard_history)
        risk_score = min(1.0, total_hazards / 10)  # Scale risk with hazard frequency

        recommendations = []
        for hazard_type, count in hazard_counts.items():
            if count >= 2:
                recommendations.append(
                    f"Frequent {hazard_type.lower()} detections. "
                    f"Consider implementing additional safety measures."
                )

        return {
            'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
            'risk_score': risk_score,
            'recommendations': recommendations
        }