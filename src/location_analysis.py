"""
Location-based Soil Analysis for India with Real-time Features
Provides detailed soil information based on Indian geographical locations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json

class IndianLocationAnalyzer:
    """Analyzes soil characteristics based on Indian locations"""
    
    def __init__(self):
        self.indian_states = {
            "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400},
            "Bihar": {"lat": 25.0961, "lon": 85.3131},
            "Gujarat": {"lat": 22.2587, "lon": 71.1924},
            "Haryana": {"lat": 29.0588, "lon": 76.0856},
            "Karnataka": {"lat": 15.3173, "lon": 75.7139},
            "Kerala": {"lat": 10.8505, "lon": 76.2711},
            "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569},
            "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
            "Punjab": {"lat": 31.1471, "lon": 75.3412},
            "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
            "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
            "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462},
            "West Bengal": {"lat": 22.9868, "lon": 87.8550},
            "Odisha": {"lat": 20.9517, "lon": 85.0985},
            "Telangana": {"lat": 18.1124, "lon": 79.0193}
        }
        
        self.soil_characteristics = {
            "Andhra Pradesh": {
                "dominant_soil": "Red Soil",
                "secondary_soil": "Black Soil",
                "ph_range": [6.0, 8.0],
                "organic_matter": "Low to Medium",
                "drainage": "Good",
                "fertility": "Medium",
                "major_crops": ["Rice", "Cotton", "Sugarcane", "Groundnut"],
                "challenges": ["Salinity in coastal areas", "Drought prone"],
                "recommendations": ["Organic matter addition", "Water conservation"]
            },
            "Bihar": {
                "dominant_soil": "Alluvial Soil",
                "secondary_soil": "Calcareous Soil",
                "ph_range": [7.0, 8.5],
                "organic_matter": "Medium to High",
                "drainage": "Poor to Medium",
                "fertility": "High",
                "major_crops": ["Rice", "Wheat", "Maize", "Sugarcane"],
                "challenges": ["Flooding", "Waterlogging"],
                "recommendations": ["Improved drainage", "Flood management"]
            },
            "Gujarat": {
                "dominant_soil": "Black Soil",
                "secondary_soil": "Alluvial Soil",
                "ph_range": [7.5, 9.0],
                "organic_matter": "Low",
                "drainage": "Poor",
                "fertility": "Medium to High",
                "major_crops": ["Cotton", "Groundnut", "Wheat", "Bajra"],
                "challenges": ["Salinity", "Water scarcity"],
                "recommendations": ["Salinity management", "Drip irrigation"]
            },
            "Haryana": {
                "dominant_soil": "Alluvial Soil",
                "secondary_soil": "Sandy Soil",
                "ph_range": [7.2, 8.8],
                "organic_matter": "Medium",
                "drainage": "Good",
                "fertility": "High",
                "major_crops": ["Wheat", "Rice", "Cotton", "Sugarcane"],
                "challenges": ["Declining water table", "Soil degradation"],
                "recommendations": ["Crop diversification", "Water-efficient crops"]
            },
            "Karnataka": {
                "dominant_soil": "Red Soil",
                "secondary_soil": "Black Soil",
                "ph_range": [6.0, 7.5],
                "organic_matter": "Low to Medium",
                "drainage": "Good",
                "fertility": "Medium",
                "major_crops": ["Rice", "Sugarcane", "Cotton", "Coffee"],
                "challenges": ["Nutrient deficiency", "Erosion"],
                "recommendations": ["Nutrient management", "Soil conservation"]
            },
            "Kerala": {
                "dominant_soil": "Laterite Soil",
                "secondary_soil": "Alluvial Soil",
                "ph_range": [4.5, 6.5],
                "organic_matter": "Medium to High",
                "drainage": "Excessive",
                "fertility": "Low to Medium",
                "major_crops": ["Rice", "Coconut", "Spices", "Rubber"],
                "challenges": ["Acidity", "Nutrient leaching"],
                "recommendations": ["Lime application", "Organic farming"]
            },
            "Madhya Pradesh": {
                "dominant_soil": "Black Soil",
                "secondary_soil": "Red Soil",
                "ph_range": [6.5, 8.5],
                "organic_matter": "Low to Medium",
                "drainage": "Poor to Medium",
                "fertility": "Medium to High",
                "major_crops": ["Wheat", "Soybean", "Cotton", "Rice"],
                "challenges": ["Drought", "Soil erosion"],
                "recommendations": ["Water harvesting", "Conservation tillage"]
            },
            "Maharashtra": {
                "dominant_soil": "Black Soil",
                "secondary_soil": "Red Soil",
                "ph_range": [7.0, 8.5],
                "organic_matter": "Low to Medium",
                "drainage": "Poor to Medium",
                "fertility": "Medium to High",
                "major_crops": ["Cotton", "Sugarcane", "Wheat", "Soybean"],
                "challenges": ["Irregular rainfall", "Soil degradation"],
                "recommendations": ["Drip irrigation", "Soil health cards"]
            },
            "Punjab": {
                "dominant_soil": "Alluvial Soil",
                "secondary_soil": "Sandy Loam",
                "ph_range": [7.0, 8.5],
                "organic_matter": "Medium",
                "drainage": "Good",
                "fertility": "High",
                "major_crops": ["Wheat", "Rice", "Cotton", "Maize"],
                "challenges": ["Declining soil fertility", "Water depletion"],
                "recommendations": ["Organic farming", "Crop rotation"]
            },
            "Rajasthan": {
                "dominant_soil": "Desert Soil",
                "secondary_soil": "Alluvial Soil",
                "ph_range": [7.5, 9.0],
                "organic_matter": "Very Low",
                "drainage": "Excessive",
                "fertility": "Low",
                "major_crops": ["Bajra", "Wheat", "Barley", "Cotton"],
                "challenges": ["Water scarcity", "Wind erosion"],
                "recommendations": ["Water conservation", "Agroforestry"]
            },
            "Tamil Nadu": {
                "dominant_soil": "Red Soil",
                "secondary_soil": "Black Soil",
                "ph_range": [6.0, 8.0],
                "organic_matter": "Low to Medium",
                "drainage": "Good",
                "fertility": "Medium",
                "major_crops": ["Rice", "Cotton", "Sugarcane", "Groundnut"],
                "challenges": ["Water stress", "Salinity"],
                "recommendations": ["Efficient irrigation", "Salt-tolerant varieties"]
            },
            "Uttar Pradesh": {
                "dominant_soil": "Alluvial Soil",
                "secondary_soil": "Black Soil",
                "ph_range": [7.0, 8.5],
                "organic_matter": "Medium",
                "drainage": "Medium",
                "fertility": "High",
                "major_crops": ["Wheat", "Rice", "Sugarcane", "Potato"],
                "challenges": ["Soil salinity", "Nutrient imbalance"],
                "recommendations": ["Balanced fertilization", "Gypsum application"]
            },
            "West Bengal": {
                "dominant_soil": "Alluvial Soil",
                "secondary_soil": "Laterite Soil",
                "ph_range": [6.0, 7.5],
                "organic_matter": "Medium to High",
                "drainage": "Poor to Medium",
                "fertility": "High",
                "major_crops": ["Rice", "Jute", "Tea", "Potato"],
                "challenges": ["Flooding", "Arsenic contamination"],
                "recommendations": ["Flood management", "Safe water sources"]
            },
            "Odisha": {
                "dominant_soil": "Red Soil",
                "secondary_soil": "Alluvial Soil",
                "ph_range": [5.5, 7.5],
                "organic_matter": "Medium",
                "drainage": "Medium",
                "fertility": "Medium",
                "major_crops": ["Rice", "Wheat", "Sugarcane", "Oilseeds"],
                "challenges": ["Cyclones", "Soil acidity"],
                "recommendations": ["Cyclone-resistant varieties", "Lime application"]
            },
            "Telangana": {
                "dominant_soil": "Red Soil",
                "secondary_soil": "Black Soil",
                "ph_range": [6.0, 8.0],
                "organic_matter": "Low to Medium",
                "drainage": "Good",
                "fertility": "Medium",
                "major_crops": ["Rice", "Cotton", "Maize", "Sugarcane"],
                "challenges": ["Drought", "Soil degradation"],
                "recommendations": ["Drip irrigation", "Soil conservation"]
            }
        }
    
    def get_location_info(self, state_name):
        """Get detailed location information for a state"""
        if state_name in self.soil_characteristics:
            return {
                "coordinates": self.indian_states[state_name],
                "soil_data": self.soil_characteristics[state_name]
            }
        return None
    
    def get_weather_data(self, lat, lon, api_key=None):
        """Get real-time weather data (mock implementation)"""
        # Mock weather data - in production, use actual weather API
        return {
            "temperature": np.random.normal(25, 5),
            "humidity": np.random.normal(65, 15),
            "rainfall_24h": np.random.exponential(2),
            "wind_speed": np.random.normal(10, 3),
            "pressure": np.random.normal(1013, 10),
            "last_updated": datetime.now().isoformat()
        }
    
    def analyze_soil_suitability(self, state_name, crop_type=None):
        """Analyze soil suitability for crops in a specific state"""
        location_info = self.get_location_info(state_name)
        if not location_info:
            return None
        
        soil_data = location_info["soil_data"]
        coordinates = location_info["coordinates"]
        
        # Get real-time weather
        weather = self.get_weather_data(coordinates["lat"], coordinates["lon"])
        
        # Calculate suitability scores
        suitability = {
            "overall_score": self._calculate_overall_score(soil_data, weather),
            "soil_health": self._assess_soil_health(soil_data),
            "climate_suitability": self._assess_climate_suitability(weather),
            "water_availability": self._assess_water_availability(soil_data, weather),
            "nutrient_status": self._assess_nutrient_status(soil_data)
        }
        
        return {
            "location": state_name,
            "coordinates": coordinates,
            "soil_characteristics": soil_data,
            "current_weather": weather,
            "suitability_analysis": suitability,
            "recommendations": self._generate_recommendations(soil_data, weather)
        }
    
    def _calculate_overall_score(self, soil_data, weather):
        """Calculate overall suitability score"""
        # Simple scoring algorithm
        fertility_score = 0.8 if soil_data["fertility"] == "High" else 0.6 if soil_data["fertility"] == "Medium" else 0.4
        drainage_score = 0.8 if soil_data["drainage"] == "Good" else 0.6 if soil_data["drainage"] == "Medium" else 0.4
        weather_score = 0.8 if 20 <= weather["temperature"] <= 30 else 0.6
        
        return (fertility_score + drainage_score + weather_score) / 3 * 100
    
    def _assess_soil_health(self, soil_data):
        """Assess soil health parameters"""
        return {
            "ph_status": "Optimal" if 6.0 <= soil_data["ph_range"][0] <= 7.5 else "Needs adjustment",
            "organic_matter": soil_data["organic_matter"],
            "fertility_level": soil_data["fertility"],
            "drainage_status": soil_data["drainage"]
        }
    
    def _assess_climate_suitability(self, weather):
        """Assess climate suitability"""
        temp_suitability = "Good" if 15 <= weather["temperature"] <= 35 else "Moderate"
        humidity_suitability = "Good" if 40 <= weather["humidity"] <= 80 else "Moderate"
        
        return {
            "temperature_suitability": temp_suitability,
            "humidity_suitability": humidity_suitability,
            "current_conditions": "Favorable" if temp_suitability == "Good" and humidity_suitability == "Good" else "Moderate"
        }
    
    def _assess_water_availability(self, soil_data, weather):
        """Assess water availability"""
        drainage_factor = 0.8 if soil_data["drainage"] == "Good" else 0.6 if soil_data["drainage"] == "Medium" else 0.4
        rainfall_factor = min(weather["rainfall_24h"] / 10, 1.0)  # Normalize to 0-1
        
        return {
            "water_retention": "Good" if drainage_factor > 0.7 else "Moderate",
            "recent_rainfall": f"{weather['rainfall_24h']:.1f} mm",
            "irrigation_need": "Low" if rainfall_factor > 0.5 else "High"
        }
    
    def _assess_nutrient_status(self, soil_data):
        """Assess nutrient status based on soil type"""
        soil_type = soil_data["dominant_soil"]
        
        nutrient_profiles = {
            "Alluvial Soil": {"N": "High", "P": "Medium", "K": "High"},
            "Black Soil": {"N": "Medium", "P": "Low", "K": "High"},
            "Red Soil": {"N": "Low", "P": "Low", "K": "Medium"},
            "Laterite Soil": {"N": "Low", "P": "Low", "K": "Low"},
            "Desert Soil": {"N": "Very Low", "P": "Low", "K": "Low"}
        }
        
        return nutrient_profiles.get(soil_type, {"N": "Medium", "P": "Medium", "K": "Medium"})
    
    def _generate_recommendations(self, soil_data, weather):
        """Generate specific recommendations"""
        recommendations = soil_data["recommendations"].copy()
        
        # Add weather-based recommendations
        if weather["temperature"] > 30:
            recommendations.append("Consider heat-resistant varieties")
        if weather["humidity"] < 40:
            recommendations.append("Increase irrigation frequency")
        if weather["rainfall_24h"] > 20:
            recommendations.append("Ensure proper drainage")
        
        return recommendations
    
    def create_location_map(self, selected_states=None):
        """Create an interactive map of Indian states with soil information"""
        if selected_states is None:
            selected_states = list(self.indian_states.keys())
        
        # Prepare data for mapping
        map_data = []
        for state in selected_states:
            coords = self.indian_states[state]
            soil_info = self.soil_characteristics.get(state, {})
            
            map_data.append({
                "State": state,
                "Latitude": coords["lat"],
                "Longitude": coords["lon"],
                "Dominant_Soil": soil_info.get("dominant_soil", "Unknown"),
                "Fertility": soil_info.get("fertility", "Unknown"),
                "pH_Range": f"{soil_info.get('ph_range', [0,0])[0]}-{soil_info.get('ph_range', [0,0])[1]}",
                "Major_Crops": ", ".join(soil_info.get("major_crops", [])[:2])
            })
        
        df = pd.DataFrame(map_data)
        
        # Create interactive map
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            hover_name="State",
            hover_data=["Dominant_Soil", "Fertility", "pH_Range", "Major_Crops"],
            color="Fertility",
            color_discrete_map={"High": "green", "Medium": "orange", "Low": "red"},
            size_max=15,
            zoom=4,
            title="Indian Soil Characteristics by State"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": 20.5937, "lon": 78.9629},
            height=600
        )
        
        return fig
    
    def create_soil_comparison_chart(self, states):
        """Create comparison charts for multiple states"""
        comparison_data = []
        
        for state in states:
            soil_info = self.soil_characteristics.get(state, {})
            coords = self.indian_states[state]
            weather = self.get_weather_data(coords["lat"], coords["lon"])
            
            fertility_score = {"High": 3, "Medium": 2, "Low": 1}.get(soil_info.get("fertility", "Medium"), 2)
            ph_score = 3 if 6.0 <= soil_info.get("ph_range", [7])[0] <= 7.5 else 2
            organic_score = {"High": 3, "Medium": 2, "Low": 1}.get(soil_info.get("organic_matter", "Medium").split()[0], 2)
            
            comparison_data.append({
                "State": state,
                "Fertility": fertility_score,
                "pH_Suitability": ph_score,
                "Organic_Matter": organic_score,
                "Temperature": weather["temperature"],
                "Humidity": weather["humidity"]
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row["Fertility"], row["pH_Suitability"], row["Organic_Matter"], 
                   min(row["Temperature"]/10, 3), min(row["Humidity"]/30, 3)],
                theta=["Fertility", "pH Suitability", "Organic Matter", "Temperature", "Humidity"],
                fill='toself',
                name=row["State"]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 3])
            ),
            showlegend=True,
            title="Multi-State Soil Comparison"
        )
        
        return fig

def create_real_time_dashboard(location_analyzer, state_name):
    """Create a real-time dashboard for a specific location"""
    analysis = location_analyzer.analyze_soil_suitability(state_name)
    
    if not analysis:
        return None
    
    # Create dashboard metrics
    metrics = {
        "Overall Score": f"{analysis['suitability_analysis']['overall_score']:.1f}%",
        "Soil Health": analysis['suitability_analysis']['soil_health']['fertility_level'],
        "Current Temp": f"{analysis['current_weather']['temperature']:.1f}°C",
        "Humidity": f"{analysis['current_weather']['humidity']:.1f}%",
        "Rainfall (24h)": f"{analysis['current_weather']['rainfall_24h']:.1f}mm"
    }
    
    return {
        "metrics": metrics,
        "analysis": analysis,
        "recommendations": analysis["recommendations"]
    }