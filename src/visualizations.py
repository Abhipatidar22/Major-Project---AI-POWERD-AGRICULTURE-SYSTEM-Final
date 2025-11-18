"""
Advanced visualization components for the agriculture system
Includes interactive charts, feature importance plots, and model explanations
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def plot_feature_importance(feature_importance_data, top_n=10):
    """Create interactive feature importance plot"""
    df = pd.DataFrame(feature_importance_data).head(top_n)
    
    fig = px.bar(
        df, 
        x='importance', 
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Feature Importance', 'feature': 'Features'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig

def plot_prediction_confidence(predictions_data, crop_names):
    """Create confidence visualization for predictions"""
    fig = go.Figure(data=[
        go.Bar(
            x=crop_names,
            y=predictions_data,
            marker_color=predictions_data,
            marker_colorscale='RdYlGn',
            text=[f'{p:.1%}' for p in predictions_data],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Crop Recommendation Confidence',
        xaxis_title='Crops',
        yaxis_title='Probability',
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400
    )
    
    return fig

def plot_sensor_values_radar(sensor_data, sensor_ranges):
    """Create radar chart for sensor values"""
    categories = []
    values = []
    normalized_values = []
    
    for sensor, value in sensor_data.items():
        if sensor in sensor_ranges and isinstance(value, (int, float)):
            categories.append(sensor.replace('_', ' ').title())
            values.append(value)
            
            # Normalize to 0-1 scale
            min_val, max_val = sensor_ranges[sensor]
            normalized = (value - min_val) / (max_val - min_val)
            normalized_values.append(max(0, min(1, normalized)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Sensor Values',
        line_color='rgb(0, 150, 136)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['Min', '25%', '50%', '75%', 'Max']
            )
        ),
        showlegend=True,
        title="Sensor Values Distribution",
        height=500
    )
    
    return fig

def plot_historical_predictions(log_data):
    """Create timeline plot of historical predictions"""
    if not log_data:
        return None
    
    df = pd.DataFrame(log_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['confidence'] = [max(pred.values()) if isinstance(pred, dict) else pred for pred in df['confidence']]
    
    fig = px.line(
        df, 
        x='timestamp', 
        y='confidence',
        title='Prediction Confidence Over Time',
        labels={'confidence': 'Max Confidence', 'timestamp': 'Time'}
    )
    
    fig.add_hline(
        y=0.8, 
        line_dash="dash", 
        line_color="green",
        annotation_text="High Confidence Threshold"
    )
    
    fig.update_layout(height=300)
    
    return fig

def plot_nutrient_balance(nitrogen, phosphorus, potassium):
    """Create ternary plot for NPK balance"""
    total = nitrogen + phosphorus + potassium
    if total == 0:
        return None
    
    n_pct = (nitrogen / total) * 100
    p_pct = (phosphorus / total) * 100
    k_pct = (potassium / total) * 100
    
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [n_pct],
        'b': [p_pct], 
        'c': [k_pct],
        'marker': {
            'symbol': 'circle',
            'size': 15,
            'color': 'red'
        },
        'text': ['Current NPK'],
        'hovertemplate': 'N: %{a:.1f}%<br>P: %{b:.1f}%<br>K: %{c:.1f}%<extra></extra>'
    }))
    
    # Add optimal ranges for different crops
    optimal_zones = [
        {'name': 'Wheat Optimal', 'a': [40], 'b': [30], 'c': [30], 'color': 'blue'},
        {'name': 'Rice Optimal', 'a': [45], 'b': [25], 'c': [30], 'color': 'green'},
        {'name': 'Maize Optimal', 'a': [42], 'b': [28], 'c': [30], 'color': 'orange'}
    ]
    
    for zone in optimal_zones:
        fig.add_trace(go.Scatterternary({
            'mode': 'markers',
            'a': zone['a'],
            'b': zone['b'],
            'c': zone['c'],
            'marker': {
                'symbol': 'diamond',
                'size': 10,
                'color': zone['color']
            },
            'name': zone['name'],
            'showlegend': True
        }))
    
    fig.update_layout({
        'ternary': {
            'sum': 100,
            'aaxis': {'title': 'Nitrogen %'},
            'baxis': {'title': 'Phosphorus %'},
            'caxis': {'title': 'Potassium %'}
        },
        'title': 'NPK Balance Analysis',
        'height': 500
    })
    
    return fig

def plot_crop_suitability_matrix(predictions_proba, feature_contributions, crop_names):
    """Create heatmap showing crop suitability based on different factors"""
    # Group feature contributions by category
    categories = {
        'Nutrients': ['Nitrogen_ppm', 'Phosphorus_ppm', 'Potassium_ppm', 'NPK_ratio', 'nutrient_balance'],
        'Soil Properties': ['pH', 'Moisture_vol_pct', 'EC_dS_per_m', 'soil_Clay', 'soil_Loam', 'soil_Sandy'],
        'Climate': ['Soil_Temperature_C', 'Ambient_Humidity_pct', 'temp_humidity_index'],
        'Texture': ['HSV_mean_H', 'HSV_mean_S', 'Texture_Contrast', 'Texture_Entropy', 'texture_complexity']
    }
    
    category_scores = {}
    for category, features in categories.items():
        score = sum(contrib['contribution'] for contrib in feature_contributions 
                   if contrib['feature'] in features)
        category_scores[category] = score
    
    # Create heatmap data
    heatmap_data = []
    for crop_idx, crop in enumerate(crop_names):
        row = [predictions_proba[crop_idx] * category_scores[cat] for cat in categories.keys()]
        heatmap_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(categories.keys()),
        y=crop_names,
        colorscale='RdYlGn',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Crop Suitability by Factor Category',
        xaxis_title='Factor Categories',
        yaxis_title='Crops',
        height=400
    )
    
    return fig

def create_dashboard_metrics(sensor_data, prediction_result):
    """Create dashboard-style metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Soil pH",
            value=f"{sensor_data.get('pH', 0):.1f}",
            delta=f"{sensor_data.get('pH', 6.5) - 6.5:.1f}" if 'pH' in sensor_data else None
        )
    
    with col2:
        st.metric(
            label="Moisture %",
            value=f"{sensor_data.get('Moisture_vol_pct', 0):.1f}%",
            delta=f"{sensor_data.get('Moisture_vol_pct', 25) - 25:.1f}%" if 'Moisture_vol_pct' in sensor_data else None
        )
    
    with col3:
        st.metric(
            label="Temperature °C",
            value=f"{sensor_data.get('Soil_Temperature_C', 0):.1f}°C",
            delta=f"{sensor_data.get('Soil_Temperature_C', 20) - 20:.1f}°C" if 'Soil_Temperature_C' in sensor_data else None
        )
    
    with col4:
        if prediction_result:
            confidence = max(prediction_result) if isinstance(prediction_result, (list, np.ndarray)) else 0
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.1%}",
                delta="High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            )

def plot_seasonal_recommendations(historical_data=None):
    """Create seasonal crop recommendation calendar"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Sample seasonal data (would be replaced with actual historical data)
    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane']
    seasonal_suitability = np.random.rand(len(crops), 12)
    
    fig = go.Figure(data=go.Heatmap(
        z=seasonal_suitability,
        x=months,
        y=crops,
        colorscale='RdYlGn',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Seasonal Crop Suitability Calendar',
        xaxis_title='Months',
        yaxis_title='Crops',
        height=300
    )
    
    return fig