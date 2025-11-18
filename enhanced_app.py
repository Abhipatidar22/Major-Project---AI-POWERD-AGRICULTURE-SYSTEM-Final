"""
Enhanced AI Agriculture System with Advanced Features
Integrates new dataset, real-time processing, and advanced visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px

# Import custom modules
from src.features import moisture_proxy_from_image, load_label_encoder
from src.utils import load_pickle, load_keras_model, topk_probs, perm_importance_table, try_import_tf
from src.sensor_processor import SensorDataProcessor, DataLogger, simulate_sensor_reading
from src.visualizations import (
    plot_feature_importance, plot_prediction_confidence, plot_sensor_values_radar,
    plot_historical_predictions, plot_nutrient_balance, create_dashboard_metrics,
    plot_crop_suitability_matrix, plot_seasonal_recommendations
)
from src.location_analysis import IndianLocationAnalyzer, create_real_time_dashboard
try:
    from src.enhanced_crop_trainer import EnhancedCropRecommendationModel
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False
    print("Enhanced models not available. Using basic models.")

# Initialize paths
MODELS = Path('models')
DATA = Path('data') / 'synthetic'

# Initialize processors
sensor_processor = SensorDataProcessor()
data_logger = DataLogger()
location_analyzer = IndianLocationAnalyzer()

# Simple wrapper class for loaded enhanced crop model
class LoadedEnhancedCropModel:
    def __init__(self, model_data):
        # Defensive loading: model_data may have different shapes depending on how it was saved.
        # Use .get and try multiple fallbacks to avoid KeyError (which previously caused 'classes' error).
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler')
        self.label_encoder = model_data.get('label_encoder')
        self.feature_names = model_data.get('feature_names', [])

        # crop_varieties may be under several possible keys
        self.crop_varieties = model_data.get('crop_varieties') or model_data.get('varieties') or []

        # Resolve class labels safely:
        classes = model_data.get('classes')
        if not classes and self.label_encoder:
            # label_encoder may be a dict with 'classes' or a sklearn LabelEncoder-like object
            if isinstance(self.label_encoder, dict) and 'classes' in self.label_encoder:
                classes = self.label_encoder['classes']
            else:
                try:
                    classes = list(getattr(self.label_encoder, 'classes_', []))
                except Exception:
                    classes = None

        # Fallback to crop_varieties list if present
        if not classes and self.crop_varieties:
            classes = self.crop_varieties

        # Final fallback: if model exposes classes_ (e.g., sklearn classifier)
        if not classes and hasattr(self.model, 'classes_'):
            try:
                classes = list(getattr(self.model, 'classes_', []))
            except Exception:
                classes = []

        self.classes = list(classes) if classes else []
        # store any warnings about missing pieces
        self._load_warnings = []
        if not self.classes:
            self._load_warnings.append('No class labels found in model_data (checked keys: classes, label_encoder, crop_varieties, model.classes_)')
        if not self.scaler:
            self._load_warnings.append('No scaler found in model_data')
        
    def predict_with_variety(self, features, top_k=8):
        """Predict crop varieties with confidence and suitability scoring"""
        # Scale features
        if self.scaler is None:
            raise RuntimeError('Enhanced model scaler is missing; cannot scale features.')
        features_scaled = self.scaler.transform([features])
        
        # Get predictions
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Guard against missing class labels
            try:
                crop_name = self.classes[idx]
            except Exception:
                crop_name = f'Unknown_{idx}'
            confidence = probabilities[idx]
            
            # Parse crop and variety from full name
            if '_' in crop_name:
                crop, variety = crop_name.split('_', 1)
            else:
                crop, variety = crop_name, 'Standard'
            
            # Calculate suitability score (simple version)
            suitability_score = min(1.0, confidence * 1.2)  # Boost confidence for suitability
            
            results.append({
                'crop': crop,
                'variety': variety,
                'full_name': crop_name,
                'confidence': confidence,
                'suitability_score': suitability_score
            })
        
        return results
    
    def get_seasonal_recommendations(self, season):
        """Get seasonal crop recommendations"""
        seasonal_crops = {}
        for crop_name in (self.classes or []):
            # Simple seasonal logic - can be enhanced later
            if season.lower() in ['kharif', 'monsoon']:
                if any(keyword in crop_name.lower() for keyword in ['rice', 'cotton', 'kharif']):
                    seasonal_crops[crop_name] = {'suitability': 0.8}
            elif season.lower() in ['rabi', 'winter']:
                if any(keyword in crop_name.lower() for keyword in ['wheat', 'rabi']):
                    seasonal_crops[crop_name] = {'suitability': 0.8}
            elif season.lower() in ['zaid', 'summer']:
                if any(keyword in crop_name.lower() for keyword in ['spring']):
                    seasonal_crops[crop_name] = {'suitability': 0.8}
        
        return seasonal_crops

# Initialize enhanced models if available
enhanced_crop_model = None
enhanced_model_status = "Not Available"
enhanced_model_error = None

if ENHANCED_MODELS_AVAILABLE:
    try:
        enhanced_model_path = MODELS / 'enhanced_crop_model.pkl'
        if enhanced_model_path.exists():
            import joblib
            model_data = joblib.load(enhanced_model_path)
            enhanced_crop_model = LoadedEnhancedCropModel(model_data)
            # If loader recorded warnings (e.g., missing classes/scaler), surface them but keep model
            if getattr(enhanced_crop_model, '_load_warnings', None):
                enhanced_model_status = "Loaded with Warnings"
                enhanced_model_error = '; '.join(enhanced_crop_model._load_warnings)
            else:
                enhanced_model_status = "Loaded Successfully"
        else:
            enhanced_model_status = "Model file not found"
    except Exception as e:
        enhanced_model_status = "Loading Error"
        enhanced_model_error = str(e)

# Page configuration
st.set_page_config(
    page_title='🌾 Smart Agriculture System | AI-Powered Crop Recommendations', 
    page_icon='🌾',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '## Smart Agriculture System\nAI-powered crop recommendations with real-time analysis for Indian farmers. Get personalized suggestions for crops, varieties, and farming practices based on your soil and environmental conditions.'
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #388E3C;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Welcome section with better introduction
st.markdown('<h1 class="main-header">🌾 Smart Agriculture System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">🤖 AI-Powered Crop Recommendations | 📍 Real-time Location Analysis for India</p>', unsafe_allow_html=True)

# Quick start guide
with st.expander("ℹ️ How to Use This System - Quick Start Guide", expanded=False):
    st.markdown("""
    ### 🚀 Get Started in 3 Simple Steps:
    
    1. **📍 Choose Your Location**: Select your state and region for location-specific recommendations
    2. **📊 Input Your Data**: Choose from:
       - 📝 **Manual Entry**: Enter your soil and environmental data
       - 🎲 **Smart Simulation**: Let AI generate realistic data for your location
       - 📁 **CSV Upload**: Upload your own dataset
    3. **🌾 Get Recommendations**: Receive AI-powered crop variety suggestions with confidence scores
    
    ### 🎯 What You'll Get:
    - **Variety-Specific Recommendations**: Not just "Rice" but "Basmati Rice", "Jasmine Rice", etc.
    - **Confidence Scores**: See how sure the AI is about each recommendation
    - **Environmental Suitability**: Know if conditions are optimal for each crop
    - **Seasonal Guidance**: Best crops for Kharif, Rabi, and Zaid seasons
    
    ### 🔬 Available Features:
    - 🌾 **14 Crop Varieties**: Rice, Wheat, Cotton with multiple varieties
    - 🗺️ **15 Indian States**: Location-specific soil and climate data
    - 🧪 **Soil Analysis**: pH, nutrients, moisture, and more
    - 📈 **Yield Prediction**: Expected harvest quantities
    - 🦠 **Disease Detection**: Plant health monitoring
    """)

st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">🤖 AI-Powered Crop Recommendations | 📍 Real-time Location Analysis for India</p>', unsafe_allow_html=True)



# Sidebar for system status and settings
with st.sidebar:
    st.markdown("### 🏠 Navigation Hub")
    st.markdown("**Welcome!** Use this panel to monitor system status and adjust settings.")
    
    st.header("🔧 System Status")
    st.markdown("*Current model availability and performance*")
    
    # Check model availability
    enhanced_model_available = enhanced_crop_model is not None
    advanced_model_available = (MODELS / 'advanced_crop_model.pkl').exists()
    basic_model_available = (MODELS / 'crop_rf.pkl').exists()
    
    if enhanced_model_available:
        st.success("✅ Enhanced Variety Model Loaded")
        st.info(f"🌾 {len(enhanced_crop_model.classes)} crop varieties available")
    elif enhanced_model_status == "Loading Error":
        st.error(f"❌ Enhanced Model Error: {enhanced_model_error}")
        if advanced_model_available:
            st.warning("⚠️ Falling back to Advanced Model")
        elif basic_model_available:
            st.warning("⚠️ Falling back to Basic Model")
        else:
            st.error("❌ No Backup Models Available")
    elif enhanced_model_status == "Model file not found":
        st.warning("⚠️ Enhanced model not found")
        if advanced_model_available:
            st.success("✅ Advanced Model Available")
        elif basic_model_available:
            st.warning("⚠️ Basic Model Available")
        else:
            st.error("❌ No Models Available")
    elif advanced_model_available:
        st.success("✅ Advanced Model Loaded")
    elif basic_model_available:
        st.warning("⚠️ Using Basic Model")
    else:
        st.error("❌ No Models Available")
    
    st.header("⚙️ Settings")
    st.markdown("*Customize your experience*")
    
    with st.expander("🎛️ Advanced Settings", expanded=True):
        real_time_mode = st.checkbox(
            "🔄 Real-time Sensor Mode", 
            value=False,
            help="Enable this to simulate real-time sensor data updates. Perfect for demonstration purposes!"
        )
        show_explanations = st.checkbox(
            "📖 Show AI Explanations", 
            value=True,
            help="Display detailed explanations about how the AI makes its recommendations. Great for learning!"
        )
        log_predictions = st.checkbox(
            "📝 Log Predictions", 
            value=True,
            help="Save prediction history for analysis and tracking performance over time."
        )
    
    st.header("📊 Quick Stats")
    recent_predictions = data_logger.get_recent_predictions(5)
    st.write(f"Recent predictions: {len(recent_predictions)}")
    
    # Debug information (can be removed later)
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"Enhanced Models Available: {ENHANCED_MODELS_AVAILABLE}")
        st.write(f"Enhanced Model Status: {enhanced_model_status}")
        st.write(f"Enhanced Model Object: {enhanced_crop_model is not None}")
        st.write(f"Models Path: {MODELS}")
        st.write(f"Enhanced Model File Exists: {(MODELS / 'enhanced_crop_model.pkl').exists()}")
        if enhanced_model_error:
            st.write(f"Error: {enhanced_model_error}")
    
    if recent_predictions:
        try:
            # Extract confidence values with proper type handling
            confidence_values = []
            for pred in recent_predictions:
                conf = pred.get('confidence', [0])
                if isinstance(conf, list) and conf:
                    confidence_values.append(float(max(conf)))
                elif isinstance(conf, (int, float)):
                    confidence_values.append(float(conf))
                else:
                    confidence_values.append(0.0)
            
            if confidence_values:
                avg_confidence = np.mean(confidence_values, dtype=np.float64)
                st.write(f"Avg confidence: {avg_confidence:.1%}")
        except Exception as e:
            st.write("Avg confidence: N/A")

# Main content tabs with better descriptions
st.markdown("### 🎯 Choose Your Analysis Type")
st.markdown("Select the tab below for the type of agricultural analysis you need:")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '� Crop Recommendations', 
    '🔬 Soil Analysis', 
    '📈 Yield Prediction', 
    '🍃 Disease Detection',
    '📍 Location Analysis',
    '📊 Dashboard'
])

# Tab 1: Advanced Crop Recommendation
with tab1:
    st.markdown('<h2 class="section-header">🌾 Smart Crop Recommendation System</h2>', unsafe_allow_html=True)
    
    # Add helpful context
    st.info("""💡 **Pro Tip**: This system recommends specific crop varieties (not just crop types) based on your 
    environmental conditions. Get personalized suggestions with confidence scores!""")
    
    # Success stories or testimonials could go here
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("🎯 Crop Varieties", "14+", help="Including Basmati Rice, Durum Wheat, Bt Cotton, etc.")
    with col_info2:
        st.metric("📊 Accuracy", "60.4%", help="Model accuracy across all crop varieties")
    with col_info3:
        st.metric("🗺️ Coverage", "15 States", help="Covers major agricultural states in India")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("� How Do You Want to Provide Your Data?")
        
        st.markdown("**Select the method that works best for you:**")
        
        # Input method selection with descriptions
        method_options = [
            "📝 Manual Entry - I'll enter my soil and environmental data manually",
            "🎲 Smart Simulation - Generate realistic data for my location (Recommended for demo)", 
            "📁 CSV Upload - I have my own data file to upload"
        ]
        
        selected_option = st.selectbox(
            "Choose Your Data Input Method:",
            options=method_options,
            index=1,
            help="Select how you want to provide your agricultural data"
        )
        
        # Map back to original method names for compatibility
        if "Manual Entry" in selected_option:
            input_method = "Manual Entry"
        elif "Smart Simulation" in selected_option:
            input_method = "Real-time Simulation"
        else:
            input_method = "CSV Upload"
        
        if input_method == "Manual Entry":
            # Manual sensor input with better organization
            st.markdown("### 📋 Enter Your Agricultural Data")
            st.markdown("*Provide accurate measurements for better crop recommendations*")
            
            with st.expander("🌡️ Environmental Conditions", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    moisture = st.slider(
                        "💧 Soil Moisture (%)", 
                        0.0, 50.0, 25.0, 0.1,
                        help="Percentage of water content in soil. Optimal range: 20-30% for most crops"
                    )
                    pH = st.slider(
                        "⚗️ Soil pH", 
                        3.0, 10.0, 6.5, 0.1,
                        help="Soil acidity level. Most crops prefer pH 6.0-7.5 (slightly acidic to neutral)"
                    )
                    temp = st.slider(
                        "🌡️ Soil Temperature (°C)", 
                        -5.0, 45.0, 25.0, 0.1,
                        help="Current soil temperature. Affects seed germination and root growth"
                    )
                    humidity = st.slider(
                        "💨 Ambient Humidity (%)", 
                        0.0, 100.0, 60.0, 1.0,
                        help="Air humidity percentage. Affects plant transpiration and disease risk"
                    )
                
                with col_b:
                    nitrogen = st.number_input(
                        "🌿 Nitrogen (ppm)", 
                        0, 800, 200, 10,
                        help="Essential for leaf growth and chlorophyll production. High nitrogen = greener plants"
                    )
                    phosphorus = st.number_input(
                        "🌸 Phosphorus (ppm)", 
                        0, 200, 60, 5,
                        help="Important for root development, flowering, and fruiting. Critical during early growth"
                    )
                    potassium = st.number_input(
                        "🍃 Potassium (ppm)", 
                        0, 500, 150, 10,
                        help="Enhances disease resistance, water regulation, and fruit quality"
                    )
                    ec = st.slider(
                        "⚡ Electrical Conductivity (dS/m)", 
                        0.0, 5.0, 1.5, 0.1,
                        help="Measures soil salinity. High EC can stress plants and reduce yields"
                    )
            
            with st.expander("🎨 Visual & Texture Properties"):
                col_c, col_d = st.columns(2)
                with col_c:
                    hsv_h = st.slider("HSV Hue", 0.0, 360.0, 30.0, 1.0)
                    hsv_s = st.slider("HSV Saturation", 0.0, 255.0, 80.0, 1.0)
                
                with col_d:
                    texture_contrast = st.slider("Texture Contrast", 0.0, 1.0, 0.5, 0.01)
                    texture_entropy = st.slider("Texture Entropy", 0.0, 2.0, 0.8, 0.01)
            
            soil_type = st.selectbox("Soil Type", ["Clay", "Loam", "Sandy"])
            
            sensor_data = {
                'Moisture_vol_pct': moisture,
                'pH': pH,
                'Nitrogen_ppm': nitrogen,
                'Phosphorus_ppm': phosphorus,
                'Potassium_ppm': potassium,
                'Soil_Temperature_C': temp,
                'Ambient_Humidity_pct': humidity,
                'EC_dS_per_m': ec,
                'HSV_mean_H': hsv_h,
                'HSV_mean_S': hsv_s,
                'Texture_Contrast': texture_contrast,
                'Texture_Entropy': texture_entropy
            }
        
        elif input_method == "Real-time Simulation":
            st.markdown("### 🎲 Smart Data Simulation")
            st.info("🔄 **Perfect for Demo**: AI will generate realistic agricultural data based on Indian farming conditions!")
            
            col_sim1, col_sim2 = st.columns([2, 1])
            with col_sim1:
                if st.button("🎲 Generate New Reading", type="primary"):
                    sensor_data, soil_type = simulate_sensor_reading()
                    st.success("✨ New sensor reading generated! Check the values below.")
                else:
                    sensor_data, soil_type = simulate_sensor_reading()
            
            with col_sim2:
                st.markdown("**🎯 What's simulated:**")
                st.markdown("- Soil conditions")  
                st.markdown("- Weather data")
                st.markdown("- Nutrient levels")
                st.markdown("- Environmental factors")
            
            # Display simulated values
            st.subheader("📊 Current Sensor Readings")
            create_dashboard_metrics(sensor_data, None)
        
        else:  # CSV Upload
            st.markdown("### 📁 Upload Your Data File")
            st.markdown("Upload a CSV file with your agricultural measurements for batch analysis.")
            
            # Add example of expected format
            with st.expander("📋 Expected CSV Format", expanded=False):
                st.markdown("""
                **Required columns (examples):**
                - Moisture_vol_pct: 25.0
                - pH: 6.5  
                - Nitrogen_ppm: 200
                - Phosphorus_ppm: 60
                - Potassium_ppm: 150
                - Soil_Temperature_C: 25.0
                - Ambient_Humidity_pct: 60.0
                
                *The system will use the first row of data for predictions.*
                """)
            
            uploaded_file = st.file_uploader(
                "📂 Choose CSV File", 
                type=['csv'],
                help="Upload a CSV file containing your soil and environmental measurements"
            )
            
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Found {len(df)} rows of data.")
                
                with st.expander("👀 Data Preview", expanded=True):
                    st.dataframe(df.head())
                
                # Use first row for prediction
                if len(df) > 0:
                    row = df.iloc[0]
                    sensor_data = row.to_dict()
                    soil_type = sensor_data.get('Soil_Type', 'Clay')
    
    with col2:
        st.subheader("🎯 Get Your Recommendation")
        
        st.markdown("""
        **Ready for analysis?** 
        Click below to get AI-powered crop variety recommendations based on your data.
        """)
        
        # Make the button more prominent
        if st.button("🚀 Get Crop Recommendation", type="primary", use_container_width=True):
            try:
                # Process sensor data
                processed_data, validation_results, warnings = sensor_processor.process_real_time_input(
                    sensor_data, soil_type
                )
                
                # Display warnings if any
                if warnings:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning("⚠️ Data Quality Warnings:")
                    for warning in warnings:
                        st.write(f"• {warning}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Make prediction
                prediction_made = False
                
                # Try enhanced variety-based model first
                if enhanced_crop_model is not None:
                    try:
                        # Prepare features for enhanced model
                        enhanced_features = [
                            processed_data.get('Soil_Temperature_C', 25),  # Temperature
                            processed_data.get('Ambient_Humidity_pct', 65),  # Humidity  
                            processed_data.get('pH', 6.5),  # pH
                            processed_data.get('Nitrogen_ppm', 100),  # Nitrogen
                            processed_data.get('Phosphorus_ppm', 50),  # Phosphorus
                            processed_data.get('Potassium_ppm', 150),  # Potassium
                            100,  # Rainfall (default)
                            processed_data.get('Moisture_vol_pct', 25),  # Moisture
                            processed_data.get('EC_dS_per_m', 1.5),  # EC
                            20, 77,  # Latitude, Longitude (default for India)
                            12, 22, 12,  # Wind speed, Solar radiation, Day length
                            1, 0, 0  # Season encoding (Kharif default)
                        ]
                        
                        # Get variety-based predictions
                        variety_predictions = enhanced_crop_model.predict_with_variety(enhanced_features, top_k=8)
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("🌾 Enhanced Variety-Based Recommendations:")
                        
                        # Display top predictions with varieties
                        for i, pred in enumerate(variety_predictions, 1):
                            confidence_bar = "▓" * int(pred['confidence'] * 20) + "░" * (20 - int(pred['confidence'] * 20))
                            suitability_color = "🟢" if pred['suitability_score'] > 0.8 else "🟡" if pred['suitability_score'] > 0.6 else "🔴"
                            
                            st.markdown(f"""
                            **{i}. {pred['full_name']}** {suitability_color}
                            - Confidence: {pred['confidence']:.1%} `{confidence_bar}`
                            - Suitability: {pred['suitability_score']:.1%}
                            - Type: {pred['crop']} → {pred['variety']}
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Enhanced visualizations
                        st.subheader("📊 Variety Analysis")
                        
                        # Create variety confidence chart
                        variety_names = [pred['full_name'] for pred in variety_predictions]
                        variety_confidences = [pred['confidence'] for pred in variety_predictions]
                        variety_suitability = [pred['suitability_score'] for pred in variety_predictions]
                        
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Prediction Confidence', 'Environmental Suitability'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # Confidence chart
                        fig.add_trace(
                            go.Bar(
                                x=variety_confidences,
                                y=variety_names,
                                orientation='h',
                                name='Confidence',
                                marker_color='lightblue'
                            ),
                            row=1, col=1
                        )
                        
                        # Suitability chart
                        fig.add_trace(
                            go.Bar(
                                x=variety_suitability,
                                y=variety_names,
                                orientation='h',
                                name='Suitability',
                                marker_color='lightgreen'
                            ),
                            row=1, col=2
                        )
                        
                        fig.update_layout(
                            title="Crop Variety Analysis",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Seasonal recommendations
                        if show_explanations:
                            st.subheader("🗓️ Seasonal Recommendations")
                            
                            for season in ['Kharif', 'Rabi', 'Zaid']:
                                seasonal_crops = enhanced_crop_model.get_seasonal_recommendations(season)
                                
                                if seasonal_crops:
                                    st.write(f"**{season} Season:**")
                                    season_text = []
                                    for crop, info in list(seasonal_crops.items())[:5]:
                                        season_text.append(f"• {crop} (Suitability: {info['suitability']:.1%})")
                                    st.write("\n".join(season_text))
                        
                        prediction_made = True
                        
                        # Log prediction
                        if log_predictions:
                            top_prediction = variety_predictions[0]
                            data_logger.log_prediction(
                                processed_data, 
                                top_prediction['full_name'], 
                                {pred['full_name']: pred['confidence'] for pred in variety_predictions[:3]}
                            )
                    
                    except Exception as e:
                        st.error(f"Enhanced model error: {str(e)}")
                        st.info("Falling back to basic models...")
                
                # Fallback to advanced model
                if not prediction_made and advanced_model_available:
                    try:
                        from src.advanced_crop_model import predict_with_explanation
                        model_data = load_pickle(MODELS / 'advanced_crop_model.pkl')
                        crop_labels = load_label_encoder(MODELS / 'advanced_crops_labels.json')
                        
                        # Prepare features in correct order
                        feature_values = [processed_data.get(f, 0) for f in model_data['features']]
                        
                        proba, feature_contributions = predict_with_explanation(model_data, feature_values)
                        
                        # Display results
                        top_predictions = sorted(
                            [(crop_labels['classes'][i], proba[i]) for i in range(len(proba))],
                            key=lambda x: x[1], reverse=True
                        )[:3]
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("🎯 Advanced Model Prediction:")
                        for i, (crop, confidence) in enumerate(top_predictions):
                            st.write(f"{i+1}. **{crop}**: {confidence:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualizations
                        st.subheader("📈 Prediction Analysis")
                        
                        # Confidence chart
                        fig_conf = plot_prediction_confidence(proba, crop_labels['classes'])
                        st.plotly_chart(fig_conf, use_container_width=True)
                        
                        if show_explanations:
                            # Feature importance
                            fig_importance = plot_feature_importance(feature_contributions[:10])
                            st.plotly_chart(fig_importance, use_container_width=True)
                            
                            # Suitability matrix
                            fig_matrix = plot_crop_suitability_matrix(
                                proba, feature_contributions, crop_labels['classes']
                            )
                            st.plotly_chart(fig_matrix, use_container_width=True)
                        
                        prediction_made = True
                        
                        # Log prediction
                        if log_predictions:
                            data_logger.log_prediction(
                                processed_data, top_predictions[0][0], 
                                {crop: conf for crop, conf in top_predictions}
                            )
                    
                    except Exception as e:
                        st.error(f"Advanced model error: {str(e)}")
                
                # Fallback to basic model
                if not prediction_made and basic_model_available:
                    try:
                        bundle = load_pickle(MODELS / 'crop_rf.pkl')
                        model, features = bundle['model'], bundle['features']
                        crops_le = load_label_encoder(MODELS / 'crops_labels.json')
                        
                        # Map processed data to basic model features
                        basic_features = {
                            'N': processed_data.get('Nitrogen_ppm', 100),
                            'P': processed_data.get('Phosphorus_ppm', 50),
                            'K': processed_data.get('Potassium_ppm', 150),
                            'pH': processed_data.get('pH', 6.5),
                            'temperature': processed_data.get('Soil_Temperature_C', 25),
                            'humidity': processed_data.get('Ambient_Humidity_pct', 60),
                            'rainfall': 100,  # Default value
                            'soil_sandy': processed_data.get('soil_Sandy', 0),
                            'soil_clay': processed_data.get('soil_Clay', 1),
                            'soil_loam': processed_data.get('soil_Loam', 0),
                        }
                        
                        X = pd.DataFrame([[basic_features.get(f, 0) for f in features]], columns=features)
                        proba = model.predict_proba(X)[0]
                        
                        st.success('🎯 Basic Model Predictions:')
                        top_3 = topk_probs(proba, crops_le['classes'], k=3)
                        st.table(pd.DataFrame(top_3, columns=['Crop', 'Probability']))
                        
                        prediction_made = True
                    
                    except Exception as e:
                        st.error(f"Basic model error: {str(e)}")
                
                if not prediction_made:
                    if not enhanced_crop_model and not advanced_model_available and not basic_model_available:
                        st.error("❌ No models available for prediction")
                    else:
                        st.warning("⚠️ Prediction failed. Please check input data.")
            
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Visualization section moved to main tab1 block

# Tab 2: Enhanced Soil Analysis (keeping original functionality)
with tab2:
    st.markdown('<h2 class="section-header">🌱 Soil Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader('Upload soil image (jpg/png)', type=['jpg','jpeg','png'])
        use_sample = st.checkbox('Use a sample image', value=True)
        if use_sample:
            samples = sorted((DATA/'soil_images').glob('*.png'))
            if samples:
                file = samples[0].open('rb')
                st.caption(f'Using sample: {samples[0].name}')
    
    with col2:
        if file:
            img = Image.open(file).convert('RGB')
            st.image(img, caption='Input soil image', use_column_width=True)
            moist = moisture_proxy_from_image(img)
            st.caption(f'Moisture proxy (brightness-based): **{moist:.2f}**')
            
            model_path = MODELS / 'soil_cnn.keras'
            if model_path.exists() and try_import_tf():
                try:
                    model = load_keras_model(model_path)
                    le = load_label_encoder(MODELS / 'soil_labels.json')
                    x = np.array(img.resize((64,64))) / 255.0
                    x = np.expand_dims(x, 0)
                    pred = model.predict(x, verbose=0)[0]
                    idx = int(np.argmax(pred))
                    name = le['classes'][idx]
                    
                    st.success(f'Predicted soil: **{name}**')
                    
                    # Enhanced visualization
                    fig = plot_prediction_confidence(pred, le['classes'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning('Soil CNN unavailable. Install TensorFlow and re-run prepare_demo.py.')
                    st.exception(e)
            else:
                st.info('Soil CNN not installed/trained — only moisture proxy shown.')

# Tab 3: Enhanced Yield Prediction (keeping original functionality)
with tab3:
    st.markdown('<h2 class="section-header">🌻 Yield Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox('Crop', ['wheat','rice','maize'])
        region = st.selectbox('Region', ['region_1','region_2','region_3'])
        soil = st.selectbox('Soil', ['sandy','clay','loam'])
    
    with col2:
        temp = st.number_input('Avg Temp (°C)', -5.0, 55.0, 24.0, 0.1)
        hum = st.number_input('Avg Humidity (%)', 0.0, 100.0, 55.0, 1.0)
        rain = st.number_input('Avg Rainfall (mm)', 0.0, 400.0, 100.0, 1.0)
        N = st.number_input('N', 0.0, 200.0, 80.0, 1.0)
        P = st.number_input('P', 0.0, 200.0, 40.0, 1.0)
        K = st.number_input('K', 0.0, 200.0, 40.0, 1.0)
    
    if st.button('Predict yield range'):
        try:
            q10 = load_pickle(MODELS / 'yield_q10.pkl')
            q50 = load_pickle(MODELS / 'yield_q50.pkl')
            q90 = load_pickle(MODELS / 'yield_q90.pkl')
            features = q50['features']
            
            row = {
                'temperature': temp, 'humidity': hum, 'rainfall': rain,
                'N': N, 'P': P, 'K': K,
                'soil_sandy': int(soil=='sandy'),
                'soil_clay': int(soil=='clay'),
                'soil_loam': int(soil=='loam'),
                'crop_wheat': int(crop=='wheat'),
                'crop_rice': int(crop=='rice'),
                'crop_maize': int(crop=='maize'),
                'region_region_1': int(region=='region_1'),
                'region_region_2': int(region=='region_2'),
                'region_region_3': int(region=='region_3'),
            }
            
            X = pd.DataFrame([[row.get(f, 0) for f in features]], columns=features)
            y10 = float(q10['model'].predict(X)[0])
            y50 = float(q50['model'].predict(X)[0])
            y90 = float(q90['model'].predict(X)[0])
            
            st.success(f'Predicted yield range: **{y10:.2f} – {y90:.2f}** (median {y50:.2f})')
            
            # Enhanced visualization
            fig = px.bar(
                x=['Lower Bound (10%)', 'Median (50%)', 'Upper Bound (90%)'],
                y=[y10, y50, y90],
                title='Yield Prediction Range',
                labels={'x': 'Quantile', 'y': 'Yield (tons/hectare)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning('Yield models not found. Run `python src/prepare_demo.py`.')
        except Exception as e:
            st.exception(e)

# Tab 4: Enhanced Disease Detection (keeping original functionality)
with tab4:
    st.markdown('<h2 class="section-header">🍃 Leaf Disease Detection</h2>', unsafe_allow_html=True)
    
    file = st.file_uploader('Upload leaf image (jpg/png)', type=['jpg','jpeg','png'], key='leaf')
    if st.button('Detect disease'):
        if not file:
            st.warning('Please upload a leaf image.')
        else:
            img = Image.open(file).convert('RGB')
            st.image(img, caption='Input leaf', use_column_width=True)
            
            model_path = MODELS / 'leaf_cnn.keras'
            if model_path.exists() and try_import_tf():
                try:
                    model = load_keras_model(model_path)
                    le = load_label_encoder(MODELS / 'leaf_labels.json')
                    x = np.array(img.resize((64,64))) / 255.0
                    x = np.expand_dims(x, 0)
                    pred = model.predict(x, verbose=0)[0]
                    idx = int(np.argmax(pred))
                    name = le['classes'][idx]
                    
                    st.success(f'Predicted: **{name}**')
                    
                    # Enhanced visualization
                    fig = plot_prediction_confidence(pred, le['classes'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning('Leaf CNN unavailable. Install TensorFlow and re-run prepare_demo.py.')
                    st.exception(e)
            else:
                st.info('Leaf CNN not installed/trained.')

# Tab 5: Location-based Analysis for India
with tab5:
    st.markdown('<h2 class="section-header">📍 Real-time Location-based Soil Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🗺️ Select Location")
        
        # State selection
        selected_state = st.selectbox(
            "Choose Indian State:",
            list(location_analyzer.indian_states.keys()),
            index=0
        )
        
        # Multiple state comparison
        st.subheader("🔍 Multi-State Comparison")
        comparison_states = st.multiselect(
            "Select states to compare:",
            list(location_analyzer.indian_states.keys()),
            default=[selected_state]
        )
        
        # Real-time update settings
        st.subheader("⚙️ Real-time Settings")
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
        
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.rerun()
    
    with col2:
        # Real-time dashboard for selected state
        st.subheader(f"🌍 Real-time Analysis: {selected_state}")
        
        dashboard = create_real_time_dashboard(location_analyzer, selected_state)
        
        if dashboard:
            # Display metrics
            metrics = dashboard["metrics"]
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                st.metric("Overall Score", metrics["Overall Score"])
            with col_m2:
                st.metric("Soil Health", metrics["Soil Health"])
            with col_m3:
                st.metric("Temperature", metrics["Current Temp"])
            with col_m4:
                st.metric("Humidity", metrics["Humidity"])
            with col_m5:
                st.metric("Rainfall", metrics["Rainfall (24h)"])
            
            # Detailed analysis
            analysis = dashboard["analysis"]
            
            # Soil characteristics
            with st.expander("🌱 Detailed Soil Characteristics", expanded=True):
                soil_data = analysis["soil_characteristics"]
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.write(f"**Dominant Soil:** {soil_data['dominant_soil']}")
                    st.write(f"**Secondary Soil:** {soil_data['secondary_soil']}")
                    st.write(f"**pH Range:** {soil_data['ph_range'][0]:.1f} - {soil_data['ph_range'][1]:.1f}")
                    st.write(f"**Organic Matter:** {soil_data['organic_matter']}")
                
                with col_s2:
                    st.write(f"**Drainage:** {soil_data['drainage']}")
                    st.write(f"**Fertility:** {soil_data['fertility']}")
                    st.write(f"**Major Crops:** {', '.join(soil_data['major_crops'])}")
            
            # Current weather conditions
            with st.expander("🌦️ Current Weather Conditions"):
                weather = analysis["current_weather"]
                
                col_w1, col_w2, col_w3 = st.columns(3)
                with col_w1:
                    st.metric("Temperature", f"{weather['temperature']:.1f}°C")
                    st.metric("Humidity", f"{weather['humidity']:.1f}%")
                
                with col_w2:
                    st.metric("Rainfall (24h)", f"{weather['rainfall_24h']:.1f}mm")
                    st.metric("Wind Speed", f"{weather['wind_speed']:.1f}km/h")
                
                with col_w3:
                    st.metric("Pressure", f"{weather['pressure']:.0f}hPa")
                    st.caption(f"Last updated: {weather['last_updated'][:19]}")
            
            # Recommendations
            with st.expander("💡 AI-Generated Recommendations"):
                recommendations = dashboard["recommendations"]
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            
            # Challenges and solutions
            with st.expander("⚠️ Regional Challenges & Solutions"):
                st.write("**Common Challenges:**")
                for challenge in soil_data["challenges"]:
                    st.write(f"• {challenge}")
    
    # Interactive map
    st.subheader("🗺️ Interactive Soil Map of India")
    
    # Map options
    col_map1, col_map2 = st.columns([3, 1])
    
    with col_map2:
        show_all_states = st.checkbox("Show all states", value=True)
        map_states = list(location_analyzer.indian_states.keys()) if show_all_states else comparison_states
        
        filter_by_fertility = st.selectbox("Filter by fertility:", ["All", "High", "Medium", "Low"])
        
        if filter_by_fertility != "All":
            map_states = [
                state for state in map_states 
                if location_analyzer.soil_characteristics.get(state, {}).get("fertility") == filter_by_fertility
            ]
    
    with col_map1:
        if map_states:
            fig_map = location_analyzer.create_location_map(map_states)
            st.plotly_chart(fig_map, use_container_width=True)
    
    # Multi-state comparison
    if len(comparison_states) > 1:
        st.subheader("📊 Multi-State Soil Comparison")
        
        # Comparison radar chart
        fig_comparison = location_analyzer.create_soil_comparison_chart(comparison_states)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Comparison table
        comparison_data = []
        for state in comparison_states:
            soil_info = location_analyzer.soil_characteristics.get(state, {})
            comparison_data.append({
                "State": state,
                "Dominant Soil": soil_info.get("dominant_soil", "Unknown"),
                "Fertility": soil_info.get("fertility", "Unknown"),
                "pH Range": f"{soil_info.get('ph_range', [0,0])[0]}-{soil_info.get('ph_range', [0,0])[1]}",
                "Drainage": soil_info.get("drainage", "Unknown"),
                "Major Crops": ", ".join(soil_info.get("major_crops", [])[:3])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

# Tab 6: Analytics Dashboard
with tab6:
    st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Prediction History")
        recent_logs = data_logger.get_recent_predictions(20)
        
        if recent_logs:
            fig_history = plot_historical_predictions(recent_logs)
            if fig_history:
                st.plotly_chart(fig_history, use_container_width=True)
            
            # Recent predictions table
            df_recent = pd.DataFrame([
                {
                    'Timestamp': log['timestamp'][:19],
                    'Prediction': log['prediction'],
                    'Confidence': f"{max(log['confidence'].values()) if isinstance(log['confidence'], dict) else log['confidence']:.1%}"
                }
                for log in recent_logs[-10:]
            ])
            st.dataframe(df_recent)
        else:
            st.info("No prediction history available yet.")
    
    with col2:
        st.subheader("🗓️ Seasonal Recommendations")
        fig_seasonal = plot_seasonal_recommendations()
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        st.subheader("🔧 System Information")
        st.write(f"**Models Available:** {len(list(MODELS.glob('*.pkl')))}")
        st.write(f"**Data Points Logged:** {len(recent_logs)}")
        st.write(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🌾 Advanced AI-Powered Agriculture System | Enhanced with Real-time Processing & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)