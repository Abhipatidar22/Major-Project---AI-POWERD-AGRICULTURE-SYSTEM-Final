#!/usr/bin/env python3
"""
Direct Enhanced Crop Model Training
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def create_enhanced_crop_dataset():
    """Create enhanced dataset with crop varieties"""
    
    # Enhanced crop data with varieties
    crops_data = []
    
    # Rice varieties
    rice_varieties = {
        'Basmati': {'temp': [20, 35], 'humidity': [70, 85], 'ph': [5.5, 6.8], 'season': 'Kharif'},
        'Jasmine': {'temp': [22, 32], 'humidity': [65, 80], 'ph': [6.0, 7.0], 'season': 'Kharif'},
        'Arborio': {'temp': [18, 28], 'humidity': [60, 75], 'ph': [6.2, 7.2], 'season': 'Rabi'},
        'Kharif Rice': {'temp': [25, 35], 'humidity': [75, 90], 'ph': [5.8, 7.0], 'season': 'Kharif'},
        'Rabi Rice': {'temp': [18, 25], 'humidity': [50, 70], 'ph': [6.0, 7.5], 'season': 'Rabi'}
    }
    
    # Wheat varieties
    wheat_varieties = {
        'Durum': {'temp': [15, 22], 'humidity': [50, 65], 'ph': [6.0, 7.5], 'season': 'Rabi'},
        'Hard Red': {'temp': [12, 25], 'humidity': [45, 60], 'ph': [6.2, 7.8], 'season': 'Rabi'},
        'Soft White': {'temp': [14, 20], 'humidity': [40, 55], 'ph': [6.5, 8.0], 'season': 'Rabi'},
        'Winter Wheat': {'temp': [10, 18], 'humidity': [35, 50], 'ph': [6.0, 7.5], 'season': 'Rabi'},
        'Spring Wheat': {'temp': [18, 28], 'humidity': [50, 70], 'ph': [6.5, 7.8], 'season': 'Zaid'}
    }
    
    # Cotton varieties
    cotton_varieties = {
        'Bt Cotton': {'temp': [21, 32], 'humidity': [60, 80], 'ph': [5.8, 8.0], 'season': 'Kharif'},
        'Organic Cotton': {'temp': [20, 30], 'humidity': [55, 75], 'ph': [6.0, 7.5], 'season': 'Kharif'},
        'Hybrid Cotton': {'temp': [22, 35], 'humidity': [65, 85], 'ph': [6.2, 8.2], 'season': 'Kharif'},
        'Desi Cotton': {'temp': [18, 28], 'humidity': [50, 70], 'ph': [5.5, 7.8], 'season': 'Kharif'}
    }
    
    # Generate synthetic data for each variety
    all_varieties = {
        'Rice': rice_varieties,
        'Wheat': wheat_varieties,
        'Cotton': cotton_varieties
    }
    
    np.random.seed(42)
    
    for base_crop, varieties in all_varieties.items():
        for variety, params in varieties.items():
            for _ in range(200):  # 200 samples per variety
                # Generate features based on variety preferences
                temp_range = params['temp']
                temp = np.random.normal((temp_range[0] + temp_range[1])/2, 3)
                temp = np.clip(temp, temp_range[0]-5, temp_range[1]+5)
                
                humidity_range = params['humidity']
                humidity = np.random.normal((humidity_range[0] + humidity_range[1])/2, 5)
                humidity = np.clip(humidity, humidity_range[0]-10, humidity_range[1]+10)
                
                ph_range = params['ph']
                ph = np.random.normal((ph_range[0] + ph_range[1])/2, 0.3)
                ph = np.clip(ph, ph_range[0]-0.5, ph_range[1]+0.5)
                
                # Generate other features
                nitrogen = np.random.normal(120, 30)
                phosphorus = np.random.normal(60, 20) 
                potassium = np.random.normal(100, 25)
                rainfall = np.random.normal(800, 200)
                moisture = np.random.normal(25, 8)
                ec = np.random.normal(1.2, 0.4)
                
                # Location (India coordinates)
                latitude = np.random.uniform(8, 37)  # India latitude range
                longitude = np.random.uniform(68, 97)  # India longitude range
                
                # Environmental factors
                wind_speed = np.random.normal(12, 4)
                solar_radiation = np.random.normal(20, 5)
                day_length = np.random.uniform(10, 14)
                
                # Season encoding (one-hot)
                season = params['season']
                season_kharif = 1 if season == 'Kharif' else 0
                season_rabi = 1 if season == 'Rabi' else 0
                season_zaid = 1 if season == 'Zaid' else 0
                
                # Create full crop name
                full_crop_name = f"{base_crop}_{variety}"
                
                row = {
                    'Temperature': temp,
                    'Humidity': humidity,
                    'pH': ph,
                    'Nitrogen': nitrogen,
                    'Phosphorus': phosphorus,
                    'Potassium': potassium,
                    'Rainfall': rainfall,
                    'Moisture': moisture,
                    'EC': ec,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'Wind_Speed': wind_speed,
                    'Solar_Radiation': solar_radiation,
                    'Day_Length': day_length,
                    'Season_Kharif': season_kharif,
                    'Season_Rabi': season_rabi,
                    'Season_Zaid': season_zaid,
                    'Crop': base_crop,
                    'Variety': variety,
                    'Full_Crop_Name': full_crop_name
                }
                
                crops_data.append(row)
    
    return pd.DataFrame(crops_data)

def train_enhanced_crop_model_direct():
    """Train enhanced crop recommendation model directly"""
    
    print("Creating enhanced crop dataset...")
    df = create_enhanced_crop_dataset()
    print(f"Dataset created: {len(df)} samples, {df['Full_Crop_Name'].nunique()} varieties")
    
    # Prepare features and target
    feature_columns = [
        'Temperature', 'Humidity', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium',
        'Rainfall', 'Moisture', 'EC', 'Latitude', 'Longitude', 'Wind_Speed',
        'Solar_Radiation', 'Day_Length', 'Season_Kharif', 'Season_Rabi', 'Season_Zaid'
    ]
    
    X = df[feature_columns]
    y = df['Full_Crop_Name']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_encoded)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Create crop variety mapping
    crop_varieties = {}
    for index, row in df.iterrows():
        base_crop = row['Crop']
        variety = row['Variety']
        full_name = row['Full_Crop_Name']
        
        if base_crop not in crop_varieties:
            crop_varieties[base_crop] = {}
        
        if variety not in crop_varieties[base_crop]:
            crop_varieties[base_crop][variety] = {
                'full_name': full_name,
                'season_preference': row[['Season_Kharif', 'Season_Rabi', 'Season_Zaid']].values.tolist()
            }
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_columns,
        'crop_varieties': crop_varieties,
        'classes': label_encoder.classes_.tolist(),
        'accuracy': accuracy,
        'trained_date': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, 'models/enhanced_crop_model.pkl')
    
    # Save labels separately
    labels_data = {
        'classes': label_encoder.classes_.tolist(),
        'crop_varieties': crop_varieties
    }
    
    with open('models/enhanced_crop_labels.json', 'w') as f:
        json.dump(labels_data, f, indent=2)
    
    print("\\nModel saved successfully!")
    print("Files created:")
    print("- models/enhanced_crop_model.pkl")
    print("- models/enhanced_crop_labels.json")
    
    return model_data

if __name__ == "__main__":
    print("Enhanced Crop Model Training")
    print("=" * 40)
    
    # Change to correct directory
    project_dir = r"c:\\Users\\Asus\\Downloads\\Major Project - AI POWERED AGRICULTURE SYSTEM\\ai_agri_full_fixed"
    os.chdir(project_dir)
    
    try:
        model_data = train_enhanced_crop_model_direct()
        print("\\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()