"""
Enhanced Crop Recommendation Model with Real-time Data Integration
Supports multiple crop varieties and real-time environmental factors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedCropRecommendationModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.crop_varieties = {}
        self.seasonal_factors = {}
        
    def prepare_enhanced_dataset(self):
        """Create enhanced dataset with variety of crops and real-time factors"""
        
        # Enhanced crop varieties with seasonal preferences
        crop_data = {
            'Rice': {
                'varieties': ['Basmati', 'Jasmine', 'Arborio', 'Kharif Rice', 'Rabi Rice'],
                'season_preference': {'Kharif': 0.8, 'Rabi': 0.6, 'Zaid': 0.3},
                'water_requirement': 'High',
                'temperature_range': [20, 35],
                'ph_preference': [5.5, 7.0]
            },
            'Wheat': {
                'varieties': ['Durum', 'Hard Red', 'Soft White', 'Winter Wheat', 'Spring Wheat'],
                'season_preference': {'Kharif': 0.2, 'Rabi': 0.9, 'Zaid': 0.1},
                'water_requirement': 'Medium',
                'temperature_range': [15, 25],
                'ph_preference': [6.0, 7.5]
            },
            'Cotton': {
                'varieties': ['Bt Cotton', 'Organic Cotton', 'Hybrid Cotton', 'Desi Cotton'],
                'season_preference': {'Kharif': 0.9, 'Rabi': 0.1, 'Zaid': 0.2},
                'water_requirement': 'Medium',
                'temperature_range': [21, 32],
                'ph_preference': [5.8, 8.0]
            },
            'Sugarcane': {
                'varieties': ['Co-86032', 'Co-0238', 'Co-62175', 'Early Maturing', 'Late Maturing'],
                'season_preference': {'Kharif': 0.7, 'Rabi': 0.8, 'Zaid': 0.9},
                'water_requirement': 'High',
                'temperature_range': [20, 30],
                'ph_preference': [6.5, 7.5]
            },
            'Maize': {
                'varieties': ['Sweet Corn', 'Dent Corn', 'Flint Corn', 'Hybrid Maize', 'Baby Corn'],
                'season_preference': {'Kharif': 0.8, 'Rabi': 0.6, 'Zaid': 0.7},
                'water_requirement': 'Medium',
                'temperature_range': [21, 27],
                'ph_preference': [6.0, 6.8]
            },
            'Groundnut': {
                'varieties': ['Spanish', 'Runner', 'Virginia', 'Valencia', 'Bunch Type'],
                'season_preference': {'Kharif': 0.9, 'Rabi': 0.5, 'Zaid': 0.3},
                'water_requirement': 'Low',
                'temperature_range': [20, 30],
                'ph_preference': [6.0, 7.0]
            },
            'Barley': {
                'varieties': ['Six-row', 'Two-row', 'Hulled', 'Hulless', 'Malting Barley'],
                'season_preference': {'Kharif': 0.1, 'Rabi': 0.9, 'Zaid': 0.2},
                'water_requirement': 'Low',
                'temperature_range': [12, 20],
                'ph_preference': [6.0, 7.0]
            },
            'Millets': {
                'varieties': ['Pearl Millet', 'Finger Millet', 'Foxtail Millet', 'Little Millet'],
                'season_preference': {'Kharif': 0.8, 'Rabi': 0.4, 'Zaid': 0.6},
                'water_requirement': 'Very Low',
                'temperature_range': [25, 35],
                'ph_preference': [5.0, 7.5]
            },
            'Pulses': {
                'varieties': ['Chickpea', 'Pigeon Pea', 'Black Gram', 'Green Gram', 'Lentil'],
                'season_preference': {'Kharif': 0.6, 'Rabi': 0.8, 'Zaid': 0.4},
                'water_requirement': 'Low',
                'temperature_range': [20, 30],
                'ph_preference': [6.0, 7.5]
            }
        }
        
        self.crop_varieties = crop_data
        
        # Generate synthetic training data with variety information
        np.random.seed(42)
        n_samples = 5000
        
        data = []
        
        for _ in range(n_samples):
            # Select random crop and variety
            crop = np.random.choice(list(crop_data.keys()))
            variety = np.random.choice(crop_data[crop]['varieties'])
            
            # Generate features based on crop preferences
            temp_range = crop_data[crop]['temperature_range']
            ph_range = crop_data[crop]['ph_preference']
            
            # Environmental features
            temperature = np.random.normal((temp_range[0] + temp_range[1])/2, 3)
            temperature = np.clip(temperature, temp_range[0]-5, temp_range[1]+5)
            
            ph = np.random.normal((ph_range[0] + ph_range[1])/2, 0.5)
            ph = np.clip(ph, ph_range[0]-1, ph_range[1]+1)
            
            # Seasonal factor
            current_season = np.random.choice(['Kharif', 'Rabi', 'Zaid'])
            season_factor = crop_data[crop]['season_preference'][current_season]
            
            # Add noise based on season suitability
            if season_factor < 0.5:
                temperature += np.random.normal(0, 2)  # More variation for unsuitable seasons
                ph += np.random.normal(0, 0.3)
            
            # Soil nutrients (vary by crop type)
            if crop in ['Rice', 'Sugarcane']:
                nitrogen = np.random.normal(80, 20)
                phosphorus = np.random.normal(60, 15)
                potassium = np.random.normal(40, 10)
            elif crop in ['Wheat', 'Barley']:
                nitrogen = np.random.normal(120, 25)
                phosphorus = np.random.normal(50, 12)
                potassium = np.random.normal(60, 15)
            else:
                nitrogen = np.random.normal(100, 30)
                phosphorus = np.random.normal(45, 20)
                potassium = np.random.normal(50, 20)
            
            # Environmental conditions
            humidity = np.random.normal(65, 15)
            rainfall = np.random.exponential(100) if crop_data[crop]['water_requirement'] == 'High' else np.random.exponential(50)
            
            # Soil properties
            moisture = np.random.normal(25, 8)
            ec = np.random.normal(1.5, 0.5)
            
            # Location-based features (simplified)
            latitude = np.random.uniform(8, 35)  # India's latitude range
            longitude = np.random.uniform(68, 98)  # India's longitude range
            
            # Real-time factors
            wind_speed = np.random.normal(10, 3)
            solar_radiation = np.random.normal(20, 5)
            day_length = np.random.normal(12, 2)
            
            data.append({
                'Temperature': temperature,
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
                'Season_Kharif': 1 if current_season == 'Kharif' else 0,
                'Season_Rabi': 1 if current_season == 'Rabi' else 0,
                'Season_Zaid': 1 if current_season == 'Zaid' else 0,
                'Crop': crop,
                'Variety': variety,
                'Full_Crop_Name': f"{crop}_{variety.replace(' ', '_')}"
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, df, target_column='Full_Crop_Name'):
        """Train enhanced crop recommendation model"""
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['Crop', 'Variety', 'Full_Crop_Name']]
        X = df[feature_columns]
        y = df[target_column]
        
        self.feature_names = feature_columns
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train ensemble model
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
        
        # Cross-validation
        rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
        gb_scores = cross_val_score(gb, X_train, y_train, cv=5)
        
        print(f"Random Forest CV Score: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")
        print(f"Gradient Boosting CV Score: {gb_scores.mean():.4f} (+/- {gb_scores.std() * 2:.4f})")
        
        # Choose best model
        if rf_scores.mean() > gb_scores.mean():
            self.model = rf
            print("Selected: Random Forest")
        else:
            self.model = gb
            print("Selected: Gradient Boosting")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10))
        
        return accuracy, feature_importance
    
    def predict_with_variety(self, features, top_k=5):
        """Predict crop varieties with confidence scores"""
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            crop_variety = self.label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx]
            
            # Parse crop and variety
            parts = crop_variety.split('_')
            crop = parts[0]
            variety = '_'.join(parts[1:]).replace('_', ' ')
            
            predictions.append({
                'crop': crop,
                'variety': variety,
                'full_name': crop_variety.replace('_', ' '),
                'confidence': confidence,
                'suitability_score': self._calculate_suitability(crop, features)
            })
        
        return predictions
    
    def _calculate_suitability(self, crop, features):
        """Calculate crop suitability based on environmental conditions"""
        
        if crop not in self.crop_varieties:
            return 0.5
        
        crop_info = self.crop_varieties[crop]
        score = 1.0
        
        # Temperature suitability
        temp_range = crop_info['temperature_range']
        temp = features[0]  # Assuming temperature is first feature
        if temp_range[0] <= temp <= temp_range[1]:
            temp_score = 1.0
        else:
            temp_deviation = min(abs(temp - temp_range[0]), abs(temp - temp_range[1]))
            temp_score = max(0, 1 - temp_deviation / 10)
        
        # pH suitability
        ph_range = crop_info['ph_preference']
        ph = features[2]  # Assuming pH is third feature
        if ph_range[0] <= ph <= ph_range[1]:
            ph_score = 1.0
        else:
            ph_deviation = min(abs(ph - ph_range[0]), abs(ph - ph_range[1]))
            ph_score = max(0, 1 - ph_deviation / 2)
        
        # Combine scores
        suitability = (temp_score + ph_score) / 2
        return suitability
    
    def get_seasonal_recommendations(self, current_season='Kharif'):
        """Get seasonal crop recommendations"""
        
        seasonal_crops = {}
        for crop, info in self.crop_varieties.items():
            season_score = info['season_preference'].get(current_season, 0.5)
            if season_score >= 0.6:
                seasonal_crops[crop] = {
                    'varieties': info['varieties'],
                    'suitability': season_score,
                    'water_requirement': info['water_requirement']
                }
        
        return seasonal_crops
    
    def save_model(self, model_path='models/enhanced_crop_model.pkl'):
        """Save the trained model and metadata"""
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'crop_varieties': self.crop_varieties,
            'trained_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        
        # Save variety information as JSON
        with open('models/crop_varieties.json', 'w') as f:
            json.dump(self.crop_varieties, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/enhanced_crop_model.pkl'):
        """Load the trained model"""
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.crop_varieties = model_data.get('crop_varieties', {})
        
        print("Model loaded successfully")

def main():
    """Train and save the enhanced crop recommendation model"""
    
    print("Training Enhanced Crop Recommendation Model...")
    print("=" * 50)
    
    # Initialize model
    model = EnhancedCropRecommendationModel()
    
    # Generate enhanced dataset
    print("Generating enhanced training dataset...")
    df = model.prepare_enhanced_dataset()
    print(f"Dataset created with {len(df)} samples and {len(df.columns)} features")
    print(f"Number of unique crop varieties: {df['Full_Crop_Name'].nunique()}")
    
    # Train model
    print("\nTraining model...")
    accuracy, feature_importance = model.train_model(df)
    
    # Save model
    print("\nSaving model...")
    model.save_model()
    
    # Test prediction
    print("\nTesting prediction...")
    sample_features = [25, 65, 6.5, 100, 50, 60, 80, 30, 1.2, 20, 77, 12, 22, 12, 1, 0, 0]  # Sample features
    predictions = model.predict_with_variety(sample_features)
    
    print("\nSample Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['full_name']}: {pred['confidence']:.3f} (Suitability: {pred['suitability_score']:.3f})")
    
    # Get seasonal recommendations
    seasonal = model.get_seasonal_recommendations('Kharif')
    print(f"\nKharif Season Recommendations: {list(seasonal.keys())}")
    
    print("\nModel training completed successfully!")
    return model

# Export function for external use
def train_enhanced_crop_model():
    """External function to train the enhanced crop model"""
    return main()

if __name__ == "__main__":
    model = main()