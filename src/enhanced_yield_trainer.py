"""
Enhanced Yield Prediction Model with Real-time Data Integration
Supports multiple crop varieties and environmental factors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedYieldPredictionModel:
    def __init__(self):
        self.models = {}  # Multiple models for different quantiles
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.yield_factors = {}
        
    def setup_yield_factors(self):
        """Setup comprehensive yield influencing factors"""
        
        self.yield_factors = {
            'environmental': {
                'temperature': {
                    'optimal_ranges': {
                        'Rice': [20, 35],
                        'Wheat': [15, 25], 
                        'Corn': [18, 27],
                        'Cotton': [23, 32],
                        'Sugarcane': [20, 30],
                        'Soybean': [20, 30]
                    },
                    'impact_factor': 0.25
                },
                'rainfall': {
                    'optimal_ranges': {
                        'Rice': [1200, 1800],
                        'Wheat': [450, 650],
                        'Corn': [500, 800],
                        'Cotton': [500, 1000],
                        'Sugarcane': [1000, 1500],
                        'Soybean': [450, 700]
                    },
                    'impact_factor': 0.30
                },
                'humidity': {
                    'optimal_ranges': {
                        'Rice': [70, 85],
                        'Wheat': [55, 70],
                        'Corn': [60, 75],
                        'Cotton': [60, 80],
                        'Sugarcane': [65, 80],
                        'Soybean': [60, 75]
                    },
                    'impact_factor': 0.15
                }
            },
            'soil': {
                'ph': {
                    'optimal_ranges': {
                        'Rice': [5.5, 6.5],
                        'Wheat': [6.0, 7.0],
                        'Corn': [6.0, 6.8],
                        'Cotton': [5.8, 8.0],
                        'Sugarcane': [6.5, 7.5],
                        'Soybean': [6.0, 6.8]
                    },
                    'impact_factor': 0.20
                },
                'organic_matter': {
                    'minimum_required': 2.0,
                    'optimal_range': [3.0, 5.0],
                    'impact_factor': 0.15
                },
                'nutrient_balance': {
                    'NPK_ratios': {
                        'Rice': [3, 1, 2],
                        'Wheat': [4, 2, 3],
                        'Corn': [4, 2, 3],
                        'Cotton': [3, 2, 4],
                        'Sugarcane': [3, 1, 3],
                        'Soybean': [2, 2, 3]
                    },
                    'impact_factor': 0.25
                }
            },
            'management': {
                'irrigation_efficiency': {
                    'impact_factor': 0.20,
                    'methods': {
                        'Drip': 1.2,
                        'Sprinkler': 1.1,
                        'Flood': 0.9,
                        'Furrow': 0.95
                    }
                },
                'pest_management': {
                    'impact_factor': 0.15,
                    'effectiveness': {
                        'Integrated': 1.15,
                        'Chemical': 1.0,
                        'Organic': 0.95,
                        'Biological': 1.05
                    }
                },
                'variety_selection': {
                    'impact_factor': 0.25,
                    'types': {
                        'High_Yield': 1.2,
                        'Disease_Resistant': 1.1,
                        'Drought_Tolerant': 1.05,
                        'Traditional': 1.0
                    }
                }
            },
            'seasonal': {
                'planting_timing': {
                    'impact_factor': 0.20,
                    'seasons': {
                        'Optimal': 1.15,
                        'Early': 1.05,
                        'Late': 0.95,
                        'Off_Season': 0.8
                    }
                },
                'harvest_timing': {
                    'impact_factor': 0.10,
                    'conditions': {
                        'Optimal': 1.1,
                        'Good': 1.05,
                        'Poor': 0.9
                    }
                }
            }
        }
    
    def generate_enhanced_yield_dataset(self, n_samples=10000):
        """Generate comprehensive yield dataset with multiple factors"""
        
        crops = ['Rice', 'Wheat', 'Corn', 'Cotton', 'Sugarcane', 'Soybean']
        regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast']
        soil_types = ['Clay', 'Sandy', 'Loam', 'Silt']
        irrigation_methods = ['Drip', 'Sprinkler', 'Flood', 'Furrow']
        
        data = []
        
        np.random.seed(42)
        
        for _ in range(n_samples):
            # Basic crop information
            crop = np.random.choice(crops)
            region = np.random.choice(regions)
            soil_type = np.random.choice(soil_types)
            
            # Environmental factors
            temp_range = self.yield_factors['environmental']['temperature']['optimal_ranges'][crop]
            base_temp = np.random.normal((temp_range[0] + temp_range[1])/2, 3)
            temperature = np.clip(base_temp, temp_range[0]-10, temp_range[1]+10)
            
            rainfall_range = self.yield_factors['environmental']['rainfall']['optimal_ranges'][crop]
            base_rainfall = np.random.normal((rainfall_range[0] + rainfall_range[1])/2, 100)
            rainfall = np.clip(base_rainfall, rainfall_range[0]-300, rainfall_range[1]+500)
            
            humidity_range = self.yield_factors['environmental']['humidity']['optimal_ranges'][crop]
            base_humidity = np.random.normal((humidity_range[0] + humidity_range[1])/2, 5)
            humidity = np.clip(base_humidity, humidity_range[0]-15, humidity_range[1]+15)
            
            # Soil factors
            ph_range = self.yield_factors['soil']['ph']['optimal_ranges'][crop]
            base_ph = np.random.normal((ph_range[0] + ph_range[1])/2, 0.3)
            soil_ph = np.clip(base_ph, ph_range[0]-1, ph_range[1]+1)
            
            organic_matter = np.random.normal(3.5, 1.0)
            organic_matter = np.clip(organic_matter, 0.5, 8.0)
            
            # Nutrient levels
            npk_ratio = self.yield_factors['soil']['nutrient_balance']['NPK_ratios'][crop]
            base_n = np.random.normal(120, 30)
            nitrogen = base_n * npk_ratio[0] / sum(npk_ratio)
            phosphorus = base_n * npk_ratio[1] / sum(npk_ratio)
            potassium = base_n * npk_ratio[2] / sum(npk_ratio)
            
            # Management factors
            irrigation_method = np.random.choice(irrigation_methods)
            pest_management = np.random.choice(['Integrated', 'Chemical', 'Organic', 'Biological'])
            variety_type = np.random.choice(['High_Yield', 'Disease_Resistant', 'Drought_Tolerant', 'Traditional'])
            planting_timing = np.random.choice(['Optimal', 'Early', 'Late', 'Off_Season'])
            
            # Additional real-time factors
            wind_speed = np.random.normal(10, 3)
            solar_radiation = np.random.normal(20, 5)
            co2_level = np.random.normal(410, 20)  # ppm
            
            # Farm management
            farm_size = np.random.exponential(5)  # hectares
            experience_years = np.random.randint(1, 40)
            technology_adoption = np.random.uniform(0.3, 1.0)
            
            # Calculate base yield
            base_yield = self._calculate_base_yield(crop, region)
            
            # Apply environmental factors
            temp_factor = self._calculate_environmental_factor(
                temperature, temp_range, self.yield_factors['environmental']['temperature']['impact_factor']
            )
            rainfall_factor = self._calculate_environmental_factor(
                rainfall, rainfall_range, self.yield_factors['environmental']['rainfall']['impact_factor']
            )
            humidity_factor = self._calculate_environmental_factor(
                humidity, humidity_range, self.yield_factors['environmental']['humidity']['impact_factor']
            )
            
            # Apply soil factors  
            ph_factor = self._calculate_environmental_factor(
                soil_ph, ph_range, self.yield_factors['soil']['ph']['impact_factor']
            )
            om_factor = self._calculate_organic_matter_factor(organic_matter)
            
            # Apply management factors
            irrigation_factor = self.yield_factors['management']['irrigation_efficiency']['methods'][irrigation_method]
            pest_factor = self.yield_factors['management']['pest_management']['effectiveness'][pest_management]
            variety_factor = self.yield_factors['management']['variety_selection']['types'][variety_type]
            timing_factor = self.yield_factors['seasonal']['planting_timing']['seasons'][planting_timing]
            
            # Technology and experience factors
            tech_factor = 0.8 + 0.4 * technology_adoption
            exp_factor = min(1.2, 0.8 + experience_years * 0.01)
            
            # Calculate final yield
            yield_multiplier = (temp_factor * rainfall_factor * humidity_factor * 
                              ph_factor * om_factor * irrigation_factor * 
                              pest_factor * variety_factor * timing_factor * 
                              tech_factor * exp_factor)
            
            # Add some randomness
            random_factor = np.random.normal(1.0, 0.15)
            final_yield = base_yield * yield_multiplier * random_factor
            final_yield = max(0, final_yield)  # Ensure non-negative yield
            
            data.append({
                'Crop': crop,
                'Region': region,
                'Soil_Type': soil_type,
                'Temperature': temperature,
                'Rainfall': rainfall,
                'Humidity': humidity,
                'Soil_pH': soil_ph,
                'Organic_Matter': organic_matter,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium,
                'Irrigation_Method': irrigation_method,
                'Pest_Management': pest_management,
                'Variety_Type': variety_type,
                'Planting_Timing': planting_timing,
                'Wind_Speed': wind_speed,
                'Solar_Radiation': solar_radiation,
                'CO2_Level': co2_level,
                'Farm_Size': farm_size,
                'Experience_Years': experience_years,
                'Technology_Adoption': technology_adoption,
                'Yield_tons_per_hectare': final_yield
            })
        
        return pd.DataFrame(data)
    
    def _calculate_base_yield(self, crop, region):
        """Calculate base yield for crop and region"""
        base_yields = {
            'Rice': {'North': 4.5, 'South': 5.2, 'East': 4.8, 'West': 3.9, 'Central': 4.2, 'Northeast': 3.5},
            'Wheat': {'North': 4.2, 'South': 2.8, 'East': 3.5, 'West': 3.8, 'Central': 4.0, 'Northeast': 2.5},
            'Corn': {'North': 5.5, 'South': 6.2, 'East': 5.8, 'West': 4.5, 'Central': 5.2, 'Northeast': 3.8},
            'Cotton': {'North': 2.2, 'South': 2.8, 'East': 2.1, 'West': 2.5, 'Central': 2.4, 'Northeast': 1.8},
            'Sugarcane': {'North': 65, 'South': 75, 'East': 70, 'West': 60, 'Central': 68, 'Northeast': 55},
            'Soybean': {'North': 2.8, 'South': 3.2, 'East': 3.0, 'West': 2.5, 'Central': 2.9, 'Northeast': 2.3}
        }
        return base_yields.get(crop, {}).get(region, 3.0)
    
    def _calculate_environmental_factor(self, value, optimal_range, impact_factor):
        """Calculate environmental impact factor"""
        if optimal_range[0] <= value <= optimal_range[1]:
            return 1.0 + impact_factor * 0.2  # Small bonus for optimal conditions
        else:
            deviation = min(abs(value - optimal_range[0]), abs(value - optimal_range[1]))
            max_deviation = max(optimal_range[1] - optimal_range[0], 10)  # Prevent division by zero
            penalty = impact_factor * (deviation / max_deviation)
            return max(0.5, 1.0 - penalty)
    
    def _calculate_organic_matter_factor(self, organic_matter):
        """Calculate organic matter impact factor"""
        if organic_matter >= 3.0:
            return 1.0 + 0.15 * min((organic_matter - 3.0) / 2.0, 1.0)
        else:
            return max(0.7, 1.0 - 0.3 * (3.0 - organic_matter) / 3.0)
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        
        # Encode categorical variables
        categorical_columns = ['Crop', 'Region', 'Soil_Type', 'Irrigation_Method', 
                             'Pest_Management', 'Variety_Type', 'Planting_Timing']
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        # Select features
        feature_columns = [col for col in df_encoded.columns if col != 'Yield_tons_per_hectare']
        X = df_encoded[feature_columns]
        y = df_encoded['Yield_tons_per_hectare']
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train_quantile_models(self, df, quantiles=[0.1, 0.5, 0.9]):
        """Train models for different yield quantiles"""
        
        X, y = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        for quantile in quantiles:
            print(f"Training model for quantile {quantile}...")
            
            # Use GradientBoostingRegressor with quantile loss
            if quantile == 0.5:
                # Use RandomForest for median prediction
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Use GradientBoosting for quantile regression
                model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42,
                    loss='quantile',
                    alpha=quantile
                )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models[f'q{int(quantile*100)}'] = model
            
            results[quantile] = {
                'cv_score': -cv_scores.mean(),
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"Quantile {quantile}: CV MSE = {-cv_scores.mean():.3f}, "
                  f"Test MSE = {mse:.3f}, R² = {r2:.3f}")
        
        # Feature importance (from median model)
        if 'q50' in self.models:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['q50'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(feature_importance.head(10))
        
        return results
    
    def predict_yield_with_uncertainty(self, features_dict):
        """Predict yield with uncertainty bounds"""
        
        # Prepare features
        feature_row = []
        for feature in self.feature_names:
            if feature in ['Crop', 'Region', 'Soil_Type', 'Irrigation_Method', 
                          'Pest_Management', 'Variety_Type', 'Planting_Timing']:
                # Encode categorical
                if feature in features_dict and feature in self.label_encoders:
                    encoded_value = self.label_encoders[feature].transform([features_dict[feature]])[0]
                    feature_row.append(encoded_value)
                else:
                    feature_row.append(0)  # Default value
            else:
                # Numerical feature
                feature_row.append(features_dict.get(feature, 0))
        
        # Scale features
        features_scaled = self.scaler.transform([feature_row])
        
        # Predict with all models
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions[model_name] = max(0, pred)  # Ensure non-negative
        
        # Calculate prediction interval
        lower_bound = predictions.get('q10', predictions.get('q50', 0) * 0.7)
        median = predictions.get('q50', 0)
        upper_bound = predictions.get('q90', predictions.get('q50', 0) * 1.3)
        
        return {
            'predicted_yield': median,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': upper_bound - lower_bound,
            'confidence_interval': f"{lower_bound:.2f} - {upper_bound:.2f}",
            'all_quantiles': predictions
        }
    
    def get_yield_recommendations(self, features_dict, crop):
        """Get recommendations to improve yield"""
        
        recommendations = []
        
        # Check environmental factors
        if crop in self.yield_factors['environmental']['temperature']['optimal_ranges']:
            temp_range = self.yield_factors['environmental']['temperature']['optimal_ranges'][crop]
            current_temp = features_dict.get('Temperature', 25)
            
            if current_temp < temp_range[0]:
                recommendations.append(f"Consider greenhouse cultivation or delayed planting (current temp: {current_temp:.1f}°C, optimal: {temp_range[0]}-{temp_range[1]}°C)")
            elif current_temp > temp_range[1]:
                recommendations.append(f"Implement cooling strategies or shade nets (current temp: {current_temp:.1f}°C, optimal: {temp_range[0]}-{temp_range[1]}°C)")
        
        # Check soil pH
        if crop in self.yield_factors['soil']['ph']['optimal_ranges']:
            ph_range = self.yield_factors['soil']['ph']['optimal_ranges'][crop]
            current_ph = features_dict.get('Soil_pH', 6.5)
            
            if current_ph < ph_range[0]:
                recommendations.append(f"Apply lime to increase soil pH (current: {current_ph:.1f}, optimal: {ph_range[0]}-{ph_range[1]})")
            elif current_ph > ph_range[1]:
                recommendations.append(f"Apply sulfur to decrease soil pH (current: {current_ph:.1f}, optimal: {ph_range[0]}-{ph_range[1]})")
        
        # Check organic matter
        current_om = features_dict.get('Organic_Matter', 2.0)
        if current_om < 3.0:
            recommendations.append(f"Increase organic matter through compost or manure application (current: {current_om:.1f}%, target: >3.0%)")
        
        # Check irrigation method
        current_irrigation = features_dict.get('Irrigation_Method', 'Flood')
        if current_irrigation in ['Flood', 'Furrow']:
            recommendations.append("Consider upgrading to drip or sprinkler irrigation for better water efficiency")
        
        # Check variety selection
        current_variety = features_dict.get('Variety_Type', 'Traditional')
        if current_variety == 'Traditional':
            recommendations.append("Consider high-yield or disease-resistant varieties for better productivity")
        
        return recommendations
    
    def save_models(self, model_path='models/enhanced_yield_models.pkl'):
        """Save all trained models and metadata"""
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'yield_factors': self.yield_factors,
            'trained_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        
        # Save yield factors as JSON
        with open('models/yield_factors.json', 'w') as f:
            json.dump(self.yield_factors, f, indent=2)
        
        print(f"Enhanced yield models saved to {model_path}")

def main():
    """Train and save enhanced yield prediction models"""
    
    print("Training Enhanced Yield Prediction Models...")
    print("=" * 50)
    
    # Initialize model
    model = EnhancedYieldPredictionModel()
    
    # Setup yield factors
    model.setup_yield_factors()
    print("Yield factors database setup completed")
    
    # Generate enhanced dataset
    print("Generating enhanced yield dataset...")
    df = model.generate_enhanced_yield_dataset(n_samples=15000)
    print(f"Dataset created with {len(df)} samples and {len(df.columns)} features")
    print(f"Yield statistics: Mean={df['Yield_tons_per_hectare'].mean():.2f}, Std={df['Yield_tons_per_hectare'].std():.2f}")
    
    # Train models
    print("\nTraining quantile regression models...")
    results = model.train_quantile_models(df)
    
    # Save models
    print("\nSaving models...")
    model.save_models()
    
    # Test prediction
    print("\nTesting prediction...")
    sample_features = {
        'Crop': 'Rice',
        'Region': 'South',
        'Soil_Type': 'Clay',
        'Temperature': 28,
        'Rainfall': 1400,
        'Humidity': 75,
        'Soil_pH': 6.2,
        'Organic_Matter': 3.5,
        'Nitrogen': 150,
        'Phosphorus': 50,
        'Potassium': 100,
        'Irrigation_Method': 'Drip',
        'Pest_Management': 'Integrated',
        'Variety_Type': 'High_Yield',
        'Planting_Timing': 'Optimal',
        'Wind_Speed': 12,
        'Solar_Radiation': 22,
        'CO2_Level': 415,
        'Farm_Size': 3.5,
        'Experience_Years': 15,
        'Technology_Adoption': 0.8
    }
    
    prediction = model.predict_yield_with_uncertainty(sample_features)
    recommendations = model.get_yield_recommendations(sample_features, 'Rice')
    
    print(f"\nSample Prediction:")
    print(f"Expected Yield: {prediction['predicted_yield']:.2f} tons/hectare")
    print(f"Confidence Interval: {prediction['confidence_interval']} tons/hectare")
    print(f"Uncertainty: ±{prediction['uncertainty']:.2f} tons/hectare")
    
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    print("\nEnhanced yield prediction model training completed!")
    return model

# Export function for external use
def train_enhanced_yield_model():
    """External function to train the enhanced yield model"""
    return main()

if __name__ == "__main__":
    model = main()