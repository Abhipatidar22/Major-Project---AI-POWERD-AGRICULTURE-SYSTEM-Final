"""
Advanced Crop Recommendation Model using the new dataset
Includes feature engineering, model explainability, and performance metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

from features import save_label_encoder
from utils import save_pickle

def load_and_preprocess_data(csv_path):
    """Load and preprocess the crop recommendation dataset"""
    df = pd.read_csv(csv_path)
    
    # Select relevant features for crop recommendation
    feature_cols = [
        'Moisture_vol_pct', 'pH', 'Nitrogen_ppm', 'Phosphorus_ppm', 'Potassium_ppm',
        'Soil_Temperature_C', 'Ambient_Humidity_pct', 'EC_dS_per_m',
        'HSV_mean_H', 'HSV_mean_S', 'Texture_Contrast', 'Texture_Entropy',
        'Soil_Type'
    ]
    
    # Create feature dataframe
    X = df[feature_cols].copy()
    y = df['Recommended_Crop']
    
    # One-hot encode soil type
    X = pd.get_dummies(X, columns=['Soil_Type'], prefix='soil', drop_first=False)
    
    # Create derived features
    X['NPK_ratio'] = X['Nitrogen_ppm'] / (X['Phosphorus_ppm'] + X['Potassium_ppm'] + 1e-6)
    X['nutrient_balance'] = np.sqrt(X['Nitrogen_ppm']**2 + X['Phosphorus_ppm']**2 + X['Potassium_ppm']**2)
    X['moisture_pH_interaction'] = X['Moisture_vol_pct'] * X['pH']
    X['temp_humidity_index'] = X['Soil_Temperature_C'] * X['Ambient_Humidity_pct'] / 100
    X['texture_complexity'] = X['Texture_Contrast'] * X['Texture_Entropy']
    
    return X, y

def train_advanced_crop_model(csv_path, models_dir):
    """Train advanced crop recommendation models with hyperparameter tuning"""
    
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(csv_path)
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    save_label_encoder(models_dir / 'advanced_crops_labels.json', le.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with hyperparameter tuning
    print("Training Random Forest with hyperparameter tuning...")
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    
    # Train Gradient Boosting
    print("Training Gradient Boosting Classifier...")
    gb_params = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [5, 7, 9]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    
    best_gb = gb_grid.best_estimator_
    
    # Evaluate models
    rf_score = accuracy_score(y_test, best_rf.predict(X_test))
    gb_score = accuracy_score(y_test, best_gb.predict(X_test_scaled))
    
    print(f"Random Forest Accuracy: {rf_score:.4f}")
    print(f"Gradient Boosting Accuracy: {gb_score:.4f}")
    
    # Select best model
    if rf_score >= gb_score:
        best_model = best_rf
        model_type = 'RandomForest'
        use_scaling = False
        X_test_final = X_test
    else:
        best_model = best_gb
        model_type = 'GradientBoosting'
        use_scaling = True
        X_test_final = X_test_scaled
    
    # Generate detailed performance report
    y_pred = best_model.predict(X_test_final)
    print(f"\nBest Model: {model_type}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model and metadata
    model_data = {
        'model': best_model,
        'scaler': scaler if use_scaling else None,
        'features': list(X.columns),
        'model_type': model_type,
        'use_scaling': use_scaling,
        'feature_importance': feature_importance.to_dict('records'),
        'accuracy': rf_score if model_type == 'RandomForest' else gb_score,
        'cross_val_scores': cross_val_score(best_model, X_train_scaled if use_scaling else X_train, y_train, cv=5)
    }
    
    save_pickle(model_data, models_dir / 'advanced_crop_model.pkl')
    
    return model_data

def predict_with_explanation(model_data, input_features):
    """Make prediction with feature importance explanation"""
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Prepare input
    X = pd.DataFrame([input_features], columns=features)
    
    if model_data['use_scaling'] and scaler:
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0]
    else:
        proba = model.predict_proba(X)[0]
    
    # Get feature contributions (simplified explanation)
    feature_contributions = []
    for i, feature in enumerate(features):
        importance = next(item['importance'] for item in model_data['feature_importance'] if item['feature'] == feature)
        contribution = importance * abs(input_features[i]) if isinstance(input_features[i], (int, float)) else importance
        feature_contributions.append({
            'feature': feature,
            'value': input_features[i],
            'contribution': contribution
        })
    
    feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
    
    return proba, feature_contributions

if __name__ == '__main__':
    # Set paths
    csv_path = Path('c:/Users/Asus/Downloads/crop_recommendation_dataset.csv')
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    if csv_path.exists():
        print("Training advanced crop recommendation model...")
        model_data = train_advanced_crop_model(csv_path, models_dir)
        print("Advanced model training completed!")
    else:
        print(f"Dataset not found at {csv_path}")
        print("Please ensure the crop_recommendation_dataset.csv is in the correct location.")