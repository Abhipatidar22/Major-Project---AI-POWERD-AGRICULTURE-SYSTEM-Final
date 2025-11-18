#!/usr/bin/env python3
"""
Simple Enhanced Model Training Script
Trains all enhanced models for AI Agriculture System
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("Starting enhanced model training...")

# Ensure we're in the right directory
project_dir = r"c:\Users\Asus\Downloads\Major Project - AI POWERED AGRICULTURE SYSTEM\ai_agri_full_fixed"
os.chdir(project_dir)

if project_dir not in sys.path:
    sys.path.append(project_dir)

try:
    # 1. Train Enhanced Crop Recommendation Model
    print("\n1. Training Enhanced Crop Recommendation Model...")
    from src.enhanced_crop_trainer import train_enhanced_crop_model
    
    crop_model = train_enhanced_crop_model()
    print("   - Enhanced crop model trained successfully")
    
    # 2. Train Enhanced Disease Detection Model  
    print("\n2. Training Enhanced Disease Detection Model...")
    from src.enhanced_disease_trainer import train_enhanced_disease_model
    
    disease_model = train_enhanced_disease_model()
    print("   - Enhanced disease model trained successfully")
    
    # 3. Train Enhanced Soil Classification Model
    print("\n3. Training Enhanced Soil Classification Model...")
    from src.enhanced_soil_trainer import train_enhanced_soil_model
    
    soil_model = train_enhanced_soil_model()
    print("   - Enhanced soil model trained successfully")
    
    # 4. Train Enhanced Yield Prediction Model
    print("\n4. Training Enhanced Yield Prediction Model...")
    from src.enhanced_yield_trainer import train_enhanced_yield_model
    
    yield_models = train_enhanced_yield_model()
    print("   - Enhanced yield models trained successfully")
    
    print("\n" + "="*50)
    print("ALL ENHANCED MODELS TRAINED SUCCESSFULLY!")
    print("="*50)
    
    # Verify models exist
    models_dir = os.path.join(project_dir, 'models')
    expected_models = [
        'enhanced_crop_model.pkl',
        'enhanced_crop_labels.json', 
        'enhanced_disease_cnn.keras',
        'enhanced_disease_labels.json',
        'enhanced_soil_cnn.keras',
        'enhanced_soil_labels.json',
        'enhanced_yield_q10.pkl',
        'enhanced_yield_q50.pkl', 
        'enhanced_yield_q90.pkl'
    ]
    
    print("\nModel Verification:")
    for model_file in expected_models:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            print(f"   [OK] {model_file}")
        else:
            print(f"   [MISSING] {model_file}")
    
except Exception as e:
    print(f"Error during training: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTraining script completed!")