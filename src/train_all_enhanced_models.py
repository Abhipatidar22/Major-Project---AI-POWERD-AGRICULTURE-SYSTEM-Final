"""
Master Training Script for All Enhanced AI Agriculture Models
Trains all 4 modules with real-time data integration and variety support
"""

import sys
import os
import time
from datetime import datetime

def train_all_enhanced_models():
    """Train all enhanced AI agriculture models"""
    
    print("🌾 Enhanced AI Agriculture System - Model Training")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    models_trained = []
    
    try:
        # 1. Train Enhanced Crop Recommendation Model
        print("\n🌱 STEP 1: Training Enhanced Crop Recommendation Model")
        print("-" * 50)
        from enhanced_crop_trainer import main as train_crop
        crop_model = train_crop()
        models_trained.append("Crop Recommendation")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error training crop model: {e}")
    
    try:
        # 2. Train Enhanced Disease Detection Model
        print("\n🍃 STEP 2: Training Enhanced Disease Detection Model")
        print("-" * 50)
        from enhanced_disease_trainer import main as train_disease
        disease_model = train_disease()
        models_trained.append("Disease Detection")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error training disease model: {e}")
    
    try:
        # 3. Train Enhanced Soil Classification Model
        print("\n🌍 STEP 3: Training Enhanced Soil Classification Model")
        print("-" * 50)
        from enhanced_soil_trainer import main as train_soil
        soil_model = train_soil()
        models_trained.append("Soil Classification")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error training soil model: {e}")
    
    try:
        # 4. Train Enhanced Yield Prediction Model
        print("\n📈 STEP 4: Training Enhanced Yield Prediction Model")
        print("-" * 50)
        from enhanced_yield_trainer import main as train_yield
        yield_model = train_yield()
        models_trained.append("Yield Prediction")
        
    except Exception as e:
        print(f"❌ Error training yield model: {e}")
    
    # Training Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ Successfully trained {len(models_trained)}/4 models:")
    
    for i, model in enumerate(models_trained, 1):
        print(f"   {i}. {model}")
    
    if len(models_trained) < 4:
        failed_models = set(["Crop Recommendation", "Disease Detection", 
                           "Soil Classification", "Yield Prediction"]) - set(models_trained)
        print(f"\n❌ Failed to train {len(failed_models)} models:")
        for model in failed_models:
            print(f"   - {model}")
    
    print("\n📁 Model files saved in: models/")
    print("   - enhanced_crop_model.pkl")
    print("   - enhanced_disease_model.h5")  
    print("   - enhanced_soil_cnn.h5")
    print("   - enhanced_yield_models.pkl")
    print("   - Various JSON metadata files")
    
    print("\n🚀 Your enhanced AI Agriculture System is ready!")
    print("   Run the application with: streamlit run enhanced_app.py")
    
    return len(models_trained) == 4

def create_model_summary():
    """Create a summary of all enhanced models"""
    
    summary = {
        "training_date": datetime.now().isoformat(),
        "models": {
            "crop_recommendation": {
                "file": "enhanced_crop_model.pkl",
                "features": "Multi-variety crop support with real-time environmental data",
                "capabilities": [
                    "40+ crop varieties prediction",
                    "Seasonal recommendations", 
                    "Environmental suitability analysis",
                    "Confidence scoring with explanations"
                ]
            },
            "disease_detection": {
                "file": "enhanced_disease_model.h5",
                "features": "Advanced CNN with environmental context",
                "capabilities": [
                    "6 disease categories with detailed info",
                    "Environmental factor integration",
                    "Treatment recommendations",
                    "Urgency assessment"
                ]
            },
            "soil_classification": {
                "file": "enhanced_soil_cnn.h5",
                "features": "Comprehensive soil analysis with property prediction",
                "capabilities": [
                    "5 soil types with detailed characteristics",
                    "Soil property prediction (pH, OM, etc.)",
                    "Management recommendations",
                    "Crop suitability mapping"
                ]
            },
            "yield_prediction": {
                "file": "enhanced_yield_models.pkl",
                "features": "Quantile regression with uncertainty estimation",
                "capabilities": [
                    "Uncertainty bounds (10th, 50th, 90th percentiles)",
                    "Multi-factor yield analysis",
                    "Management recommendations",
                    "Technology adoption impact"
                ]
            }
        },
        "enhancements": [
            "Real-time environmental data integration",
            "Location-based analysis for Indian states",
            "Variety-specific recommendations",
            "Comprehensive uncertainty quantification",
            "Advanced visualization and explanations",
            "Mobile-responsive web interface"
        ]
    }
    
    import json
    with open('models/model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📋 Model summary saved to: models/model_summary.json")

if __name__ == "__main__":
    print("Starting enhanced model training...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train all models
    success = train_all_enhanced_models()
    
    if success:
        create_model_summary()
        print("\n🎊 All enhanced models trained successfully!")
    else:
        print("\n⚠️  Some models failed to train. Check the error messages above.")
        
    print("\n" + "="*60)