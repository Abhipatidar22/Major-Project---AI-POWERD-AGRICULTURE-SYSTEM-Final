"""
Enhanced Soil Classification Model with Real-time Analysis
Supports detailed soil properties and real-time environmental factors
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image, ImageDraw
import os
import json
import joblib
from datetime import datetime

class EnhancedSoilClassificationModel:
    def __init__(self):
        self.cnn_model = None
        self.property_models = {}
        self.scaler = StandardScaler()
        self.soil_database = {}
        self.input_shape = (128, 128, 3)
        self.soil_classes = []
        
    def setup_soil_database(self):
        """Setup comprehensive soil information database"""
        
        self.soil_database = {
            'Clay': {
                'description': 'Fine-textured soil with excellent water retention',
                'particle_size': '< 0.002 mm',
                'water_retention': 'High',
                'drainage': 'Poor',
                'fertility': 'High',
                'ph_range': [6.0, 8.5],
                'organic_matter_range': [2.5, 5.0],
                'nutrient_availability': {
                    'Nitrogen': 'High',
                    'Phosphorus': 'Medium',
                    'Potassium': 'High'
                },
                'suitable_crops': ['Rice', 'Wheat', 'Cotton', 'Sugarcane'],
                'management_practices': [
                    'Improve drainage with organic matter',
                    'Avoid working when wet',
                    'Add gypsum for structure improvement',
                    'Use raised beds in wet areas'
                ],
                'color_characteristics': {
                    'typical_colors': [(139, 69, 19), (160, 82, 45), (210, 180, 140)],
                    'moisture_variations': 'Darker when wet'
                }
            },
            'Sandy': {
                'description': 'Coarse-textured soil with good drainage but low water retention',
                'particle_size': '0.05 - 2.0 mm',
                'water_retention': 'Low',
                'drainage': 'Excellent',
                'fertility': 'Low to Medium',
                'ph_range': [5.5, 7.5],
                'organic_matter_range': [0.5, 2.5],
                'nutrient_availability': {
                    'Nitrogen': 'Low',
                    'Phosphorus': 'Low',
                    'Potassium': 'Medium'
                },
                'suitable_crops': ['Millets', 'Groundnut', 'Watermelon', 'Carrots'],
                'management_practices': [
                    'Add organic matter regularly',
                    'Use drip irrigation',
                    'Apply frequent light fertilization',
                    'Plant cover crops'
                ],
                'color_characteristics': {
                    'typical_colors': [(255, 218, 185), (222, 184, 135), (245, 245, 220)],
                    'moisture_variations': 'Slightly darker when wet'
                }
            },
            'Loam': {
                'description': 'Ideal soil with balanced properties',
                'particle_size': 'Mixed (Sand, Silt, Clay)',
                'water_retention': 'Good',
                'drainage': 'Good',
                'fertility': 'High',
                'ph_range': [6.0, 7.5],
                'organic_matter_range': [3.0, 6.0],
                'nutrient_availability': {
                    'Nitrogen': 'High',
                    'Phosphorus': 'High',
                    'Potassium': 'High'
                },
                'suitable_crops': ['Most crops', 'Vegetables', 'Fruits', 'Grains'],
                'management_practices': [
                    'Maintain organic matter',
                    'Regular soil testing',
                    'Balanced fertilization',
                    'Crop rotation'
                ],
                'color_characteristics': {
                    'typical_colors': [(101, 67, 33), (139, 90, 43), (165, 42, 42)],
                    'moisture_variations': 'Noticeably darker when wet'
                }
            },
            'Silt': {
                'description': 'Medium-textured soil with good water and nutrient retention',
                'particle_size': '0.002 - 0.05 mm',
                'water_retention': 'High',
                'drainage': 'Moderate',
                'fertility': 'High',
                'ph_range': [6.0, 7.8],
                'organic_matter_range': [2.0, 4.5],
                'nutrient_availability': {
                    'Nitrogen': 'High',
                    'Phosphorus': 'High',
                    'Potassium': 'Medium'
                },
                'suitable_crops': ['Soybeans', 'Corn', 'Small grains', 'Pasture'],
                'management_practices': [
                    'Prevent compaction',
                    'Improve drainage if needed',
                    'Control erosion',
                    'Add organic matter'
                ],
                'color_characteristics': {
                    'typical_colors': [(119, 136, 153), (128, 128, 128), (169, 169, 169)],
                    'moisture_variations': 'Much darker when wet'
                }
            },
            'Peat': {
                'description': 'Organic soil with very high organic matter content',
                'particle_size': 'Organic matter',
                'water_retention': 'Very High',
                'drainage': 'Poor',
                'fertility': 'Variable',
                'ph_range': [3.5, 6.0],
                'organic_matter_range': [20.0, 80.0],
                'nutrient_availability': {
                    'Nitrogen': 'Very High',
                    'Phosphorus': 'Low',
                    'Potassium': 'Low'
                },
                'suitable_crops': ['Cranberries', 'Blueberries', 'Specialty crops'],
                'management_practices': [
                    'Improve drainage',
                    'Add lime to reduce acidity',
                    'Supplement P and K',
                    'Prevent subsidence'
                ],
                'color_characteristics': {
                    'typical_colors': [(64, 64, 64), (47, 79, 79), (25, 25, 25)],
                    'moisture_variations': 'Very dark, almost black when wet'
                }
            }
        }
        
        self.soil_classes = list(self.soil_database.keys())
    
    def generate_synthetic_soil_data(self, output_dir='data/synthetic/soil_images_enhanced', samples_per_class=300):
        """Generate enhanced synthetic soil image data"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for soil_type in self.soil_classes:
            soil_dir = os.path.join(output_dir, soil_type.lower())
            if not os.path.exists(soil_dir):
                os.makedirs(soil_dir)
            
            soil_info = self.soil_database[soil_type]
            
            for i in range(samples_per_class):
                img = self._create_soil_pattern(soil_type, soil_info)
                img_path = os.path.join(soil_dir, f"{soil_type.lower()}_{i}.png")
                img.save(img_path)
        
        print(f"Generated enhanced soil data in {output_dir}")
    
    def _create_soil_pattern(self, soil_type, soil_info):
        """Create realistic soil texture patterns"""
        
        img = Image.new('RGB', (128, 128))
        draw = ImageDraw.Draw(img)
        
        # Get base colors for soil type
        base_colors = soil_info['color_characteristics']['typical_colors']
        base_color = base_colors[np.random.randint(0, len(base_colors))]
        
        # Fill with base color
        draw.rectangle([0, 0, 128, 128], fill=base_color)
        
        pixels = np.array(img)
        
        # Add texture based on soil type
        if soil_type == 'Clay':
            # Fine, smooth texture with some cracks
            pixels = self._add_clay_texture(pixels)
        elif soil_type == 'Sandy':
            # Grainy, coarse texture
            pixels = self._add_sandy_texture(pixels)
        elif soil_type == 'Loam':
            # Mixed texture
            pixels = self._add_loam_texture(pixels)
        elif soil_type == 'Silt':
            # Smooth, fine texture
            pixels = self._add_silt_texture(pixels)
        elif soil_type == 'Peat':
            # Very dark, fibrous texture
            pixels = self._add_peat_texture(pixels)
        
        # Add moisture variation
        if np.random.random() > 0.5:
            pixels = self._add_moisture_effect(pixels, soil_type)
        
        # Add some random variation
        noise = np.random.normal(0, 8, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(pixels)
    
    def _add_clay_texture(self, pixels):
        """Add clay-specific texture patterns"""
        # Add some cracks
        for _ in range(np.random.randint(2, 6)):
            x1, y1 = np.random.randint(0, 128), np.random.randint(0, 128)
            x2, y2 = np.random.randint(0, 128), np.random.randint(0, 128)
            
            # Draw crack line
            length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
            for i in range(length):
                x = int(x1 + i * (x2-x1) / length)
                y = int(y1 + i * (y2-y1) / length)
                if 0 <= x < 128 and 0 <= y < 128:
                    pixels[y, x] = pixels[y, x] * 0.7  # Darker cracks
        
        return pixels
    
    def _add_sandy_texture(self, pixels):
        """Add sandy texture with grain patterns"""
        # Add granular texture
        for _ in range(np.random.randint(20, 40)):
            x, y = np.random.randint(0, 128), np.random.randint(0, 128)
            size = np.random.randint(1, 3)
            
            # Create sand grains
            color_variation = np.random.randint(-20, 20, 3)
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if 0 <= x+dx < 128 and 0 <= y+dy < 128:
                        pixels[y+dy, x+dx] = np.clip(pixels[y+dy, x+dx] + color_variation, 0, 255)
        
        return pixels
    
    def _add_loam_texture(self, pixels):
        """Add balanced loam texture"""
        # Mix of clay and sandy features
        pixels = self._add_clay_texture(pixels)
        pixels = self._add_sandy_texture(pixels)
        
        # Add organic matter spots
        for _ in range(np.random.randint(5, 12)):
            x, y = np.random.randint(5, 123), np.random.randint(5, 123)
            size = np.random.randint(2, 5)
            
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if 0 <= x+dx < 128 and 0 <= y+dy < 128 and dx*dx + dy*dy <= size*size:
                        pixels[y+dy, x+dx] = pixels[y+dy, x+dx] * 0.8  # Darker organic spots
        
        return pixels
    
    def _add_silt_texture(self, pixels):
        """Add fine silt texture"""
        # Very fine, smooth texture with minimal variation
        smooth_noise = np.random.normal(0, 5, pixels.shape)
        pixels = pixels + smooth_noise
        return pixels
    
    def _add_peat_texture(self, pixels):
        """Add organic peat texture"""
        # Very dark with fibrous patterns
        pixels = pixels * 0.3  # Much darker
        
        # Add fibrous organic matter
        for _ in range(np.random.randint(10, 20)):
            x1, y1 = np.random.randint(0, 128), np.random.randint(0, 128)
            length = np.random.randint(10, 30)
            angle = np.random.uniform(0, 2*np.pi)
            
            for i in range(length):
                x = int(x1 + i * np.cos(angle))
                y = int(y1 + i * np.sin(angle))
                if 0 <= x < 128 and 0 <= y < 128:
                    pixels[y, x] = np.minimum(pixels[y, x] + 30, 255)  # Lighter fibers
        
        return pixels
    
    def _add_moisture_effect(self, pixels, soil_type):
        """Add moisture effects to soil"""
        moisture_factor = 0.7 if soil_type in ['Clay', 'Peat'] else 0.85
        return pixels * moisture_factor
    
    def create_enhanced_cnn_model(self):
        """Create enhanced CNN model for soil classification"""
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.soil_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_property_prediction_models(self):
        """Train models to predict soil properties from environmental data"""
        
        # Generate training data for property prediction
        n_samples = 2000
        
        # Environmental features
        features = []
        properties = {
            'ph': [],
            'organic_matter': [],
            'moisture_content': [],
            'bulk_density': [],
            'cation_exchange_capacity': []
        }
        
        for soil_type in self.soil_classes:
            soil_info = self.soil_database[soil_type]
            
            for _ in range(n_samples // len(self.soil_classes)):
                # Environmental features
                temperature = np.random.normal(25, 8)
                humidity = np.random.normal(65, 20)
                rainfall = np.random.exponential(100)
                elevation = np.random.uniform(0, 3000)
                
                # Soil type encoding
                soil_encoding = [1 if soil_type == s else 0 for s in self.soil_classes]
                
                feature_row = [temperature, humidity, rainfall, elevation] + soil_encoding
                features.append(feature_row)
                
                # Generate property values based on soil type
                ph_range = soil_info['ph_range']
                om_range = soil_info['organic_matter_range']
                
                properties['ph'].append(np.random.uniform(ph_range[0], ph_range[1]))
                properties['organic_matter'].append(np.random.uniform(om_range[0], om_range[1]))
                properties['moisture_content'].append(np.random.uniform(5, 40))
                properties['bulk_density'].append(np.random.uniform(0.8, 1.8))
                properties['cation_exchange_capacity'].append(np.random.uniform(5, 50))
        
        features = np.array(features)
        features_scaled = self.scaler.fit_transform(features)
        
        # Train individual models for each property
        for prop_name, prop_values in properties.items():
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, prop_values, test_size=0.2, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.property_models[prop_name] = model
            print(f"{prop_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
    
    def train_cnn_model(self, data_dir='data/synthetic/soil_images_enhanced', epochs=50):
        """Train the CNN model for soil classification"""
        
        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Create and train model
        self.cnn_model = self.create_enhanced_cnn_model()
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.cnn_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        return history
    
    def predict_comprehensive_soil_analysis(self, image_path, environmental_data=None):
        """Comprehensive soil analysis including type and properties"""
        
        # Image classification
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = self.cnn_model.predict(img_array)[0]
        top_idx = np.argmax(predictions)
        predicted_soil = self.soil_classes[top_idx]
        confidence = float(predictions[top_idx])
        
        result = {
            'soil_type': predicted_soil,
            'confidence': confidence,
            'soil_info': self.soil_database[predicted_soil],
            'predicted_properties': {}
        }
        
        # Predict soil properties if environmental data is provided
        if environmental_data and self.property_models:
            # Prepare features
            temp = environmental_data.get('temperature', 25)
            humidity = environmental_data.get('humidity', 65)
            rainfall = environmental_data.get('rainfall', 100)
            elevation = environmental_data.get('elevation', 500)
            
            # Soil type encoding
            soil_encoding = [1 if predicted_soil == s else 0 for s in self.soil_classes]
            features = np.array([[temp, humidity, rainfall, elevation] + soil_encoding])
            features_scaled = self.scaler.transform(features)
            
            # Predict properties
            for prop_name, model in self.property_models.items():
                predicted_value = model.predict(features_scaled)[0]
                result['predicted_properties'][prop_name] = float(predicted_value)
        
        return result
    
    def save_models(self, cnn_path='models/enhanced_soil_cnn.h5', properties_path='models/soil_properties.pkl'):
        """Save all trained models"""
        
        # Save CNN model
        self.cnn_model.save(cnn_path)
        
        # Save property models and metadata
        model_data = {
            'property_models': self.property_models,
            'scaler': self.scaler,
            'soil_database': self.soil_database,
            'soil_classes': self.soil_classes,
            'trained_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, properties_path)
        
        # Save soil database as JSON
        with open('models/soil_database.json', 'w') as f:
            json.dump(self.soil_database, f, indent=2)
        
        print(f"Enhanced soil models saved")

def main():
    """Train and save enhanced soil classification models"""
    
    print("Training Enhanced Soil Classification Models...")
    print("=" * 50)
    
    # Initialize model
    model = EnhancedSoilClassificationModel()
    
    # Setup soil database
    model.setup_soil_database()
    print(f"Soil database setup with {len(model.soil_classes)} classes")
    
    # Generate synthetic data
    print("Generating enhanced synthetic soil data...")
    model.generate_synthetic_soil_data()
    
    # Train property prediction models
    print("Training soil property prediction models...")
    model.train_property_prediction_models()
    
    # Train CNN model
    print("Training CNN model for soil classification...")
    history = model.train_cnn_model(epochs=30)
    
    # Save models
    print("Saving enhanced soil models...")
    model.save_models()
    
    print("Enhanced soil classification model training completed!")
    return model

# Export function for external use
def train_enhanced_soil_model():
    """External function to train the enhanced soil model"""
    return main()

if __name__ == "__main__":
    model = main()