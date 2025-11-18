"""
Enhanced Disease Detection Model with Real-time Analysis
Supports multiple plant diseases and real-time environmental factors
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class EnhancedDiseaseDetectionModel:
    def __init__(self):
        self.model = None
        self.disease_info = {}
        self.input_shape = (128, 128, 3)
        self.classes = []
        
    def setup_disease_database(self):
        """Setup comprehensive disease information database"""
        
        self.disease_info = {
            'Healthy': {
                'description': 'Healthy plant with no visible diseases',
                'symptoms': ['Normal green color', 'No spots or lesions', 'Good leaf structure'],
                'treatment': ['Continue regular care', 'Monitor for changes', 'Maintain good practices'],
                'prevention': ['Proper watering', 'Good air circulation', 'Regular inspection'],
                'severity': 'None',
                'environmental_factors': {
                    'temperature_range': [15, 35],
                    'humidity_range': [40, 70],
                    'light_requirement': 'Medium to High'
                }
            },
            'Rust': {
                'description': 'Fungal disease causing orange-red pustules on leaves',
                'symptoms': ['Orange-red pustules', 'Yellow spots', 'Leaf yellowing', 'Premature leaf drop'],
                'treatment': ['Fungicide application', 'Remove infected leaves', 'Improve air circulation'],
                'prevention': ['Avoid overhead watering', 'Plant resistant varieties', 'Proper spacing'],
                'severity': 'Moderate to High',
                'environmental_factors': {
                    'temperature_range': [20, 25],
                    'humidity_range': [70, 90],
                    'light_requirement': 'Medium'
                }
            },
            'Leaf_Spot': {
                'description': 'Bacterial or fungal disease causing spots on leaves',
                'symptoms': ['Dark spots on leaves', 'Yellow halos around spots', 'Leaf wilting'],
                'treatment': ['Copper-based fungicides', 'Remove infected parts', 'Reduce watering'],
                'prevention': ['Avoid wet foliage', 'Good drainage', 'Crop rotation'],
                'severity': 'Moderate',
                'environmental_factors': {
                    'temperature_range': [22, 28],
                    'humidity_range': [60, 85],
                    'light_requirement': 'Medium'
                }
            },
            'Blight': {
                'description': 'Severe disease causing rapid plant tissue death',
                'symptoms': ['Brown/black lesions', 'Rapid spread', 'Plant wilting', 'Stem cankers'],
                'treatment': ['Immediate fungicide treatment', 'Remove all infected plants', 'Soil treatment'],
                'prevention': ['Resistant varieties', 'Proper sanitation', 'Avoid overcrowding'],
                'severity': 'High',
                'environmental_factors': {
                    'temperature_range': [18, 24],
                    'humidity_range': [80, 95],
                    'light_requirement': 'Low to Medium'
                }
            },
            'Powdery_Mildew': {
                'description': 'Fungal disease creating white powdery coating',
                'symptoms': ['White powdery patches', 'Leaf distortion', 'Stunted growth'],
                'treatment': ['Sulfur-based fungicides', 'Baking soda spray', 'Improve ventilation'],
                'prevention': ['Avoid overhead watering', 'Proper spacing', 'Choose resistant varieties'],
                'severity': 'Moderate',
                'environmental_factors': {
                    'temperature_range': [20, 30],
                    'humidity_range': [50, 70],
                    'light_requirement': 'Medium to High'
                }
            },
            'Mosaic_Virus': {
                'description': 'Viral disease causing mottled leaf patterns',
                'symptoms': ['Mottled yellow-green patterns', 'Leaf curling', 'Stunted growth'],
                'treatment': ['Remove infected plants', 'Control insect vectors', 'No chemical cure'],
                'prevention': ['Use certified seeds', 'Control aphids/thrips', 'Quarantine new plants'],
                'severity': 'High',
                'environmental_factors': {
                    'temperature_range': [25, 35],
                    'humidity_range': [40, 60],
                    'light_requirement': 'High'
                }
            }
        }
        
        self.classes = list(self.disease_info.keys())
        
    def generate_synthetic_data(self, output_dir='data/synthetic/disease_images', samples_per_class=200):
        """Generate synthetic disease image data"""
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create base patterns for different diseases
        disease_patterns = {
            'Healthy': self._create_healthy_pattern,
            'Rust': self._create_rust_pattern,
            'Leaf_Spot': self._create_spot_pattern,
            'Blight': self._create_blight_pattern,
            'Powdery_Mildew': self._create_mildew_pattern,
            'Mosaic_Virus': self._create_mosaic_pattern
        }
        
        for disease, pattern_func in disease_patterns.items():
            disease_dir = os.path.join(output_dir, disease)
            if not os.path.exists(disease_dir):
                os.makedirs(disease_dir)
            
            for i in range(samples_per_class):
                img = pattern_func()
                img_path = os.path.join(disease_dir, f"{disease}_{i}.png")
                img.save(img_path)
        
        print(f"Generated synthetic disease data in {output_dir}")
    
    def _create_healthy_pattern(self):
        """Create healthy leaf pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))  # Forest green
        pixels = np.array(img)
        
        # Add natural variation
        noise = np.random.normal(0, 10, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(pixels)
    
    def _create_rust_pattern(self):
        """Create rust disease pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))
        pixels = np.array(img)
        
        # Add rust spots
        for _ in range(np.random.randint(5, 15)):
            x, y = np.random.randint(10, 118), np.random.randint(10, 118)
            size = np.random.randint(3, 8)
            
            # Create rust-colored spots
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if x+dx < 128 and y+dy < 128 and dx*dx + dy*dy <= size*size:
                        pixels[y+dy, x+dx] = [205, 92, 92]  # Indian red
        
        # Add yellow halos
        noise = np.random.normal(0, 15, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(pixels)
    
    def _create_spot_pattern(self):
        """Create leaf spot pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))
        pixels = np.array(img)
        
        # Add dark spots
        for _ in range(np.random.randint(3, 10)):
            x, y = np.random.randint(5, 123), np.random.randint(5, 123)
            size = np.random.randint(2, 6)
            
            # Dark brown spots
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if x+dx < 128 and y+dy < 128 and dx*dx + dy*dy <= size*size:
                        pixels[y+dy, x+dx] = [101, 67, 33]  # Dark brown
        
        return Image.fromarray(pixels)
    
    def _create_blight_pattern(self):
        """Create blight pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))
        pixels = np.array(img)
        
        # Add large dark lesions
        for _ in range(np.random.randint(2, 6)):
            x, y = np.random.randint(10, 118), np.random.randint(10, 118)
            size = np.random.randint(8, 15)
            
            # Dark lesions
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if x+dx < 128 and y+dy < 128 and dx*dx + dy*dy <= size*size:
                        pixels[y+dy, x+dx] = [47, 79, 79]  # Dark slate gray
        
        return Image.fromarray(pixels)
    
    def _create_mildew_pattern(self):
        """Create powdery mildew pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))
        pixels = np.array(img)
        
        # Add white powdery areas
        for _ in range(np.random.randint(3, 8)):
            x, y = np.random.randint(10, 118), np.random.randint(10, 118)
            size = np.random.randint(5, 12)
            
            # White powdery patches
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    if x+dx < 128 and y+dy < 128 and dx*dx + dy*dy <= size*size:
                        alpha = 0.7 * (1 - (dx*dx + dy*dy) / (size*size))
                        pixels[y+dy, x+dx] = (1-alpha) * pixels[y+dy, x+dx] + alpha * np.array([240, 240, 240])
        
        return Image.fromarray(pixels.astype(np.uint8))
    
    def _create_mosaic_pattern(self):
        """Create mosaic virus pattern"""
        img = Image.new('RGB', (128, 128), color=(34, 139, 34))
        pixels = np.array(img)
        
        # Add mottled yellow-green patterns
        for i in range(0, 128, 8):
            for j in range(0, 128, 8):
                if np.random.random() > 0.5:
                    # Yellow patches
                    pixels[j:j+8, i:i+8] = [154, 205, 50]  # Yellow green
        
        return Image.fromarray(pixels)
    
    def create_enhanced_model(self):
        """Create enhanced CNN model for disease detection"""
        
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
            Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data_dir='data/synthetic/disease_images', epochs=50):
        """Train the enhanced disease detection model"""
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = test_datagen.flow_from_directory(
            data_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Create model
        self.model = self.create_enhanced_model()
        
        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        return history
    
    def predict_disease(self, image_path, environmental_data=None):
        """Predict disease with environmental context"""
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img_array)[0]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        
        results = []
        for idx in top_indices:
            disease = self.classes[idx]
            confidence = float(predictions[idx])
            
            # Environmental suitability check
            env_factor = 1.0
            if environmental_data and disease in self.disease_info:
                env_factor = self._calculate_environmental_suitability(disease, environmental_data)
            
            adjusted_confidence = confidence * env_factor
            
            results.append({
                'disease': disease,
                'confidence': confidence,
                'adjusted_confidence': adjusted_confidence,
                'environmental_factor': env_factor,
                'info': self.disease_info.get(disease, {}),
                'treatment_urgency': self._get_treatment_urgency(disease, confidence)
            })
        
        return results
    
    def _calculate_environmental_suitability(self, disease, env_data):
        """Calculate how environmental conditions affect disease probability"""
        
        if disease not in self.disease_info:
            return 1.0
        
        disease_env = self.disease_info[disease]['environmental_factors']
        
        suitability = 1.0
        
        # Temperature factor
        if 'temperature' in env_data:
            temp = env_data['temperature']
            temp_range = disease_env['temperature_range']
            if temp_range[0] <= temp <= temp_range[1]:
                temp_factor = 1.2  # Favorable for disease
            else:
                deviation = min(abs(temp - temp_range[0]), abs(temp - temp_range[1]))
                temp_factor = max(0.3, 1.0 - deviation / 15)
            suitability *= temp_factor
        
        # Humidity factor
        if 'humidity' in env_data:
            humidity = env_data['humidity']
            humidity_range = disease_env['humidity_range']
            if humidity_range[0] <= humidity <= humidity_range[1]:
                humidity_factor = 1.3  # Very favorable for disease
            else:
                deviation = min(abs(humidity - humidity_range[0]), abs(humidity - humidity_range[1]))
                humidity_factor = max(0.2, 1.0 - deviation / 30)
            suitability *= humidity_factor
        
        return min(suitability, 2.0)  # Cap at 2x
    
    def _get_treatment_urgency(self, disease, confidence):
        """Determine treatment urgency based on disease and confidence"""
        
        if confidence < 0.3:
            return "Monitor"
        elif disease == 'Healthy':
            return "None"
        elif disease in ['Blight', 'Mosaic_Virus']:
            return "Urgent" if confidence > 0.7 else "High"
        elif disease in ['Rust', 'Leaf_Spot']:
            return "Moderate" if confidence > 0.6 else "Low"
        else:
            return "Low"
    
    def save_model(self, model_path='models/enhanced_disease_model.h5'):
        """Save the trained model and metadata"""
        
        self.model.save(model_path)
        
        # Save disease information
        with open('models/disease_info.json', 'w') as f:
            json.dump(self.disease_info, f, indent=2)
        
        # Save class names
        with open('models/disease_classes.json', 'w') as f:
            json.dump({'classes': self.classes}, f, indent=2)
        
        print(f"Enhanced disease model saved to {model_path}")

def main():
    """Train and save the enhanced disease detection model"""
    
    print("Training Enhanced Disease Detection Model...")
    print("=" * 50)
    
    # Initialize model
    model = EnhancedDiseaseDetectionModel()
    
    # Setup disease database
    model.setup_disease_database()
    print(f"Disease database setup with {len(model.classes)} classes")
    
    # Generate synthetic data
    print("Generating synthetic disease data...")
    model.generate_synthetic_data()
    
    # Train model
    print("Training disease detection model...")
    history = model.train_model(epochs=30)
    
    # Save model
    print("Saving enhanced disease model...")
    model.save_model()
    
    print("Enhanced disease detection model training completed!")
    return model

# Export function for external use
def train_enhanced_disease_model():
    """External function to train the enhanced disease model"""
    return main()

if __name__ == "__main__":
    model = main()