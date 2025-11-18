"""
Real-time sensor data processor and validator
Handles live sensor inputs and data quality checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

class SensorDataProcessor:
    def __init__(self):
        self.sensor_ranges = {
            'Moisture_vol_pct': (0, 50),
            'pH': (3.0, 10.0),
            'Nitrogen_ppm': (0, 800),
            'Phosphorus_ppm': (0, 200),
            'Potassium_ppm': (0, 500),
            'Soil_Temperature_C': (-5, 45),
            'Ambient_Humidity_pct': (0, 100),
            'EC_dS_per_m': (0.0, 5.0),
            'HSV_mean_H': (0, 360),
            'HSV_mean_S': (0, 255),
            'Texture_Contrast': (0, 1),
            'Texture_Entropy': (0, 2)
        }
    
    def validate_sensor_data(self, data):
        """Validate sensor data against expected ranges"""
        validation_results = {}
        warnings = []
        
        for sensor, value in data.items():
            if sensor in self.sensor_ranges:
                min_val, max_val = self.sensor_ranges[sensor]
                if min_val <= value <= max_val:
                    validation_results[sensor] = 'valid'
                else:
                    validation_results[sensor] = 'out_of_range'
                    warnings.append(f"{sensor}: {value} is outside normal range ({min_val}-{max_val})")
            else:
                validation_results[sensor] = 'unknown_sensor'
                warnings.append(f"Unknown sensor: {sensor}")
        
        return validation_results, warnings
    
    def clean_sensor_data(self, data):
        """Clean and normalize sensor data"""
        cleaned_data = data.copy()
        
        # Handle missing values with reasonable defaults
        defaults = {
            'Moisture_vol_pct': 25.0,
            'pH': 6.5,
            'Nitrogen_ppm': 100,
            'Phosphorus_ppm': 50,
            'Potassium_ppm': 150,
            'Soil_Temperature_C': 20.0,
            'Ambient_Humidity_pct': 60.0,
            'EC_dS_per_m': 1.0,
            'HSV_mean_H': 30.0,
            'HSV_mean_S': 80.0,
            'Texture_Contrast': 0.5,
            'Texture_Entropy': 0.8
        }
        
        for sensor, default_val in defaults.items():
            if sensor not in cleaned_data or cleaned_data[sensor] is None:
                cleaned_data[sensor] = default_val
        
        # Clip values to valid ranges
        for sensor, (min_val, max_val) in self.sensor_ranges.items():
            if sensor in cleaned_data:
                cleaned_data[sensor] = np.clip(cleaned_data[sensor], min_val, max_val)
        
        return cleaned_data
    
    def process_real_time_input(self, sensor_input, soil_type):
        """Process real-time sensor input for crop recommendation"""
        # Validate input
        validation_results, warnings = self.validate_sensor_data(sensor_input)
        
        # Clean data
        cleaned_data = self.clean_sensor_data(sensor_input)
        
        # Add soil type one-hot encoding
        soil_types = ['Clay', 'Loam', 'Sandy']
        for soil in soil_types:
            cleaned_data[f'soil_{soil}'] = 1 if soil_type == soil else 0
        
        # Add derived features (matching advanced model)
        cleaned_data['NPK_ratio'] = cleaned_data['Nitrogen_ppm'] / (
            cleaned_data['Phosphorus_ppm'] + cleaned_data['Potassium_ppm'] + 1e-6
        )
        cleaned_data['nutrient_balance'] = np.sqrt(
            cleaned_data['Nitrogen_ppm']**2 + 
            cleaned_data['Phosphorus_ppm']**2 + 
            cleaned_data['Potassium_ppm']**2
        )
        cleaned_data['moisture_pH_interaction'] = cleaned_data['Moisture_vol_pct'] * cleaned_data['pH']
        cleaned_data['temp_humidity_index'] = cleaned_data['Soil_Temperature_C'] * cleaned_data['Ambient_Humidity_pct'] / 100
        cleaned_data['texture_complexity'] = cleaned_data['Texture_Contrast'] * cleaned_data['Texture_Entropy']
        
        return cleaned_data, validation_results, warnings

class DataLogger:
    def __init__(self, log_file='sensor_logs.json'):
        self.log_file = Path(log_file)
    
    def log_prediction(self, sensor_data, prediction, confidence, timestamp=None):
        """Log prediction results for analysis"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'sensor_data': sensor_data,
            'prediction': prediction,
            'confidence': confidence
        }
        
        # Load existing logs
        logs = []
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Sanitize log entry to ensure JSON serializability (convert numpy types)
        def _sanitize(obj):
            # Convert numpy and other non-serializable types to native Python types
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            try:
                import numpy as _np
                if isinstance(obj, _np.integer):
                    return int(obj)
                if isinstance(obj, _np.floating):
                    return float(obj)
                if isinstance(obj, _np.ndarray):
                    return _sanitize(obj.tolist())
            except Exception:
                pass
            # datetime, strings, ints, floats are ok
            return obj

        logs.append(_sanitize(log_entry))
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save logs
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_recent_predictions(self, limit=10):
        """Get recent prediction logs"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            return logs[-limit:]
        except:
            return []

def simulate_sensor_reading():
    """Simulate realistic sensor readings for testing"""
    np.random.seed()
    
    base_readings = {
        'Moisture_vol_pct': np.random.uniform(15, 40),
        'pH': np.random.uniform(5.5, 8.0),
        'Nitrogen_ppm': np.random.uniform(80, 400),
        'Phosphorus_ppm': np.random.uniform(20, 120),
        'Potassium_ppm': np.random.uniform(50, 300),
        'Soil_Temperature_C': np.random.uniform(16, 35),
        'Ambient_Humidity_pct': np.random.uniform(40, 85),
        'EC_dS_per_m': np.random.uniform(0.5, 2.5),
        'HSV_mean_H': np.random.uniform(10, 40),
        'HSV_mean_S': np.random.uniform(60, 120),
        'Texture_Contrast': np.random.uniform(0.3, 0.9),
        'Texture_Entropy': np.random.uniform(0.5, 1.2)
    }
    
    soil_type = np.random.choice(['Clay', 'Loam', 'Sandy'])
    
    return base_readings, soil_type

if __name__ == '__main__':
    # Test the sensor data processor
    processor = SensorDataProcessor()
    logger = DataLogger()
    
    # Simulate sensor reading
    sensor_data, soil_type = simulate_sensor_reading()
    print("Simulated sensor reading:")
    for sensor, value in sensor_data.items():
        print(f"  {sensor}: {value:.2f}")
    print(f"  Soil Type: {soil_type}")
    
    # Process the data
    processed_data, validation, warnings = processor.process_real_time_input(sensor_data, soil_type)
    
    print("\nValidation results:")
    for sensor, status in validation.items():
        print(f"  {sensor}: {status}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("\nProcessed data ready for model prediction!")