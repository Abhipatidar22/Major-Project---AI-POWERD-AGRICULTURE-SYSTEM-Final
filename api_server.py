"""FastAPI server exposing AI Agriculture functionality.
Run: uvicorn api_server:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, os, io
from typing import Optional, List
from pathlib import Path
import numpy as np
from PIL import Image

# Import existing logic
from src.location_analysis import IndianLocationAnalyzer

MODELS_DIR = Path("models")

# Attempt to load enhanced crop model
crop_model_data = None
crop_model = None
scaler = None
label_encoder = None
feature_names: List[str] = []
classes: List[str] = []
model_load_error: Optional[str] = None

if (MODELS_DIR / "enhanced_crop_model.pkl").exists():
    try:
        crop_model_data = joblib.load(MODELS_DIR / "enhanced_crop_model.pkl")
        crop_model = crop_model_data.get("model")
        scaler = crop_model_data.get("scaler")
        label_encoder = crop_model_data.get("label_encoder")
        feature_names = crop_model_data.get("feature_names", [])
        classes = crop_model_data.get("classes", []) or list(getattr(label_encoder, "classes_", []))
    except Exception as e:
        model_load_error = str(e)
else:
    model_load_error = "Model file not found"

location_analyzer = IndianLocationAnalyzer()

# Real sensor reading (shared state). Will be updated by MQTT subscriber if configured.
latest_sensor_reading = {
    "moisture": 25.0,
    "temperature": 26.0,
    "humidity": 65.0,
    "nitrogen": 120.0,
    "timestamp": __import__('datetime').datetime.utcnow().isoformat()+"Z",
    "source": "synthetic"
}

MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "agri/sensors")
_mqtt_started = False

def _start_mqtt_if_configured():
    global _mqtt_started
    if _mqtt_started or not MQTT_HOST:
        return
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        return
    client = mqtt.Client()
    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            c.subscribe(MQTT_TOPIC)
        else:
            print("MQTT connect failed", rc)
    def on_message(c, userdata, msg):
        import json
        try:
            payload = json.loads(msg.payload.decode())
            # Expect keys: moisture, temperature, humidity, nitrogen
            for k in ["moisture","temperature","humidity","nitrogen"]:
                if k in payload and isinstance(payload[k], (int,float)):
                    latest_sensor_reading[k] = float(payload[k])
            latest_sensor_reading["timestamp"] = __import__('datetime').datetime.utcnow().isoformat()+"Z"
            latest_sensor_reading["source"] = "mqtt"
        except Exception as e:
            print("MQTT message parse error", e)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(MQTT_HOST, MQTT_PORT, 60)
        client.loop_start()
        _mqtt_started = True
        print(f"MQTT connected to {MQTT_HOST}:{MQTT_PORT} topic={MQTT_TOPIC}")
    except Exception as e:
        print("MQTT connection failed", e)

_start_mqtt_if_configured()

app = FastAPI(title="AI Powered Agriculture API", description="REST API for Soil Analysis, Crop Recommendation, Yield Prediction, Disease Detection, Location Analysis.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class SoilAnalysisIn(BaseModel):
    moisture: float
    ph: float
    nitrogen: Optional[float] = None

class SoilAnalysisOut(BaseModel):
    status: str
    recommendation: str

class CropRecIn(BaseModel):
    state: str
    season: str
    soilType: Optional[str] = None

class CropRecItem(BaseModel):
    name: str
    score: float

class CropRecOut(BaseModel):
    crops: List[CropRecItem]
    explanation: Optional[str] = None

class YieldPredIn(BaseModel):
    crop: str
    area: float
    irrigation: Optional[bool] = None

class YieldPredOut(BaseModel):
    p10: float
    p50: float
    p90: float
    units: str = "tons"

class LocationOut(BaseModel):
    state: str
    rainfall: float
    soil: str
    advisories: List[str]

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class FeatureImportanceOut(BaseModel):
    items: List[FeatureImportanceItem]

class DetailedCropItem(BaseModel):
    crop: str
    variety: str
    full_name: str
    confidence: float
    suitability_score: float

class DetailedCropRecOut(BaseModel):
    items: List[DetailedCropItem]
    explanation: str

class StatesOut(BaseModel):
    states: List[str]

class StateAnalysisOut(BaseModel):
    location: str
    overall_score: float
    fertility: str
    dominant_soil: str
    major_crops: List[str]
    recommendations: List[str]

class SensorReadingOut(BaseModel):
    moisture: float
    temperature: float
    humidity: float
    nitrogen: float
    timestamp: str

class DashboardMetricsOut(BaseModel):
    metrics: dict
    recommendations: List[str]
    state: str

class SamplePredictionItem(BaseModel):
    full_name: str
    confidence: float
    suitability_score: float

class SamplePredictionOut(BaseModel):
    items: List[SamplePredictionItem]
    model_loaded: bool
    error: Optional[str] = None

@app.get("/")
def root():
    return {
        "message": "AI Powered Agriculture API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "crop_recommendation": "/crop-recommendation",
            "soil_analysis": "/soil-analysis",
            "yield_prediction": "/yield-prediction",
            "disease_detection": "/disease-detection",
            "states": "/states",
            "location_analysis": "/location-analysis/{state}"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": crop_model is not None, "error": model_load_error}

@app.post("/soil-analysis", response_model=SoilAnalysisOut)
def soil_analysis(payload: SoilAnalysisIn):
    # Simple heuristics
    moisture_status = "Optimal" if 15 <= payload.moisture <= 35 else "Low" if payload.moisture < 15 else "High"
    ph_status = "Optimal" if 6.0 <= payload.ph <= 7.5 else "Acidic" if payload.ph < 6.0 else "Alkaline"
    nitrogen_status = "OK" if (payload.nitrogen is None or 80 <= payload.nitrogen <= 200) else "Low" if payload.nitrogen < 80 else "High"
    status = f"Moisture:{moisture_status} pH:{ph_status} N:{nitrogen_status}"
    recs = []
    if moisture_status == "Low": recs.append("Irrigate or improve soil moisture retention")
    if moisture_status == "High": recs.append("Improve drainage to prevent root rot")
    if ph_status == "Acidic": recs.append("Apply agricultural lime")
    if ph_status == "Alkaline": recs.append("Use sulfur or organic matter to lower pH")
    if nitrogen_status == "Low": recs.append("Apply nitrogen fertilizer")
    if nitrogen_status == "High": recs.append("Reduce nitrogen inputs to avoid toxicity")
    if not recs: recs.append("Soil conditions appear within optimal ranges")
    return SoilAnalysisOut(status=status, recommendation="; ".join(recs))

@app.post("/crop-recommendation", response_model=CropRecOut)
def crop_recommendation(payload: CropRecIn):
    state = payload.state
    season = payload.season.lower()
    explanation = "Scores derived from model probabilities (synthetic features)" if crop_model else "Model unavailable; static heuristic results"
    if crop_model and scaler and feature_names and classes:
        # Build synthetic feature vector with reasonable defaults
        # Map season to one-hot
        season_kharif = 1 if season.startswith("kharif") else 0
        season_rabi = 1 if season.startswith("rabi") else 0
        season_zaid = 1 if season.startswith("zaid") or season.startswith("spring") else 0
        # Basic synthetic averages (could be improved with real sensor/state data)
        feat_map = {
            'Temperature': 26,
            'Humidity': 65,
            'pH': 6.8,
            'Nitrogen': 120,
            'Phosphorus': 60,
            'Potassium': 100,
            'Rainfall': 800,
            'Moisture': 25,
            'EC': 1.2,
            'Latitude': location_analyzer.indian_states.get(state, {'lat':25})['lat'],
            'Longitude': location_analyzer.indian_states.get(state, {'lon':80})['lon'],
            'Wind_Speed': 10,
            'Solar_Radiation': 20,
            'Day_Length': 12,
            'Season_Kharif': season_kharif,
            'Season_Rabi': season_rabi,
            'Season_Zaid': season_zaid
        }
        feature_vector = [feat_map.get(fn, 0) for fn in feature_names]
        scaled = scaler.transform([feature_vector])
        probs = crop_model.predict_proba(scaled)[0]
        top_idx = np.argsort(probs)[::-1][:6]
        crops = [CropRecItem(name=classes[i], score=float(round(probs[i],4))) for i in top_idx]
    else:
        # Fallback static suggestions based on season
        season_map = {
            'kharif': ["Rice_Basmati", "Cotton_Bt Cotton", "Rice_Kharif Rice"],
            'rabi': ["Wheat_Durum", "Wheat_Winter Wheat", "Rice_Rabi Rice"],
            'zaid': ["Wheat_Spring Wheat", "Rice_Jasmine", "Cotton_Hybrid Cotton"],
        }
        picks = season_map.get(season, ["Rice_Basmati", "Wheat_Durum", "Cotton_Bt Cotton"])
        crops = [CropRecItem(name=n, score=0.5) for n in picks]
    return CropRecOut(crops=crops, explanation=explanation)

@app.post("/yield-prediction", response_model=YieldPredOut)
def yield_prediction(payload: YieldPredIn):
    # Simple synthetic quantile estimates
    base = max(payload.area, 0.1)
    crop_factor = 1.0
    if 'rice' in payload.crop.lower(): crop_factor = 1.2
    elif 'wheat' in payload.crop.lower(): crop_factor = 1.1
    elif 'cotton' in payload.crop.lower(): crop_factor = 0.8
    irrigation_boost = 0.15 if payload.irrigation else 0.0
    median = base * crop_factor * (1 + irrigation_boost)
    p10 = median * 0.7
    p90 = median * 1.3
    return YieldPredOut(p10=round(p10,2), p50=round(median,2), p90=round(p90,2))

@app.post("/disease-detection")
async def disease_detection(image: UploadFile = File(...)):
    if image.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    # Placeholder: real model inference if available
    leaf_model_path = MODELS_DIR / 'leaf_cnn.keras'
    if leaf_model_path.exists():
        # Could integrate TensorFlow inference; omitted for simplicity
        predicted = 'LeafDiseaseX'
        confidence = 0.85
        advice = 'Apply recommended fungicide and monitor moisture.'
    else:
        predicted = 'Unknown'
        confidence = 0.0
        advice = 'Model not available. Provide a clear leaf image or train CNN.'
    return {"disease": predicted, "confidence": confidence, "advice": advice}

@app.get("/location-analysis", response_model=LocationOut)
def location_analysis(lat: float, lon: float):
    # Find nearest state by Euclidean distance
    best_state = None
    best_dist = 1e9
    for state, coord in location_analyzer.indian_states.items():
        d = (coord['lat'] - lat)**2 + (coord['lon'] - lon)**2
        if d < best_dist:
            best_dist = d
            best_state = state
    if not best_state:
        raise HTTPException(status_code=404, detail="No matching state")
    info = location_analyzer.soil_characteristics.get(best_state)
    rainfall = float(np.random.normal(800, 120))
    soil = info.get('dominant_soil', 'Unknown')
    advisories = info.get('recommendations', [])
    return LocationOut(state=best_state, rainfall=round(rainfall,2), soil=soil, advisories=advisories)

# Admin placeholder
@app.post("/admin/retrain")
def admin_retrain():
    if model_load_error:
        raise HTTPException(status_code=400, detail=f"Cannot retrain: {model_load_error}")
    # In production, trigger background job or call training script.
    return {"status": "queued"}

@app.get("/model/feature-importance", response_model=FeatureImportanceOut)
def model_feature_importance(limit: int = 15):
    if not crop_model or not hasattr(crop_model, 'feature_importances_') or not feature_names:
        raise HTTPException(status_code=404, detail="Feature importance unavailable")
    importances = crop_model.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    items = [FeatureImportanceItem(feature=f, importance=float(round(i,5))) for f,i in pairs[:limit]]
    return FeatureImportanceOut(items=items)

@app.post("/crop-recommendation/detailed", response_model=DetailedCropRecOut)
def crop_recommendation_detailed(payload: CropRecIn, top_k: int = 8):
    season = payload.season.lower()
    explanation = "Enhanced detailed crop variety recommendations"
    if not crop_model or not scaler or not feature_names or not classes:
        raise HTTPException(status_code=503, detail="Model unavailable for detailed recommendations")
    season_kharif = 1 if season.startswith("kharif") else 0
    season_rabi = 1 if season.startswith("rabi") else 0
    season_zaid = 1 if season.startswith("zaid") or season.startswith("spring") else 0
    feat_map = {
        'Temperature': 26,
        'Humidity': 65,
        'pH': 6.8,
        'Nitrogen': 120,
        'Phosphorus': 60,
        'Potassium': 100,
        'Rainfall': 800,
        'Moisture': 25,
        'EC': 1.2,
        'Latitude': location_analyzer.indian_states.get(payload.state, {'lat':25})['lat'],
        'Longitude': location_analyzer.indian_states.get(payload.state, {'lon':80})['lon'],
        'Wind_Speed': 10,
        'Solar_Radiation': 20,
        'Day_Length': 12,
        'Season_Kharif': season_kharif,
        'Season_Rabi': season_rabi,
        'Season_Zaid': season_zaid
    }
    vector = [feat_map.get(fn, 0) for fn in feature_names]
    scaled = scaler.transform([vector])
    probs = crop_model.predict_proba(scaled)[0]
    idxs = np.argsort(probs)[::-1][:top_k]
    items: List[DetailedCropItem] = []
    for i in idxs:
        name = classes[i]
        if '_' in name:
            crop, variety = name.split('_',1)
        else:
            crop, variety = name, 'Standard'
        confidence = float(probs[i])
        suitability = min(1.0, confidence * 1.25)
        items.append(DetailedCropItem(crop=crop, variety=variety, full_name=name, confidence=confidence, suitability_score=suitability))
    return DetailedCropRecOut(items=items, explanation=explanation)

@app.get("/crop-seasonal")
def crop_seasonal(season: str):
    season = season.lower()
    if not classes:
        return {"items": []}
    filtered = [c for c in classes if (season.startswith('kharif') and ('rice' in c.lower() or 'cotton' in c.lower())) or (season.startswith('rabi') and ('wheat' in c.lower() or 'rabi' in c.lower())) or (season.startswith('zaid') and ('spring' in c.lower()))]
    return {"items": filtered}

@app.get("/states", response_model=StatesOut)
def list_states():
    return StatesOut(states=list(location_analyzer.indian_states.keys()))

@app.get("/states/{state}", response_model=StateAnalysisOut)
def state_analysis(state: str):
    if state not in location_analyzer.indian_states:
        raise HTTPException(status_code=404, detail="Unknown state")
    analysis = location_analyzer.analyze_soil_suitability(state)
    return StateAnalysisOut(
        location=state,
        overall_score=analysis['suitability_analysis']['overall_score'],
        fertility=analysis['soil_characteristics']['fertility'],
        dominant_soil=analysis['soil_characteristics']['dominant_soil'],
        major_crops=analysis['soil_characteristics']['major_crops'],
        recommendations=analysis['recommendations']
    )

@app.get("/sensor/simulate", response_model=SensorReadingOut)
def sensor_simulate():
    # Provide synthetic sensor reading for UI streaming
    if latest_sensor_reading.get("source") == "mqtt":
        return SensorReadingOut(
            moisture=latest_sensor_reading["moisture"],
            temperature=latest_sensor_reading["temperature"],
            humidity=latest_sensor_reading["humidity"],
            nitrogen=latest_sensor_reading["nitrogen"],
            timestamp=latest_sensor_reading["timestamp"],
        )
    reading = {
        "moisture": float(round(np.random.normal(25,5),2)),
        "temperature": float(round(np.random.normal(26,2),2)),
        "humidity": float(round(np.random.normal(65,10),2)),
        "nitrogen": float(round(np.random.normal(120,25),2)),
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()+"Z"
    }
    latest_sensor_reading.update(reading)
    latest_sensor_reading["source"] = "synthetic"
    return SensorReadingOut(**reading)

@app.get("/dashboard/state/{state}", response_model=DashboardMetricsOut)
def dashboard_state(state: str):
    if state not in location_analyzer.indian_states:
        raise HTTPException(status_code=404, detail="Unknown state")
    analysis = location_analyzer.analyze_soil_suitability(state)
    metrics = {
        "overall_score": analysis['suitability_analysis']['overall_score'],
        "temperature": analysis['current_weather']['temperature'],
        "humidity": analysis['current_weather']['humidity'],
        "rainfall_24h": analysis['current_weather']['rainfall_24h'],
        "soil_dominant": analysis['soil_characteristics']['dominant_soil']
    }
    return DashboardMetricsOut(metrics=metrics, recommendations=analysis['recommendations'], state=state)

@app.get("/model/sample-prediction", response_model=SamplePredictionOut)
def sample_prediction(top_k: int = 5):
    if not crop_model or not scaler or not feature_names or not classes:
        return SamplePredictionOut(items=[], model_loaded=False, error=model_load_error or "Model not available")
    # Use median synthetic feature vector similar to crop recommendation
    feat_map = {
        'Temperature': 26,
        'Humidity': 65,
        'pH': 6.8,
        'Nitrogen': 120,
        'Phosphorus': 60,
        'Potassium': 100,
        'Rainfall': 800,
        'Moisture': 25,
        'EC': 1.2,
        'Latitude': 25,
        'Longitude': 80,
        'Wind_Speed': 10,
        'Solar_Radiation': 20,
        'Day_Length': 12,
        'Season_Kharif': 1,
        'Season_Rabi': 0,
        'Season_Zaid': 0
    }
    vector = [feat_map.get(fn, 0) for fn in feature_names]
    scaled = scaler.transform([vector])
    probs = crop_model.predict_proba(scaled)[0]
    idxs = np.argsort(probs)[::-1][:top_k]
    items: List[SamplePredictionItem] = []
    for i in idxs:
        name = classes[i]
        confidence = float(probs[i])
        suitability = min(1.0, confidence * 1.25)
        items.append(SamplePredictionItem(full_name=name, confidence=confidence, suitability_score=suitability))
    return SamplePredictionOut(items=items, model_loaded=True)

@app.websocket("/ws/sensors")
async def websocket_sensors(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # If MQTT updated reading, use that; else generate synthetic drift
            if latest_sensor_reading.get("source") != "mqtt":
                # Light random drift to simulate changes
                import random
                latest_sensor_reading["moisture"] = max(0, latest_sensor_reading["moisture"] + random.uniform(-0.5,0.5))
                latest_sensor_reading["temperature"] = latest_sensor_reading["temperature"] + random.uniform(-0.2,0.2)
                latest_sensor_reading["humidity"] = max(0, latest_sensor_reading["humidity"] + random.uniform(-1.0,1.0))
                latest_sensor_reading["nitrogen"] = max(0, latest_sensor_reading["nitrogen"] + random.uniform(-2.0,2.0))
                latest_sensor_reading["timestamp"] = __import__('datetime').datetime.utcnow().isoformat()+"Z"
            payload = {
                "moisture": round(latest_sensor_reading["moisture"],2),
                "temperature": round(latest_sensor_reading["temperature"],2),
                "humidity": round(latest_sensor_reading["humidity"],2),
                "nitrogen": round(latest_sensor_reading["nitrogen"],2),
                "timestamp": latest_sensor_reading["timestamp"],
                "source": latest_sensor_reading.get("source","synthetic")
            }
            await ws.send_json(payload)
            await __import__('asyncio').sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
