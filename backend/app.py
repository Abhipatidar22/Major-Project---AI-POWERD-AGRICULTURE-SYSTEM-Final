"""FastAPI backend for AI Powered Agriculture System.
Endpoints:
POST /crop/recommend
POST /yield/predict
POST /soil/analyze (multipart image)
POST /disease/detect (multipart image)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from backend.utils.image_utils import load_and_preprocess

app = FastAPI(title="AI Powered Agriculture Backend", description="Crop, Yield, Soil and Disease AI endpoints", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------- Crop Recommendation ---------
class CropRecommendIn(BaseModel):
    nitrogen: float = Field(..., ge=0, description="Soil Nitrogen (ppm)")
    phosphorus: float = Field(..., ge=0, description="Soil Phosphorus (ppm)")
    potassium: float = Field(..., ge=0, description="Soil Potassium (ppm)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall (mm)")
    temperature: float = Field(..., ge=-20, le=60, description="Temperature (°C)")

class CropRecommendOut(BaseModel):
    recommended_crop: str
    confidence: float
    reasoning: str

CROP_LABELS = ["rice", "wheat", "maize", "cotton", "soybean"]

@app.post("/crop/recommend", response_model=CropRecommendOut)
def crop_recommend(payload: CropRecommendIn):
    features = np.array([
        payload.nitrogen,
        payload.phosphorus,
        payload.potassium,
        payload.ph,
        payload.rainfall,
        payload.temperature
    ], dtype=float)
    # Simple heuristic placeholder
    scores = {
        "rice": (payload.rainfall > 150) * 0.6 + (6.0 <= payload.ph <= 7.5) * 0.3,
        "wheat": (payload.temperature < 30) * 0.5 + (payload.nitrogen > 50) * 0.3,
        "maize": (payload.temperature > 25) * 0.4 + (payload.phosphorus > 40) * 0.3,
        "cotton": (payload.temperature > 28) * 0.5 + (payload.rainfall > 120) * 0.2,
        "soybean": (payload.ph > 6 and payload.ph < 7.8) * 0.5 + (payload.potassium > 40) * 0.3,
    }
    best = max(scores.items(), key=lambda x: x[1])
    return CropRecommendOut(recommended_crop=best[0], confidence=round(best[1], 2), reasoning="Heuristic scoring placeholder; replace with trained model.")

# --------- Yield Prediction ---------
class YieldPredictIn(BaseModel):
    crop: str
    area: float = Field(..., gt=0, description="Cultivation area (hectares)")
    irrigation: bool = False

class YieldPredictOut(BaseModel):
    p10: float
    p50: float
    p90: float
    units: str = "tons"

@app.post("/yield/predict", response_model=YieldPredictOut)
def yield_predict(payload: YieldPredictIn):
    base_yield_factors = {"rice": 4.5, "wheat": 3.8, "maize": 5.0, "cotton": 1.5, "soybean": 2.8}
    factor = base_yield_factors.get(payload.crop.lower(), 3.0)
    irrigation_boost = 0.15 if payload.irrigation else 0.0
    median = payload.area * factor * (1 + irrigation_boost)
    return YieldPredictOut(p10=round(median * 0.7, 2), p50=round(median, 2), p90=round(median * 1.3, 2))

# --------- Soil Analysis (Image) ---------
class SoilAnalyzeOut(BaseModel):
    soil_type: str
    confidence: float
    explanation: str

SOIL_TYPES = ["loam", "sandy", "clay"]

@app.post("/soil/analyze", response_model=SoilAnalyzeOut)
async def soil_analyze(image: UploadFile = File(...)):
    if image.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await image.read()
    arr = load_and_preprocess(data)
    # Placeholder classification using average brightness
    brightness = float(arr.mean())
    # Simple mapping
    if brightness > 0.66:
        soil = "sandy"
    elif brightness > 0.33:
        soil = "loam"
    else:
        soil = "clay"
    confidence = round(abs(brightness - 0.5) + 0.5, 2)
    return SoilAnalyzeOut(soil_type=soil, confidence=confidence, explanation="Brightness heuristic placeholder; replace with CNN model.")

# --------- Disease Detection (Leaf Image) ---------
class DiseaseDetectOut(BaseModel):
    disease: str
    confidence: float
    advice: str

DISEASE_LABELS = ["healthy", "blight", "rust", "mildew"]

@app.post("/disease/detect", response_model=DiseaseDetectOut)
async def disease_detect(image: UploadFile = File(...)):
    if image.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await image.read()
    arr = load_and_preprocess(data)
    # Placeholder: use color channel means to decide
    channel_means = arr.mean(axis=(0,1))  # RGB means
    idx = int((channel_means[1] * 10) % len(DISEASE_LABELS))  # pseudo deterministic
    disease = DISEASE_LABELS[idx]
    confidence = round(0.6 + float(channel_means[0]) * 0.4, 2)
    advice = "Monitor leaf health and apply targeted treatment." if disease != "healthy" else "No action needed."
    return DiseaseDetectOut(disease=disease, confidence=confidence, advice=advice)

# --------- Health ---------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

# --------- Run hint ---------
# uvicorn app:app --reload --port 8000
