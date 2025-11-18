# 🌾 Smart AI-Powered Agriculture System

A modern, AI-powered agricultural recommendation system built with React + Tailwind CSS frontend and Python FastAPI backend, ready for deployment on Vercel.

## 🚀 Features

### Frontend (React + Tailwind CSS)
- **🌾 Crop Recommendation**: AI-powered variety-specific crop suggestions based on soil and environmental conditions
- **🔬 Soil Analysis**: Upload soil images for AI-based soil type and nutrient analysis
- **📈 Yield Prediction**: Predict crop yields based on cultivation area and environmental factors
- **🍃 Disease Detection**: Upload leaf images to detect plant diseases using AI
- **📍 Location Analysis**: Get location-specific recommendations for 15 Indian states
- **📊 Dashboard**: Comprehensive agricultural analytics and insights

### Backend (Python + FastAPI)
- RESTful API with automatic documentation (Swagger UI)
- ML models for crop recommendation, yield prediction, soil and disease detection
- Location-based data for Indian states
- Real-time sensor simulation
- CORS-enabled for cross-origin requests

## 📁 Project Structure

```
ai_agri_full_fixed/
├── web-app/                    # React frontend (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── components/        # React components (Navbar)
│   │   ├── pages/             # Page components (Home, CropRecommendation, etc.)
│   │   ├── App.jsx            # Main app with routing
│   │   └── index.css          # Tailwind CSS styles
│   ├── vercel.json            # Vercel deployment config
│   └── package.json
│
├── api_server.py              # FastAPI backend server
├── src/                       # Python modules (features, utils, visualizations)
├── models/                    # Trained ML models
├── data/                      # Dataset files
├── requirements.txt           # Python dependencies
└── .venv/                     # Python virtual environment
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.13+ (for backend)
- Node.js 18+ (for frontend)
- Git

### Backend Setup

1. **Create virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start backend server**:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Backend will run at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend**:
```bash
cd web-app
```

2. **Install dependencies**:
```bash
npm install
```

3. **Create environment file**:
```bash
cp .env.example .env
```

Edit `.env` and set your backend URL:
```
VITE_API_URL=http://localhost:8000
```

4. **Start development server**:
```bash
npm run dev
```

Frontend will run at: `http://localhost:5173`

## 🚀 Deployment

### Deploy Frontend to Vercel

1. **Build for production**:
```bash
cd web-app
npm run build
```

2. **Deploy to Vercel**:
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

Or connect your GitHub repository to Vercel for automatic deployments.

### Deploy Backend

Backend can be deployed to:
- **Render**: Python web service
- **Railway**: Python backend
- **Heroku**: Python dyno
- **AWS/GCP/Azure**: Container or serverless deployment

Update `VITE_API_URL` in your frontend `.env` to point to your deployed backend.

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /states` - List all Indian states
- `GET /states/{state}` - Get state-specific data
- `POST /crop-recommendation` - Basic crop recommendation
- `POST /crop-recommendation/detailed` - Detailed crop recommendations with varieties
- `POST /soil-analysis` - Analyze soil from image (multipart)
- `POST /yield-prediction` - Predict crop yield
- `POST /disease-detection` - Detect plant disease from image (multipart)
- `GET /location-analysis/{state}` - Location-based analysis
- `GET /dashboard/state/{state}` - Dashboard metrics for state

### Utility Endpoints
- `GET /sensor/simulate` - Simulate sensor reading
- `GET /model/sample-prediction` - Sample prediction
- `GET /model/feature-importance` - Model feature importance
- `WS /ws/sensors` - WebSocket for real-time sensor data

## 🎨 Technology Stack

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **scikit-learn** - Machine learning
- **NumPy & Pandas** - Data processing
- **Pillow** - Image processing
- **Plotly** - Data visualization

## 📝 Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000
```

### Backend (optional)
```
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=agriculture/sensors
```

## 🧪 Testing

### Test Backend
```bash
# Run backend tests
pytest tests/

# Manual API testing
curl http://localhost:8000/health
```

### Test Frontend
```bash
cd web-app
npm run build  # Test production build
npm run preview  # Preview production build
```

## 📚 Features in Detail

### 1. Crop Recommendation
- 14+ crop varieties (Basmati Rice, Durum Wheat, Bt Cotton, etc.)
- Confidence scores for each recommendation
- Input via manual entry or smart simulation
- Location-specific suggestions for 15 Indian states

### 2. Soil Analysis
- Upload soil images
- AI-powered soil type classification
- Nutrient level estimation
- pH level analysis

### 3. Yield Prediction
- Predict yields in tons/hectare
- Based on crop type, area, and environmental conditions
- Confidence intervals (P10, P50, P90)

### 4. Disease Detection
- Upload leaf images
- Detect diseases (blight, rust, mildew, healthy)
- Actionable treatment advice
- Alternative diagnosis suggestions

### 5. Location Analysis
- 15 major Indian agricultural states
- Climate data (temperature, rainfall, humidity)
- Soil nutrients (N, P, K, pH)
- Recommended crops by region

### 6. Dashboard
- State-wise agricultural statistics
- Top crops by cultivation area
- Soil health indicators
- Seasonal recommendations

## 🐛 Troubleshooting

### Frontend not connecting to backend
- Ensure backend is running on `http://localhost:8000`
- Check `VITE_API_URL` in `.env`
- Verify CORS is enabled in `api_server.py`

### Module not found errors (Python)
- Activate virtual environment: `.venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Tailwind styles not applying
- Rebuild: `npm run build`
- Check `tailwind.config.js` content paths

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Smart Agriculture Team**

## 🙏 Acknowledgments

- Indian agricultural data sources
- Open source ML libraries (scikit-learn, FastAPI, React)
- Tailwind CSS for beautiful styling
- Vite for lightning-fast development

---

Made with 💚 for Indian Farmers 🌾
