# 🚀 Quick Start Guide

## Running the Application

### ✅ Current Status
Both servers are running and working:
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 🖥️ Start Servers Manually

#### Backend (Python FastAPI)
```powershell
cd "c:\Users\karan\Downloads\Major Project - AI POWERED AGRICULTURE SYSTEM\ai_agri_full_fixed"
.\.venv\Scripts\activate
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend (React Vite)
```powershell
cd "c:\Users\karan\Downloads\Major Project - AI POWERED AGRICULTURE SYSTEM\ai_agri_full_fixed\web-app"
npm run dev
```

### 🔧 VS Code Tasks

Use the built-in VS Code tasks:
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select either:
   - `Start Backend Server`
   - `Start Frontend Dev Server`

### 📦 Production Build

#### Build Frontend
```powershell
cd web-app
npm run build
```
Output will be in `web-app/dist/`

#### Preview Production Build
```powershell
npm run preview
```

### 🌐 Deploy to Vercel

#### Frontend Deployment
```powershell
cd web-app

# Install Vercel CLI (if not installed)
npm i -g vercel

# Deploy
vercel --prod
```

#### Environment Variables
Create `.env` in `web-app/`:
```
VITE_API_URL=https://your-backend-url.com
```

For Vercel dashboard:
- `VITE_API_URL` = your deployed backend URL

### 🐍 Deploy Backend

Recommended platforms:
1. **Render** (Easiest for Python)
   - Connect GitHub repo
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

2. **Railway**
   - Similar to Render
   - Auto-detects Python apps

3. **Heroku**
   - Create `Procfile`: `web: uvicorn api_server:app --host 0.0.0.0 --port $PORT`

### 🧪 Test Endpoints

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# Get states
Invoke-RestMethod http://localhost:8000/states

# Crop recommendation
$data = @{ state = "Maharashtra"; season = "Kharif" } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/crop-recommendation -Method POST -Body $data -ContentType "application/json"
```

### 🛠️ Troubleshooting

#### Port Already in Use
```powershell
# Kill process on port 8000
$p = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($p) { Stop-Process -Id $p -Force }

# Kill process on port 5173
$p = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($p) { Stop-Process -Id $p -Force }
```

#### Frontend Blank Page
1. Check browser console for errors
2. Verify API URL in `.env`
3. Ensure backend is running

#### API Errors
1. Check backend logs
2. Verify models are loaded: http://localhost:8000/health
3. Check CORS settings in `api_server.py`

### 📝 Features

All pages are working:
- ✅ Home - Feature showcase
- ✅ Crop Recommendation - State & season based recommendations
- ✅ Soil Analysis - Image upload for analysis
- ✅ Yield Prediction - Crop yield calculator
- ✅ Disease Detection - Plant disease from leaf images
- ✅ Location Analysis - State-specific insights
- ✅ Dashboard - Agricultural metrics

### 🎨 Tech Stack

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- React Router
- Axios

**Backend:**
- FastAPI
- Python 3.13
- scikit-learn
- NumPy, Pandas
- Pillow

---

For detailed documentation, see `README.md`
