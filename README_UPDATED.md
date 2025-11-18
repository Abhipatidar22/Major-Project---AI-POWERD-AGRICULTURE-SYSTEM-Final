# 🌾 Smart AI-Powered Agriculture System — (Enhanced)

This project is an end-to-end demo that showcases an AI pipeline for small-scale agriculture tasks. It includes:

- Variety-level crop recommendations (e.g. Basmati Rice, Durum Wheat, Bt Cotton)
- Location-aware soil analysis for India (state-level defaults and weather hooks)
- Yield prediction with quantile models (P10 / P50 / P90)
- Leaf disease detection (optional CNN)
- Soil texture classification (optional CNN)
- Streamlit-based interactive UI with guided inputs, tooltips and visualizations

Most models are trained on generated synthetic/demo data and saved under `models/`. The app is usable without TensorFlow (CNN parts are optional).

---

## Quickstart (Windows PowerShell)

1) (recommended) Create and activate a virtual environment

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1   # PowerShell
```

2) Install base dependencies (no TensorFlow required)

```powershell
python -m pip install -r requirements.txt
```

3) Train the lightweight demo crop model (fast)

```powershell
# trains the enhanced crop model saved to models/enhanced_crop_model.pkl
python direct_crop_trainer.py
```

4) (optional) Train all enhanced models (this will attempt to train CNNs if TF is installed)

```powershell
python train_enhanced_models_simple.py
```

5) Run the Streamlit app (uses the enhanced UI)

```powershell
python -m streamlit run "enhanced_app.py" --server.headless true --server.port 8504
# open http://localhost:8504 in your browser
```

### Optional: enable image CNNs
If you want the soil/leaf image models too, install TensorFlow (CPU) and prepare demo data:

```powershell
python -m pip install -r requirements_image.txt
python src/prepare_demo.py
```

If TensorFlow is not available, the CNN training/usage will be skipped and the app will still function for tabular/variety recommendations.

---

## What changed (latest features)

- Variety-based crop recommendations (14+ varieties) with suitability scoring and seasonal suggestions
- Location-based soil defaults for India and hooks for real-time weather lookup
- Improved, user-friendly Streamlit UI with quick-start guide, tooltips, CSV upload guidance and simulation mode
- Robust model-loading logic with a `LoadedEnhancedCropModel` wrapper and helpful debug output when models are missing
- Small utility scripts to train the crop model quickly (`direct_crop_trainer.py`) and a lightweight all-in-one trainer (`train_enhanced_models_simple.py`)
- Quantile yield models for uncertainty-aware predictions (q10, q50, q90)

---

## Files of interest

```
ai_agri_full_fixed/
├─ enhanced_app.py                    # main Streamlit application (enhanced UI)
├─ direct_crop_trainer.py             # quick trainer to create enhanced_crop_model.pkl
├─ train_enhanced_models_simple.py    # simplified master training script for all enhanced models
├─ requirements.txt
├─ requirements_image.txt
├─ models/
│  ├─ enhanced_crop_model.pkl
│  ├─ enhanced_crop_labels.json
│  ├─ leaf_cnn.keras (optional)
│  └─ soil_cnn.keras (optional)
└─ src/
   ├─ enhanced_crop_trainer.py
   ├─ enhanced_soil_trainer.py
   ├─ enhanced_yield_trainer.py
   ├─ enhanced_disease_trainer.py
   ├─ location_analysis.py
   └─ utils.py
```

---

## Troubleshooting — "No models available" in the UI

If the sidebar says **No Models Available**:

1. Confirm the model file exists:

```powershell
> Get-ChildItem models\enhanced_crop_model.pkl
```

2. Quick fix: re-create the crop model (fast):

```powershell
python direct_crop_trainer.py
```

3. If the file exists but the app still reports missing models, restart Streamlit and watch logs (the app emits helpful debug lines during model load):

```powershell
# stop any running python/streamlit processes then
python -m streamlit run "enhanced_app.py" --server.headless true --server.port 8504
```

4. If you want me to add extra logging or automatically create a placeholder model at startup, I can modify `enhanced_app.py` accordingly.

---

## Contributing / Next steps

- Add more crop varieties and regional parameters
- Integrate a real weather API and soil surveys for live recommendations
- Add export/print-friendly reports for farmers

Enjoy exploring the enhanced system — let me know if you want a guided demo flow or help deploying it to a server.

---

## Frontend (Next.js) Integration

The project now includes a modern Node.js + Next.js frontend (see `frontend/`) ready for Vercel deployment. It provides dedicated pages for:

- Soil Analysis (`/soil-analysis`)
- Crop Recommendation (`/crop-recommendation`)
- Yield Prediction (`/yield-prediction`)
- Disease Detection (image upload) (`/disease-detection`)
- Location Analysis (geolocation) (`/location-analysis`)
- Admin Dashboard (placeholder) (`/admin`)

### Frontend Directory Structure (simplified)

```
frontend/
   src/
      app/
         layout.tsx
         page.tsx                # Landing page (to be customized)
         soil-analysis/page.tsx
         crop-recommendation/page.tsx
         yield-prediction/page.tsx
         disease-detection/page.tsx
         location-analysis/page.tsx
         admin/page.tsx
      components/
         Nav.tsx
         ModuleCard.tsx
      lib/
         api.ts                  # Fetch helpers hitting Python backend
   .env.example
   package.json
   README.md
```

### Environment Variables

Copy `.env.example` to `.env.local` inside `frontend/` and set:

```
NEXT_PUBLIC_API_BASE_URL=https://your-python-backend.example.com
NEXT_PUBLIC_API_KEY=your-secure-key
```

### Running Frontend Locally

```powershell
cd frontend
npm install   # if dependencies not yet installed
npm run dev
# open http://localhost:3000
```

### Connecting to Python Backend

The TypeScript helper functions in `frontend/src/lib/api.ts` assume REST endpoints like:

- `POST /soil-analysis` → `{ status, recommendation }`
- `POST /crop-recommendation` → `{ crops: [{ name, score }], explanation }`
- `POST /yield-prediction` → `{ p10, p50, p90, units }`
- `POST /disease-detection` (multipart image upload) → `{ disease, confidence, advice }`
- `GET /location-analysis?lat=..&lon=..` → `{ state, rainfall, soil, advisories }`

Implement these with FastAPI (recommended) or another Python framework, enable CORS, and deploy separately. (I can scaffold FastAPI endpoints on request.)

### Deploying to Vercel

1. Push repository to GitHub (ensure `frontend/` is included).
2. In Vercel dashboard: Import project → set root directory to `frontend` → configure environment variables.
3. Trigger deploy; Vercel auto-builds Next.js.
4. Add your backend URL to `NEXT_PUBLIC_API_BASE_URL`.

### Next Steps

- Add authentication (NextAuth.js or Clerk) and role-based admin controls.
- Replace placeholder API calls with real endpoints.
- Add global state / caching (TanStack Query) around data-heavy modules.
- Add Insights dashboard (feature importance + state soil analysis).
- Add loading skeletons and error boundary components.
- Integrate map visualization for location analysis (Leaflet / Mapbox GL).

Let me know if you want auto-generated FastAPI boilerplate, auth setup, or deployment scripts next.

### Visual Design Enhancements (Latest)

- Global farming background applied site-wide with dark-mode aware overlay.
- Glass-style surfaces (`content-surface`) ensure readability over imagery.
- Sample model prediction surfaced in Admin to verify trained model.
- Root route redirects to crop recommendation for focused UX (no generic home page).

Customize background: edit `frontend/src/app/globals.css` `.site-bg` URL or store a local image at `frontend/public/background.jpg` and set:

```css
.site-bg { background: url("/background.jpg") center/cover no-repeat fixed; }
```

---
## Deployment (Vercel)

1. In Vercel dashboard choose "Import Project" and set the root directory to `frontend/`.
2. Ensure environment variables:
   - `NEXT_PUBLIC_API_BASE_URL` → your FastAPI backend URL.
   - `NEXT_PUBLIC_API_KEY` → optional API key if you add verification.
3. Vercel will detect Next.js automatically; `vercel.json` provides build hints.
4. Backend (FastAPI) deploy separately (Render, Railway, AWS, etc.).
5. Update `public/sitemap.xml` domain before production.
6. Add a custom domain and enable HTTPS (automatic).

### Performance Considerations
- Static prerendered pages minimize TTFB.
- React Query caching reduces repeated API calls and improves perceived speed.
- Use `NEXT_PUBLIC_API_BASE_URL` pointing to a geographically close region to Vercel edge.

### Optional Enhancements Before Final Submission
- Authentication & RBAC for `/admin`.
- WebSocket streaming for real sensor feed.
- Image optimization for disease detection previews.
- Add error boundaries and toast notifications for API errors.
\n+### Backend API (FastAPI)\n+\n+Added `api_server.py` exposing REST endpoints consumed by the Next.js frontend.\n+\n+Run locally (after installing new requirements):\n+\n+```powershell\n+python -m pip install -r requirements.txt\n+uvicorn api_server:app --reload --port 8000\n+```\n+\n+Endpoints (request → response shape):\n+\n+| Endpoint | Method | Purpose | Notes |\n+|----------|--------|---------|-------|\n+| /health | GET | Basic health & model load status | Returns model_loaded flag |\n+| /soil-analysis | POST | Soil heuristics | JSON: moisture, ph, nitrogen |\n+| /crop-recommendation | POST | Variety recommendations | Uses model if available, else heuristic |\n+| /yield-prediction | POST | Quantile-style yield prediction | Synthetic simple formula |\n+| /disease-detection | POST | Leaf disease detection | Image upload (multipart) placeholder |\n+| /location-analysis | GET | Reverse nearest-state soil info | lat, lon query params |\n+| /admin/retrain | POST | Placeholder retrain trigger | Stub |\n+\n+Environment integration: set `NEXT_PUBLIC_API_BASE_URL` in the frontend to the FastAPI host (e.g. http://localhost:8000).\n+\n+Security TODOs: add API key validation middleware, rate limiting, and authentication for admin endpoints.\n*** End Patch
