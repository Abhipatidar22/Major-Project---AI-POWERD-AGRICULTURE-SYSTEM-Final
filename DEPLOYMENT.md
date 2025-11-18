# Vercel Deployment Guide

## Your Repository
✅ **GitHub**: https://github.com/Abhipatidar22/Major-Project---AI-POWERD-AGRICULTURE-SYSTEM-Final

## Deploy Frontend to Vercel

### Step 1: Connect Repository
1. Go to https://vercel.com
2. Click **"Add New"** → **"Project"**
3. Import: `Abhipatidar22/Major-Project---AI-POWERD-AGRICULTURE-SYSTEM-Final`

### Step 2: Configure Build Settings
```
Framework Preset: Vite
Root Directory: web-app
Build Command: npm run build
Output Directory: dist
Install Command: npm install
```

### Step 3: Environment Variables
Add this environment variable:
```
VITE_API_URL = https://your-backend-url.com
```
(You'll update this after deploying the backend)

### Step 4: Deploy
Click **"Deploy"** - Frontend will be live in ~2 minutes!

---

## Deploy Backend (Choose One)

### Option A: Render.com (Recommended - Free)
1. Go to https://render.com
2. Click **"New"** → **"Web Service"**
3. Connect GitHub repo
4. Configure:
   - **Name**: `agri-api`
   - **Root Directory**: Leave empty
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

5. Add Environment Variables:
   ```
   PYTHON_VERSION = 3.13
   ```

6. Deploy!

### Option B: Railway.app
1. Go to https://railway.app
2. **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Configure:
   - **Root Directory**: Leave empty
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

### Option C: Heroku
```bash
# Install Heroku CLI, then:
heroku login
heroku create agri-backend
git push heroku main
```

---

## After Backend Deployment

1. Copy your backend URL (e.g., `https://agri-api.onrender.com`)
2. Go to Vercel → Your Project → **Settings** → **Environment Variables**
3. Update `VITE_API_URL` to your backend URL
4. **Redeploy** the frontend

---

## Final URLs
- **Frontend**: https://your-project.vercel.app
- **Backend**: https://your-backend-url.com
- **API Docs**: https://your-backend-url.com/docs

## Testing
- Visit frontend URL
- Test crop recommendation, sensor data, etc.
- All features should work!

---

## Need Help?
- Vercel Docs: https://vercel.com/docs
- Render Docs: https://render.com/docs
