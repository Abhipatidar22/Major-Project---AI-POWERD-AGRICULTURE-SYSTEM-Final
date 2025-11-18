import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import CropRecommendation from './pages/CropRecommendation';
import SoilAnalysis from './pages/SoilAnalysis';
import YieldPrediction from './pages/YieldPrediction';
import DiseaseDetection from './pages/DiseaseDetection';
import LocationAnalysis from './pages/LocationAnalysis';
import Dashboard from './pages/Dashboard';
import Sensors from './pages/Sensors';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/crop-recommendation" element={<CropRecommendation />} />
          <Route path="/soil-analysis" element={<SoilAnalysis />} />
          <Route path="/yield-prediction" element={<YieldPrediction />} />
          <Route path="/disease-detection" element={<DiseaseDetection />} />
          <Route path="/location-analysis" element={<LocationAnalysis />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/sensors" element={<Sensors />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
