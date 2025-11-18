import { useState, useEffect } from 'react';
import axios from 'axios';
import SensorData from '../components/SensorData';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [selectedState, setSelectedState] = useState('Maharashtra');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDashboard();
  }, [selectedState]);

  const loadDashboard = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/dashboard/state/${selectedState}`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const states = ['Maharashtra', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu'];

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">📊 Agricultural Dashboard</h1>
      
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <label className="block text-lg font-medium text-gray-700 mb-2">Select State</label>
        <select
          value={selectedState}
          onChange={(e) => setSelectedState(e.target.value)}
          className="w-full md:w-1/3 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {states.map((state) => (
            <option key={state} value={state}>{state}</option>
          ))}
        </select>
      </div>

      <div className="mb-8">
        <SensorData />
      </div>

      {loading && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔄</div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      )}

      {!loading && metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gradient-to-br from-green-400 to-green-600 text-white rounded-lg shadow-lg p-6">
            <div className="text-4xl mb-2">🌾</div>
            <div className="text-sm opacity-90">Total Crops</div>
            <div className="text-3xl font-bold">{metrics.total_crops || 0}</div>
          </div>

          <div className="bg-gradient-to-br from-blue-400 to-blue-600 text-white rounded-lg shadow-lg p-6">
            <div className="text-4xl mb-2">📈</div>
            <div className="text-sm opacity-90">Avg Yield</div>
            <div className="text-3xl font-bold">{metrics.avg_yield || 0} t/ha</div>
          </div>

          <div className="bg-gradient-to-br from-yellow-400 to-yellow-600 text-white rounded-lg shadow-lg p-6">
            <div className="text-4xl mb-2">🌡️</div>
            <div className="text-sm opacity-90">Avg Temp</div>
            <div className="text-3xl font-bold">{metrics.avg_temperature || 0}°C</div>
          </div>

          <div className="bg-gradient-to-br from-purple-400 to-purple-600 text-white rounded-lg shadow-lg p-6">
            <div className="text-4xl mb-2">💧</div>
            <div className="text-sm opacity-90">Rainfall</div>
            <div className="text-3xl font-bold">{metrics.avg_rainfall || 0} mm</div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6 md:col-span-2">
            <h3 className="text-xl font-bold text-secondary mb-4">🌾 Top Crops by Area</h3>
            <div className="space-y-3">
              {metrics.top_crops_by_area ? metrics.top_crops_by_area.slice(0, 5).map((crop, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl font-bold text-gray-300">{idx + 1}</span>
                    <span className="font-medium">{crop.name}</span>
                  </div>
                  <span className="text-sm text-gray-600">{crop.area} ha</span>
                </div>
              )) : <p className="text-gray-500">No data available</p>}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6 md:col-span-2">
            <h3 className="text-xl font-bold text-secondary mb-4">📊 Soil Health Indicators</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-green-50 p-4 rounded">
                <div className="text-sm text-gray-600">Nitrogen (N)</div>
                <div className="text-2xl font-bold text-primary">{metrics.avg_N || 0} ppm</div>
              </div>
              <div className="bg-blue-50 p-4 rounded">
                <div className="text-sm text-gray-600">Phosphorus (P)</div>
                <div className="text-2xl font-bold text-primary">{metrics.avg_P || 0} ppm</div>
              </div>
              <div className="bg-yellow-50 p-4 rounded">
                <div className="text-sm text-gray-600">Potassium (K)</div>
                <div className="text-2xl font-bold text-primary">{metrics.avg_K || 0} ppm</div>
              </div>
              <div className="bg-purple-50 p-4 rounded">
                <div className="text-sm text-gray-600">pH Level</div>
                <div className="text-2xl font-bold text-primary">{metrics.avg_pH || 0}</div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6 md:col-span-4">
            <h3 className="text-xl font-bold text-secondary mb-4">💡 Recommendations for {selectedState}</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 border-l-4 border-green-500 p-4">
                <h4 className="font-bold text-green-900 mb-2">Best Crops</h4>
                <p className="text-sm text-green-800">{metrics.recommended_crops?.join(', ') || 'Rice, Wheat, Cotton'}</p>
              </div>
              <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
                <h4 className="font-bold text-blue-900 mb-2">Optimal Season</h4>
                <p className="text-sm text-blue-800">{metrics.optimal_season || 'Kharif (Monsoon)'}</p>
              </div>
              <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4">
                <h4 className="font-bold text-yellow-900 mb-2">Key Focus</h4>
                <p className="text-sm text-yellow-800">Monitor soil moisture and nutrient levels regularly</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
