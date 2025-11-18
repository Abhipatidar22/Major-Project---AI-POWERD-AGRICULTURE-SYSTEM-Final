import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function CropRecommendation() {
  const [inputMethod, setInputMethod] = useState('manual');
  const [formData, setFormData] = useState({
    state: 'Maharashtra',
    season: 'Kharif',
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const states = ['Maharashtra', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'West Bengal', 'Bihar', 'Odisha', 'Kerala'];
  const seasons = ['Kharif', 'Rabi', 'Zaid'];

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSimulate = async () => {
    // Simulation not needed - API uses state and season
    alert(`Using ${formData.state} and ${formData.season} season for recommendations`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/crop-recommendation`, formData);
      setResults(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error getting recommendations. Please check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">🌾 Smart Crop Recommendation System</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 <strong>Pro Tip:</strong> This system recommends specific crop varieties (not just crop types) based on your environmental conditions. Get personalized suggestions with confidence scores!</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-primary">14+</div>
          <div className="text-sm text-gray-600">Crop Varieties</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-primary">60.4%</div>
          <div className="text-sm text-gray-600">Accuracy</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 text-center">
          <div className="text-2xl font-bold text-primary">15</div>
          <div className="text-sm text-gray-600">States Coverage</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📊 Input Your Data</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">Data Input Method</label>
            <select
              value={inputMethod}
              onChange={(e) => setInputMethod(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
            >
              <option value="manual">📝 Manual Entry</option>
              <option value="simulation">🎲 Smart Simulation</option>
            </select>
          </div>

          {inputMethod === 'simulation' && (
            <div className="mb-4">
              <p className="text-sm text-gray-600 mb-2">Select your location to get optimal recommendations</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">State</label>
              <select
                name="state"
                value={formData.state}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                required
              >
                {states.map((state) => (
                  <option key={state} value={state}>{state}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Season</label>
              <select
                name="season"
                value={formData.season}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                required
              >
                {seasons.map((season) => (
                  <option key={season} value={season}>{season}</option>
                ))}
              </select>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary text-white px-6 py-3 rounded-md hover:bg-secondary transition-colors font-semibold disabled:opacity-50"
            >
              {loading ? '🔄 Analyzing...' : '🌾 Get Recommendations'}
            </button>
          </form>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📋 Recommendations</h2>
          {!results && (
            <div className="text-center text-gray-500 py-12">
              <div className="text-6xl mb-4">🌾</div>
              <p>Enter your data and click "Get Recommendations" to see AI-powered crop suggestions</p>
            </div>
          )}
          {results && (
            <div className="space-y-4">
              <div className="bg-green-50 border-l-4 border-green-500 p-4">
                <h3 className="font-bold text-green-900 mb-2">🎯 Recommended Crops for {formData.state} - {formData.season}</h3>
                {results.crops && results.crops.length > 0 && (
                  <div className="space-y-2">
                    {results.crops.map((crop, idx) => (
                      <div key={idx} className={`p-3 rounded ${idx === 0 ? 'bg-white border-2 border-green-500' : 'bg-gray-50'}`}>
                        <div className="flex items-center justify-between">
                          <span className="font-bold text-lg">{idx + 1}. {crop.name}</span>
                          <span className="text-sm text-gray-600">{(crop.score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {results.explanation && (
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
                  <h3 className="font-bold text-blue-900 mb-2">💡 Explanation</h3>
                  <p className="text-sm text-gray-700">{results.explanation}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
