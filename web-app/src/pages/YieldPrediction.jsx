import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function YieldPrediction() {
  const [formData, setFormData] = useState({
    crop: 'Rice',
    area: 10,
    nitrogen: 90,
    phosphorus: 42,
    potassium: 43,
    ph: 6.5,
    rainfall: 200,
    temperature: 25,
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Soybean', 'Sugarcane', 'Potato', 'Tomato'];

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/yield-prediction`, formData);
      setResults(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error predicting yield');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">📈 Yield Prediction</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 Predict expected crop yields based on environmental conditions and cultivation area</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📊 Input Data</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Crop Type</label>
              <select
                name="crop"
                value={formData.crop}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
              >
                {crops.map((crop) => (
                  <option key={crop} value={crop}>{crop}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cultivation Area (hectares)</label>
              <input
                type="number"
                name="area"
                value={formData.area}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Nitrogen (N) - ppm</label>
              <input
                type="number"
                name="nitrogen"
                value={formData.nitrogen}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phosphorus (P) - ppm</label>
              <input
                type="number"
                name="phosphorus"
                value={formData.phosphorus}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Potassium (K) - ppm</label>
              <input
                type="number"
                name="potassium"
                value={formData.potassium}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Soil pH</label>
              <input
                type="number"
                name="ph"
                value={formData.ph}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                min="0"
                max="14"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Rainfall (mm)</label>
              <input
                type="number"
                name="rainfall"
                value={formData.rainfall}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Temperature (°C)</label>
              <input
                type="number"
                name="temperature"
                value={formData.temperature}
                onChange={handleInputChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                step="0.1"
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary text-white px-6 py-3 rounded-md hover:bg-secondary transition-colors font-semibold disabled:opacity-50"
            >
              {loading ? '🔄 Calculating...' : '📈 Predict Yield'}
            </button>
          </form>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📋 Prediction Results</h2>
          {!results && (
            <div className="text-center text-gray-500 py-12">
              <div className="text-6xl mb-4">📈</div>
              <p>Enter your data to get AI-powered yield predictions</p>
            </div>
          )}
          {results && (
            <div className="space-y-4">
              <div className="bg-green-50 border-l-4 border-green-500 p-4">
                <h3 className="font-bold text-green-900 mb-2">Expected Yield</h3>
                <div className="text-3xl font-bold text-primary">{results.predicted_yield} tons</div>
                <div className="text-sm text-gray-600">For {formData.area} hectares of {formData.crop}</div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded">
                  <h3 className="font-bold text-blue-900 mb-2">Per Hectare</h3>
                  <div className="text-xl font-bold">{(results.predicted_yield / formData.area).toFixed(2)} tons/ha</div>
                </div>
                <div className="bg-yellow-50 p-4 rounded">
                  <h3 className="font-bold text-yellow-900 mb-2">Confidence</h3>
                  <div className="text-xl font-bold">{(results.confidence * 100).toFixed(1)}%</div>
                </div>
              </div>

              {results.factors && (
                <div className="bg-gray-50 p-4 rounded">
                  <h3 className="font-bold text-gray-700 mb-2">Key Factors</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                    {results.factors.map((factor, idx) => (
                      <li key={idx}>{factor}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
