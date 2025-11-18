import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function LocationAnalysis() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('Maharashtra');
  const [stateData, setStateData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadStates();
  }, []);

  useEffect(() => {
    if (selectedState) {
      loadStateData();
    }
  }, [selectedState]);

  const loadStates = async () => {
    try {
      const response = await axios.get(`${API_URL}/states`);
      setStates(response.data.states || []);
    } catch (error) {
      console.error('Error loading states:', error);
    }
  };

  const loadStateData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/states/${selectedState}`);
      setStateData(response.data);
    } catch (error) {
      console.error('Error loading state data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">📍 Location-Based Analysis</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 Get location-specific agricultural insights and recommendations for Indian states</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <label className="block text-lg font-medium text-gray-700 mb-2">Select State</label>
        <select
          value={selectedState}
          onChange={(e) => setSelectedState(e.target.value)}
          className="w-full md:w-1/2 px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary text-lg"
        >
          {states.map((state) => (
            <option key={state} value={state}>{state}</option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔄</div>
          <p className="text-gray-600">Loading data for {selectedState}...</p>
        </div>
      )}

      {!loading && stateData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-secondary mb-4">🌾 Recommended Crops</h3>
            <div className="space-y-2">
              {stateData.major_crops && stateData.major_crops.length > 0 ? (
                stateData.major_crops.map((crop, idx) => (
                  <div key={idx} className="bg-green-50 p-3 rounded-lg border border-green-200">
                    <span className="font-medium text-green-800">{crop}</span>
                  </div>
                ))
              ) : (
                <p className="text-gray-500">No specific recommendations available</p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-secondary mb-4">🏆 Soil Fertility</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Overall Score:</span>
                <span className="text-2xl font-bold text-primary">{stateData.overall_score?.toFixed(1) || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Fertility Level:</span>
                <span className="font-bold text-green-600">{stateData.fertility || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Dominant Soil:</span>
                <span className="font-bold">{stateData.dominant_soil || 'N/A'}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-secondary mb-4">💡 Best Practices</h3>
            <div className="space-y-2">
              {stateData.recommendations && stateData.recommendations.length > 0 ? (
                stateData.recommendations.map((rec, idx) => (
                  <div key={idx} className="flex items-start space-x-2">
                    <span className="text-green-600 mt-1">✓</span>
                    <span className="text-gray-700">{rec}</span>
                  </div>
                ))
              ) : (
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
                  <li>Follow seasonal crop rotation</li>
                  <li>Monitor soil health regularly</li>
                  <li>Use drip irrigation for water conservation</li>
                  <li>Apply organic fertilizers when possible</li>
                </ul>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
