import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function SoilAnalysis() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select an image');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/soil-analysis`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Error analyzing soil image');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">🔬 Soil Analysis</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 Upload a clear image of soil to analyze soil type, pH levels, and nutrient content using AI</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📤 Upload Soil Image</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              {!preview && (
                <div>
                  <div className="text-6xl mb-4">📷</div>
                  <p className="text-gray-600 mb-4">Click to upload soil image</p>
                </div>
              )}
              {preview && (
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded" />
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="mt-4"
              />
            </div>

            <button
              type="submit"
              disabled={loading || !selectedFile}
              className="w-full bg-primary text-white px-6 py-3 rounded-md hover:bg-secondary transition-colors font-semibold disabled:opacity-50"
            >
              {loading ? '🔄 Analyzing...' : '🔬 Analyze Soil'}
            </button>
          </form>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📋 Analysis Results</h2>
          {!results && (
            <div className="text-center text-gray-500 py-12">
              <div className="text-6xl mb-4">🔬</div>
              <p>Upload a soil image to get AI-powered analysis</p>
            </div>
          )}
          {results && (
            <div className="space-y-4">
              <div className="bg-green-50 border-l-4 border-green-500 p-4">
                <h3 className="font-bold text-green-900 mb-2">Soil Type</h3>
                <div className="text-2xl font-bold text-primary capitalize">{results.soil_type}</div>
                <div className="text-sm text-gray-600">Confidence: {(results.confidence * 100).toFixed(1)}%</div>
              </div>

              <div className="bg-gray-50 p-4 rounded">
                <h3 className="font-bold text-gray-700 mb-2">💡 Recommendations</h3>
                <p className="text-gray-600">{results.explanation || 'Based on the soil analysis, appropriate crops will be recommended'}</p>
              </div>

              {results.ph_level && (
                <div className="bg-blue-50 p-4 rounded">
                  <h3 className="font-bold text-blue-900 mb-2">pH Level</h3>
                  <div className="text-xl font-bold">{results.ph_level}</div>
                </div>
              )}

              {results.moisture_level && (
                <div className="bg-yellow-50 p-4 rounded">
                  <h3 className="font-bold text-yellow-900 mb-2">Moisture Level</h3>
                  <div className="text-xl font-bold">{results.moisture_level}%</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
