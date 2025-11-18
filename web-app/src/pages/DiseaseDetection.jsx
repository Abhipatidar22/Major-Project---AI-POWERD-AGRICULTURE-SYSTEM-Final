import { useState } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function DiseaseDetection() {
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
      const response = await axios.post(`${API_URL}/disease-detection`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (error) {
      console.error('Detection error:', error);
      alert('Error detecting disease');
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (disease) => {
    if (disease === 'healthy') return 'bg-green-50 border-green-500 text-green-900';
    if (['blight', 'rust'].includes(disease)) return 'bg-red-50 border-red-500 text-red-900';
    return 'bg-yellow-50 border-yellow-500 text-yellow-900';
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">🍃 Plant Disease Detection</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 Upload a clear image of a plant leaf to detect diseases using AI-powered image recognition</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📤 Upload Leaf Image</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              {!preview && (
                <div>
                  <div className="text-6xl mb-4">🍃</div>
                  <p className="text-gray-600 mb-4">Click to upload leaf image</p>
                  <p className="text-sm text-gray-500">For best results, ensure the leaf is clearly visible and well-lit</p>
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
              {loading ? '🔄 Analyzing...' : '🍃 Detect Disease'}
            </button>
          </form>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-secondary mb-4">📋 Detection Results</h2>
          {!results && (
            <div className="text-center text-gray-500 py-12">
              <div className="text-6xl mb-4">🍃</div>
              <p>Upload a leaf image to detect plant diseases</p>
            </div>
          )}
          {results && (
            <div className="space-y-4">
              <div className={`border-l-4 p-4 ${getSeverityColor(results.disease)}`}>
                <h3 className="font-bold mb-2">Diagnosis</h3>
                <div className="text-2xl font-bold capitalize">{results.disease.replace(/_/g, ' ')}</div>
                <div className="text-sm">Confidence: {(results.confidence * 100).toFixed(1)}%</div>
              </div>

              {results.disease !== 'healthy' && (
                <div className="bg-red-50 border-l-4 border-red-500 p-4">
                  <h3 className="font-bold text-red-900 mb-2">⚠️ Action Required</h3>
                  <p className="text-red-800">{results.advice || 'Monitor leaf health and apply targeted treatment'}</p>
                </div>
              )}

              {results.disease === 'healthy' && (
                <div className="bg-green-50 border-l-4 border-green-500 p-4">
                  <h3 className="font-bold text-green-900 mb-2">✅ Plant Status</h3>
                  <p className="text-green-800">{results.advice || 'No action needed. Plant is healthy!'}</p>
                </div>
              )}

              <div className="bg-gray-50 p-4 rounded">
                <h3 className="font-bold text-gray-700 mb-2">💡 Recommendations</h3>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                  <li>Monitor plant regularly for changes</li>
                  <li>Maintain proper irrigation and drainage</li>
                  <li>Apply appropriate fertilizers based on soil analysis</li>
                  <li>Consult agricultural extension officer if needed</li>
                </ul>
              </div>

              {results.top_diseases && results.top_diseases.length > 1 && (
                <div className="bg-blue-50 p-4 rounded">
                  <h3 className="font-bold text-blue-900 mb-2">Other Possibilities</h3>
                  <div className="space-y-2">
                    {results.top_diseases.slice(1, 4).map((disease, idx) => (
                      <div key={idx} className="flex items-center justify-between">
                        <span className="capitalize">{disease.name.replace(/_/g, ' ')}</span>
                        <span className="text-sm text-gray-600">{(disease.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
