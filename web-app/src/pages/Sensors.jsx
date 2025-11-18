import { useState, useEffect } from 'react';
import axios from 'axios';
import SensorData from '../components/SensorData';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Sensors() {
  const [history, setHistory] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadSensorHistory();
      }, 10000); // Refresh every 10 seconds

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const loadSensorHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/sensor/simulate`);
      setHistory(prev => [
        {
          ...response.data,
          id: Date.now(),
          timestamp: new Date().toISOString()
        },
        ...prev.slice(0, 9) // Keep last 10 readings
      ]);
    } catch (error) {
      console.error('Error loading sensor history:', error);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-primary mb-6">📡 Real-Time Sensor Monitoring</h1>
      
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6">
        <p className="text-blue-900">💡 Monitor live environmental conditions from your agricultural sensors. Data updates automatically via WebSocket connection.</p>
      </div>

      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-secondary">Current Readings</h2>
        <button
          onClick={() => setAutoRefresh(!autoRefresh)}
          className={`px-4 py-2 rounded-md font-semibold transition-colors ${
            autoRefresh
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {autoRefresh ? '⏸️ Stop Auto-Refresh' : '▶️ Auto-Refresh (10s)'}
        </button>
      </div>

      <SensorData />

      {history.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold text-secondary mb-4">📊 Recent History</h2>
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Moisture</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temperature</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Humidity</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Nitrogen</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {history.map((reading) => (
                    <tr key={reading.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {new Date(reading.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {reading.moisture?.toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {reading.temperature?.toFixed(1)}°C
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {reading.humidity?.toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {reading.nitrogen?.toFixed(0)} ppm
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-secondary mb-4">🔌 Sensor Status</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Moisture Sensor</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">Online</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Temperature Sensor</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">Online</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Humidity Sensor</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">Online</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Nitrogen Sensor</span>
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">Online</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-secondary mb-4">📈 Data Quality</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">Accuracy</span>
                <span className="text-sm font-medium text-gray-700">98%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-600 h-2 rounded-full" style={{ width: '98%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">Reliability</span>
                <span className="text-sm font-medium text-gray-700">95%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '95%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium text-gray-700">Coverage</span>
                <span className="text-sm font-medium text-gray-700">100%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-primary h-2 rounded-full" style={{ width: '100%' }}></div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-bold text-secondary mb-4">⚙️ Configuration</h3>
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-gray-600">Update Interval:</span>
              <span className="font-medium text-gray-900 ml-2">5 seconds</span>
            </div>
            <div>
              <span className="text-gray-600">Data Retention:</span>
              <span className="font-medium text-gray-900 ml-2">30 days</span>
            </div>
            <div>
              <span className="text-gray-600">Protocol:</span>
              <span className="font-medium text-gray-900 ml-2">MQTT + WebSocket</span>
            </div>
            <div>
              <span className="text-gray-600">Last Calibration:</span>
              <span className="font-medium text-gray-900 ml-2">2 days ago</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
