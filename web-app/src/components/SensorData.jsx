import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function SensorData() {
  const [sensorData, setSensorData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isLive, setIsLive] = useState(false);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    // Load initial sensor data
    loadSensorData();
    
    // Set up polling for updates every 5 seconds
    const interval = setInterval(() => {
      if (!isLive) {
        loadSensorData();
      }
    }, 5000);

    return () => {
      clearInterval(interval);
      if (ws) {
        ws.close();
      }
    };
  }, [isLive]);

  const loadSensorData = async () => {
    try {
      const response = await axios.get(`${API_URL}/sensor/simulate`);
      setSensorData(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading sensor data:', error);
      setLoading(false);
    }
  };

  const toggleLiveMode = () => {
    if (!isLive) {
      // Connect to WebSocket
      const wsUrl = API_URL.replace('http', 'ws');
      const socket = new WebSocket(`${wsUrl}/ws/sensors`);
      
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setSensorData(data);
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsLive(false);
      };

      socket.onclose = () => {
        setIsLive(false);
      };

      setWs(socket);
      setIsLive(true);
    } else {
      // Disconnect WebSocket
      if (ws) {
        ws.close();
      }
      setWs(null);
      setIsLive(false);
    }
  };

  const getSensorStatus = (value, min, max) => {
    if (value < min) return { status: 'Low', color: 'text-yellow-600 bg-yellow-50' };
    if (value > max) return { status: 'High', color: 'text-red-600 bg-red-50' };
    return { status: 'Optimal', color: 'text-green-600 bg-green-50' };
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🔄</div>
        <p className="text-gray-600">Loading sensor data...</p>
      </div>
    );
  }

  const moistureStatus = getSensorStatus(sensorData?.moisture || 0, 15, 35);
  const tempStatus = getSensorStatus(sensorData?.temperature || 0, 15, 35);
  const humidityStatus = getSensorStatus(sensorData?.humidity || 0, 40, 80);
  const nitrogenStatus = getSensorStatus(sensorData?.nitrogen || 0, 80, 200);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-secondary">📡 Real-Time Sensor Data</h2>
          <p className="text-sm text-gray-500">
            Last updated: {sensorData?.timestamp ? new Date(sensorData.timestamp).toLocaleTimeString() : 'N/A'}
          </p>
        </div>
        <button
          onClick={toggleLiveMode}
          className={`px-4 py-2 rounded-md font-semibold transition-colors ${
            isLive
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-green-500 hover:bg-green-600 text-white'
          }`}
        >
          {isLive ? '⏸️ Stop Live' : '▶️ Start Live'}
        </button>
      </div>

      {isLive && (
        <div className="mb-4 bg-green-50 border-l-4 border-green-500 p-3">
          <p className="text-green-800 font-medium">🟢 Live Mode Active - Real-time updates via WebSocket</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="border-2 border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">💧</span>
              <h3 className="text-lg font-bold text-gray-700">Soil Moisture</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${moistureStatus.color}`}>
              {moistureStatus.status}
            </span>
          </div>
          <div className="text-4xl font-bold text-primary mb-2">
            {sensorData?.moisture?.toFixed(1) || 'N/A'}%
          </div>
          <div className="text-sm text-gray-500">Optimal: 15-35%</div>
        </div>

        <div className="border-2 border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">🌡️</span>
              <h3 className="text-lg font-bold text-gray-700">Temperature</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${tempStatus.color}`}>
              {tempStatus.status}
            </span>
          </div>
          <div className="text-4xl font-bold text-primary mb-2">
            {sensorData?.temperature?.toFixed(1) || 'N/A'}°C
          </div>
          <div className="text-sm text-gray-500">Optimal: 15-35°C</div>
        </div>

        <div className="border-2 border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">💨</span>
              <h3 className="text-lg font-bold text-gray-700">Humidity</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${humidityStatus.color}`}>
              {humidityStatus.status}
            </span>
          </div>
          <div className="text-4xl font-bold text-primary mb-2">
            {sensorData?.humidity?.toFixed(1) || 'N/A'}%
          </div>
          <div className="text-sm text-gray-500">Optimal: 40-80%</div>
        </div>

        <div className="border-2 border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">🧪</span>
              <h3 className="text-lg font-bold text-gray-700">Nitrogen (N)</h3>
            </div>
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${nitrogenStatus.color}`}>
              {nitrogenStatus.status}
            </span>
          </div>
          <div className="text-4xl font-bold text-primary mb-2">
            {sensorData?.nitrogen?.toFixed(0) || 'N/A'} ppm
          </div>
          <div className="text-sm text-gray-500">Optimal: 80-200 ppm</div>
        </div>
      </div>

      <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4">
        <h3 className="font-bold text-blue-900 mb-2">💡 Recommendations</h3>
        <ul className="list-disc list-inside space-y-1 text-sm text-blue-800">
          {moistureStatus.status === 'Low' && <li>Consider irrigation - soil moisture is below optimal</li>}
          {moistureStatus.status === 'High' && <li>Reduce watering - soil moisture is above optimal</li>}
          {tempStatus.status === 'High' && <li>Provide shade or cooling during peak hours</li>}
          {tempStatus.status === 'Low' && <li>Protect crops from cold temperatures</li>}
          {humidityStatus.status === 'Low' && <li>Increase irrigation frequency to improve humidity</li>}
          {nitrogenStatus.status === 'Low' && <li>Apply nitrogen fertilizer to improve soil nutrients</li>}
          {nitrogenStatus.status === 'High' && <li>Reduce nitrogen inputs to prevent toxicity</li>}
          {moistureStatus.status === 'Optimal' && tempStatus.status === 'Optimal' && (
            <li>Environmental conditions are optimal - maintain current practices</li>
          )}
        </ul>
      </div>
    </div>
  );
}
