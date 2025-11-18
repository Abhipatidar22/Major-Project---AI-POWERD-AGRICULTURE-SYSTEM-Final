import SensorData from '../components/SensorData';

export default function Home() {
  const features = [
    { icon: '🌾', title: 'Crop Varieties', value: '14+', desc: 'Including Basmati Rice, Durum Wheat, Bt Cotton' },
    { icon: '📊', title: 'Accuracy', value: '60.4%', desc: 'Model accuracy across all crop varieties' },
    { icon: '🗺️', title: 'Coverage', value: '15 States', desc: 'Major agricultural states in India' },
  ];

  const modules = [
    { path: '/crop-recommendation', icon: '🌾', title: 'Crop Recommendations', desc: 'Get AI-powered crop variety suggestions based on your soil and environmental conditions' },
    { path: '/soil-analysis', icon: '🔬', title: 'Soil Analysis', desc: 'Analyze soil images to determine soil type, pH, and nutrient levels' },
    { path: '/yield-prediction', icon: '📈', title: 'Yield Prediction', desc: 'Predict expected crop yields based on historical data and conditions' },
    { path: '/disease-detection', icon: '🍃', title: 'Disease Detection', desc: 'Detect plant diseases from leaf images using AI' },
    { path: '/location-analysis', icon: '📍', title: 'Location Analysis', desc: 'Get location-specific recommendations for 15 Indian states' },
    { path: '/dashboard', icon: '📊', title: 'Dashboard', desc: 'View comprehensive analytics and insights' },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-primary mb-4">🌾 Smart Agriculture System</h1>
        <p className="text-xl text-gray-600">🤖 AI-Powered Crop Recommendations | 📍 Real-time Location Analysis for India</p>
      </div>

      <div className="bg-gradient-to-r from-green-100 to-blue-100 rounded-lg p-8 mb-12">
        <h2 className="text-3xl font-bold text-primary mb-4">🚀 Get Started in 3 Simple Steps:</h2>
        <div className="space-y-4">
          <div className="flex items-start space-x-4">
            <span className="text-3xl">📍</span>
            <div>
              <h3 className="text-xl font-semibold text-secondary">Choose Your Location</h3>
              <p className="text-gray-700">Select your state and region for location-specific recommendations</p>
            </div>
          </div>
          <div className="flex items-start space-x-4">
            <span className="text-3xl">📊</span>
            <div>
              <h3 className="text-xl font-semibold text-secondary">Input Your Data</h3>
              <p className="text-gray-700">Manual entry, smart simulation, or CSV upload</p>
            </div>
          </div>
          <div className="flex items-start space-x-4">
            <span className="text-3xl">🌾</span>
            <div>
              <h3 className="text-xl font-semibold text-secondary">Get Recommendations</h3>
              <p className="text-gray-700">Receive AI-powered crop variety suggestions with confidence scores</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {features.map((feature, idx) => (
          <div key={idx} className="bg-white rounded-lg shadow-lg p-6 text-center transform hover:scale-105 transition-transform">
            <div className="text-4xl mb-3">{feature.icon}</div>
            <h3 className="text-xl font-bold text-primary mb-2">{feature.title}</h3>
            <div className="text-3xl font-bold text-secondary mb-2">{feature.value}</div>
            <p className="text-gray-600">{feature.desc}</p>
          </div>
        ))}
      </div>

      <div className="mb-12">
        <h2 className="text-3xl font-bold text-primary mb-6 text-center">📡 Live Sensor Monitoring</h2>
        <SensorData />
      </div>

      <h2 className="text-3xl font-bold text-primary mb-6 text-center">🎯 Available Features</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modules.map((module, idx) => (
          <a
            key={idx}
            href={module.path}
            className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transform hover:-translate-y-1 transition-all"
          >
            <div className="text-5xl mb-4">{module.icon}</div>
            <h3 className="text-xl font-bold text-secondary mb-2">{module.title}</h3>
            <p className="text-gray-600">{module.desc}</p>
          </a>
        ))}
      </div>
    </div>
  );
}
