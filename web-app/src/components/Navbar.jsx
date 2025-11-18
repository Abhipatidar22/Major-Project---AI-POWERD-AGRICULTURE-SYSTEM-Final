import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  const navItems = [
    { path: '/', label: '🏠 Home', icon: '🏠' },
    { path: '/crop-recommendation', label: '🌾 Crop Recommendations', icon: '🌾' },
    { path: '/soil-analysis', label: '🔬 Soil Analysis', icon: '🔬' },
    { path: '/yield-prediction', label: '📈 Yield Prediction', icon: '📈' },
    { path: '/disease-detection', label: '🍃 Disease Detection', icon: '🍃' },
    { path: '/location-analysis', label: '📍 Location Analysis', icon: '📍' },
    { path: '/sensors', label: '📡 Sensors', icon: '📡' },
    { path: '/dashboard', label: '📊 Dashboard', icon: '📊' },
  ];

  return (
    <nav className="bg-gradient-to-r from-primary to-secondary text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <span className="text-2xl">🌾</span>
            <span className="text-xl font-bold">Smart Agriculture System</span>
          </Link>
          
          <div className="hidden md:flex space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(item.path)
                    ? 'bg-white bg-opacity-20'
                    : 'hover:bg-white hover:bg-opacity-10'
                }`}
              >
                <span className="mr-1">{item.icon}</span>
                {item.label.replace(/^🏠 |^🌾 |^🔬 |^📈 |^🍃 |^📍 |^📊 /, '')}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
