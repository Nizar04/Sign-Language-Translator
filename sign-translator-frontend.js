import React, { useState, useEffect, useRef } from 'react';
import { Camera, Settings, Book, Activity, User, Save, RefreshCw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('translate');
  const [translation, setTranslation] = useState('');
  const [language, setLanguage] = useState('asl');
  const [confidence, setConfidence] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [customSigns, setCustomSigns] = useState([]);
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => wsRef.current?.close();
  }, []);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'translation') {
        setTranslation(data.text);
        setConfidence(prev => [...prev, { time: new Date().toLocaleTimeString(), value: data.confidence }]
          .slice(-20));
      }
    };
    wsRef.current.onclose = () => {
      setTimeout(connectWebSocket, 1000);
    };
  };

  const startVideoStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-900">Sign Language Translator</h1>
              </div>
              <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                <NavButton
                  icon={<Camera />}
                  text="Translate"
                  active={activeTab === 'translate'}
                  onClick={() => setActiveTab('translate')}
                />
                <NavButton
                  icon={<Book />}
                  text="Learn"
                  active={activeTab === 'learn'}
                  onClick={() => setActiveTab('learn')}
                />
                <NavButton
                  icon={<Settings />}
                  text="Settings"
                  active={activeTab === 'settings'}
                  onClick={() => setActiveTab('settings')}
                />
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {activeTab === 'translate' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="aspect-w-16 aspect-h-9 mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="rounded-lg w-full h-full object-cover"
                />
              </div>
              <div className="flex justify-between items-center">
                <button
                  onClick={() => {
                    if (!isRecording) startVideoStream();
                    setIsRecording(!isRecording);
                  }}
                  className={`px-4 py-2 rounded-md flex items-center space-x-2 ${
                    isRecording ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'
                  }`}
                >
                  <Camera size={20} />
                  <span>{isRecording ? 'Stop' : 'Start'} Translation</span>
                </button>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="px-4 py-2 rounded-md border border-gray-300"
                >
                  <option value="asl">American Sign Language</option>
                  <option value="bsl">British Sign Language</option>
                </select>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm p-6">
              <h2 className="text-lg font-semibold mb-4">Live Translation</h2>
              <div className="h-32 bg-gray-50 rounded-md p-4 mb-4">
                {translation || 'No translation available...'}
              </div>
              <div className="h-64">
                <LineChart width={500} height={200} data={confidence}>
                  <XAxis dataKey="time" />
                  <YAxis />
                  <CartesianGrid strokeDasharray="3 3" />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" />
                </LineChart>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'learn' && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Custom Signs Training</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-md font-medium mb-2">Record New Sign</h3>
                <div className="space-y-4">
                  <input
                    type="text"
                    placeholder="Sign Label"
                    className="w-full px-4 py-2 rounded-md border border-gray-300"
                  />
                  <button className="w-full px-4 py-2 bg-green-500 text-white rounded-md flex items-center justify-center space-x-2">
                    <Save size={20} />
                    <span>Record Sign</span>
                  </button>
                </div>
              </div>
              <div>
                <h3 className="text-md font-medium mb-2">Your Custom Signs</h3>
                <div className="space-y-2">
                  {customSigns.map((sign, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded-md">
                      <span>{sign.label}</span>
                      <button className="text-blue-500 hover:text-blue-700">
                        <RefreshCw size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-lg font-semibold mb-4">Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Confidence Threshold</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Camera Source</label>
                <select className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
                  <option>Default Camera</option>
                  <option>External Camera</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">Model Language</label>
                <select className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md">
                  <option>American Sign Language</option>
                  <option>British Sign Language</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const NavButton = ({ icon, text, active, onClick }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
      active
        ? 'border-blue-500 text-gray-900'
        : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
    }`}
  >
    {icon}
    <span className="ml-2">{text}</span>
  </button>
);

export default Dashboard;
