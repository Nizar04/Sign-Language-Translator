# Advanced Sign Language Translator

A production-grade sign language translation system that supports real-time translation, finger spelling, sentence construction, and multi-language support with a modern web interface.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.8+-orange.svg)
![React](https://img.shields.io/badge/react-v18.2+-61dafb.svg)

## 🌟 Features

- **Real-time Translation**
  - Multi-language support (ASL, BSL)
  - Finger spelling detection
  - Natural sentence construction
  - Confidence scoring

- **Advanced Computer Vision**
  - High-precision hand tracking
  - Multiple hand support
  - Real-time performance
  - Custom gesture recognition

- **Web Interface**
  - Modern React dashboard
  - Real-time visualization
  - Custom sign training
  - Progress tracking

- **Production Ready**
  - FastAPI backend
  - WebSocket support
  - SQLite database
  - Error handling
  - Performance optimized

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Nizar04/Sign-Language-Translator.git
cd sign-language-translator
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
cd frontend && npm install
```

4. Download pre-trained models:
```bash
python scripts/download_models.py
```

5. Start the application:
```bash
# Terminal 1: Backend
python src/main.py

# Terminal 2: Frontend
cd frontend && npm start
```

6. Open http://localhost:3000 in your browser

## 📁 Project Structure

```
sign-language-translator/
├── src/
│   ├── main.py              # Application entry point
│   ├── translator/          # Core translation logic
│   ├── models/             # ML model implementations
│   └── utils/              # Helper functions
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/         # Custom React hooks
│   │   └── pages/         # Page components
│   └── public/            # Static assets
├── models/                # Pre-trained model files
├── data/                 # Language vocabularies and configs
├── scripts/             # Utility scripts
└── tests/              # Test suite
```

## 🛠️ Technologies Used

### Backend
- Python 3.8+
- TensorFlow/PyTorch
- MediaPipe
- FastAPI
- SQLite
- OpenCV

### Frontend
- React 18
- TailwindCSS
- Recharts
- Lucide Icons
- WebSocket

## 📋 Requirements

- Python 3.8+
- Node.js 14+
- Webcam access
- 8GB+ RAM
- CUDA-compatible GPU (optional, for better performance)

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model parameters
- Language settings
- Performance options
- API configurations

## 📝 API Documentation

API documentation is available at `http://localhost:8000/docs` when running the backend server.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking
- T5 team for sentence construction
- FastAPI community
- React community

## 📬 Contact

Nizar El Mouaquit- nizarelmouaquit@protonmail.com
