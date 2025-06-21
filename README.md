# EmoScan: Real-Time Facial Emotion Recognition System

🚀 **EmoScan** is a comprehensive Python-based real-time facial emotion recognition system that detects human emotions via webcam feed. This project leverages deep learning to classify emotions such as Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust in real time.

---

## 📸 Features

### Core Features
- 🎥 **Real-time face detection** using OpenCV Haar Cascades
- 😊 **Emotion classification** with seven emotion classes using DeepFace
- 🖼️ **Live emotion label overlay** on video stream
- 💾 **Session logging** (CSV file) with detailed emotion scores
- 💻 **Multiple UI options**: Tkinter desktop app and Flask web interface
- 📊 **Real-time emotion visualization** with charts and graphs

### Advanced Features
- 🔄 **Multi-face detection** and tracking
- 📈 **Live emotion trend visualization**
- 🎨 **Modern, responsive UI** with dark theme
- ⚙️ **Configurable settings** via environment variables
- 🧪 **Comprehensive test suite**
- 📦 **Easy launcher script** for all components

---

## 🛠 Tech Stack

- **Python 3.8+** 🐍
- **OpenCV** 🎥 - Computer vision and face detection
- **DeepFace** 🤖 - Emotion recognition using pre-trained CNN
- **TensorFlow** - Deep learning backend
- **Tkinter** - Desktop GUI framework
- **Flask** - Web framework for browser interface
- **Matplotlib** - Data visualization and real-time charts
- **Pandas** - Data manipulation and CSV logging
- **NumPy** - Numerical computing

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam access
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/EmoScan.git
cd EmoScan
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
python run.py models
```

### Running the Application

#### Option 1: Main Desktop Application (Tkinter)
```bash
python run.py main
# or
python main.py
```

#### Option 2: Web Interface (Flask)
```bash
python run.py web
# or
python app.py
```
Then open your browser to: `http://localhost:5000`

#### Option 3: Emotion Visualization Dashboard
```bash
python run.py dashboard
# or
python ui/emotion_visualizer.py
```

#### Option 4: Using the Launcher Script
```bash
# Show all available commands
python run.py help

# Install dependencies
python run.py install

# Validate configuration
python run.py config

# Run tests
python test_emotion_detection.py
```

---

## 📂 Project Structure

```
EmoScan/
├── 📁 dataset/                 # Custom training dataset (optional)
├── 📁 models/                  # Pre-trained emotion detection models
│   ├── download_models.py      # Model downloader utility
│   └── model_info.txt          # Model information
├── 📁 logs/                    # Saved session logs (CSV)
│   └── sample_session.csv      # Example log file
├── 📁 ui/                      # UI components
│   ├── templates/              # Flask HTML templates
│   │   └── index.html          # Web interface template
│   └── emotion_visualizer.py   # Real-time visualization dashboard
├── 🐍 main.py                  # Main Tkinter application
├── 🌐 app.py                   # Flask web interface
├── ⚙️ config.py                # Configuration settings
├── 🚀 run.py                   # Launcher script
├── 🧪 test_emotion_detection.py # Test suite
├── 📋 requirements.txt         # Python dependencies
└── 📖 README.md                # Project documentation
```

---

## 🎯 Usage Guide

### Desktop Application (Tkinter)

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Using the interface**
   - Click "Start Detection" to begin emotion recognition
   - Your webcam feed will appear with real-time emotion labels
   - View emotion statistics in the right panel
   - Click "Save Session Log" to export data to CSV
   - Click "Stop Detection" to end the session

3. **Features**
   - Real-time video feed with emotion overlays
   - Live emotion counters
   - Session data logging
   - Modern dark theme UI

### Web Interface (Flask)

1. **Start the web server**
   ```bash
   python app.py
   ```

2. **Access the interface**
   - Open your browser to `http://localhost:5000`
   - Click "Start Detection" to begin
   - View real-time video feed and statistics
   - Use "Save Session Log" to export data
   - Click "Reset Stats" to clear counters

3. **Features**
   - Responsive web design
   - Real-time video streaming
   - Live emotion statistics
   - Mobile-friendly interface

### Emotion Visualization Dashboard

1. **Start the dashboard**
   ```bash
   python ui/emotion_visualizer.py
   ```

2. **Features**
   - Real-time emotion trend charts
   - Current emotion distribution bars
   - Data simulation for demonstration
   - Chart export functionality

---

## 📊 Data Logging

### CSV Log Format
The system automatically logs emotion data to CSV files with the following structure:

```csv
timestamp,dominant_emotion,happy,sad,angry,neutral,surprise,fear,disgust
2024-01-15 14:30:15.123,happy,85.2,2.1,1.5,8.9,1.2,0.8,0.3
2024-01-15 14:30:16.456,happy,82.7,3.2,1.8,9.1,1.5,1.2,0.5
```

### Log File Location
- **Desktop app**: `logs/emotion_session_YYYYMMDD_HHMMSS.csv`
- **Web interface**: `logs/emotion_session_web_YYYYMMDD_HHMMSS.csv`

---

## ⚙️ Configuration

### Environment Variables
You can customize the system using environment variables:

```bash
# Camera settings
export EMOSCAN_CAMERA_INDEX=0

# Web interface settings
export EMOSCAN_WEB_PORT=5000

# Debug mode
export EMOSCAN_DEBUG_MODE=true
```

### Configuration File
Edit `config.py` to modify:
- Emotion detection parameters
- Video processing settings
- UI themes and colors
- Logging preferences
- Performance settings

---

## 🧪 Testing

### Run the Test Suite
```bash
python test_emotion_detection.py
```

### Test Coverage
- ✅ Emotion detector initialization
- ✅ Face detection cascade loading
- ✅ Frame processing functionality
- ✅ Emotion logging and CSV export
- ✅ Web interface components
- ✅ Configuration validation
- ✅ Data structure integrity

---

## 🔧 Troubleshooting

### Common Issues

1. **Camera not found**
   ```
   Error: Could not open webcam
   ```
   **Solution**: Check camera permissions and try different camera indices

2. **Model download fails**
   ```
   Error: Failed to load face cascade
   ```
   **Solution**: Run `python run.py models` to download required models

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'deepface'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

4. **Performance issues**
   - Reduce video resolution in `config.py`
   - Enable GPU acceleration if available
   - Close other applications using the camera

### Performance Optimization

1. **Reduce processing load**
   ```python
   # In config.py
   VIDEO_SETTINGS = {
       'frame_width': 320,  # Reduce from 640
       'frame_height': 240, # Reduce from 480
       'processing_interval': 0.05  # Increase from 0.03
   }
   ```

2. **Enable GPU acceleration**
   ```python
   # Install GPU version of TensorFlow
   pip install tensorflow-gpu
   ```

---

## 🚀 Advanced Usage

### Custom Model Training

1. **Prepare your dataset**
   ```
   dataset/
   ├── happy/
   ├── sad/
   ├── angry/
   └── ...
   ```

2. **Train custom model**
   ```python
   # Use the dataset directory for custom training
   # See DeepFace documentation for training details
   ```

### Multi-face Detection

The system automatically detects and tracks multiple faces:
- Each face gets its own emotion label
- Statistics are aggregated across all detected faces
- Performance scales with number of faces

### Integration with Other Systems

```python
from main import EmotionDetector

# Create detector instance
detector = EmotionDetector()

# Process a single image
frame = cv2.imread('image.jpg')
processed_frame, emotions = detector.process_frame(frame)

# Get emotion data
print(f"Detected emotions: {emotions}")
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**
   ```bash
   python test_emotion_detection.py
   ```
6. **Submit a pull request**

### Development Setup

```bash
# Clone and setup
git clone https://github.com/YourUsername/EmoScan.git
cd EmoScan
pip install -r requirements.txt

# Run tests
python test_emotion_detection.py

# Check configuration
python config.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepFace** - Emotion recognition models and framework
- **OpenCV** - Computer vision and face detection
- **TensorFlow** - Deep learning backend
- **FER2013 Dataset** - Training data for emotion recognition

---

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/YourUsername/EmoScan/issues)
- **Documentation**: [Wiki](https://github.com/YourUsername/EmoScan/wiki)
- **Discussions**: [Community forum](https://github.com/YourUsername/EmoScan/discussions)

### Social Media
- Instagram: [itz_laky_](https://instagram.com/itz_laky_)
- TikTok: [@l_a_k_y_2](https://tiktok.com/@l_a_k_y_2)
- GitHub: [YourUsername](https://github.com/YourUsername)

---

## 🎯 Roadmap

### Upcoming Features
- [ ] **Liveness detection** (anti-spoofing)
- [ ] **Edge device optimization** (Raspberry Pi/Jetson Nano)
- [ ] **Real-time emotion dashboard** with advanced analytics
- [ ] **API endpoints** for integration
- [ ] **Mobile app** (React Native/Flutter)
- [ ] **Cloud deployment** options

### Version History
- **v1.0.0** - Initial release with Tkinter and Flask interfaces
- **v1.1.0** - Added real-time visualization dashboard
- **v1.2.0** - Enhanced configuration and testing
- **v2.0.0** - Multi-face detection and performance improvements

---

> **"AI that feels. Tech that connects. EmoScan: Emotion matters."** 🚀

---

**⭐ If you find this project useful, please give it a star on GitHub!**
