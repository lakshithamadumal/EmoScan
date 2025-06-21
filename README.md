# EmoScan: Real-Time Facial Emotion Recognition System

🚀 **EmoScan** is a Python-based real-time facial emotion recognition system that detects human emotions via webcam feed. This project leverages deep learning to classify emotions such as Happy, Sad, Angry, Neutral, Surprise, Fear, and Disgust in real time.

---

## 📸 Features
- 🎥 Real-time face detection using webcam
- 😊 Emotion classification with seven emotion classes
- 🖼️ Live emotion label overlay on video stream
- 📊 Optional real-time emotion trend visualization
- 💾 Session logging (CSV file)
- 💻 Simple, user-friendly UI with start/stop controls (Tkinter)

---

## 🛠 Tech Stack
- Python 🐍
- OpenCV 🎥
- DeepFace / FER library 🤖
- TensorFlow / Keras (optional for custom training)
- Tkinter / Flask (for UI)
- Matplotlib (for visualization)

---

## 🚀 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/EmoScan.git
cd EmoScan

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 How to Run

```bash
python main.py
```

### Optional: If using Flask for web interface
```bash
python app.py
```

---

## 📂 Project Structure
```
EmoScan/
├── dataset/               # (Optional) Custom training dataset
├── models/                # Pre-trained emotion detection models
├── logs/                  # Saved session logs (CSV)
├── ui/                    # UI files (if using Tkinter/Flask)
├── main.py                # Main emotion detection script
├── app.py                 # Web interface (optional)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🎥 Demo
![EmoScan Demo](demo.gif)

---

## 🚀 Future Improvements
- Multi-face emotion tracking
- Liveliness detection (anti-spoofing)
- Edge device optimization (Raspberry Pi/Jetson Nano)
- Real-time emotion dashboard with advanced analytics

---

## 🏆 Credits
- Face detection: OpenCV Haar Cascades
- Emotion recognition: Pre-trained CNN via FER/DeepFace library
- Dataset: FER2013 (Kaggle)

---

## 📜 License
This project is licensed under the MIT License.

---

## 💬 Connect with Me
- Instagram: [itz_laky_](https://instagram.com/itz_laky_)
- TikTok: [@l_a_k_y_2](https://tiktok.com/@l_a_k_y_2)
- GitHub: [YourUsername](https://github.com/YourUsername)

> "AI that feels. Tech that connects. EmoScan: Emotion matters."
