#!/usr/bin/env python3
"""
EmoScan: Flask Web Interface
Web-based alternative to the Tkinter UI for emotion recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class WebEmotionDetector:
    """Web-based emotion detector for Flask interface"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.session_data = []
        self.is_running = False
        self.cap = None
        self.face_cascade = None
        self.current_frame = None
        self.fer_model = None
        self.setup_face_detection()
        self.setup_emotion_detection()
        
    def setup_face_detection(self):
        """Initialize face detection cascade classifier"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            logger.info("Face detection cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
    
    def setup_emotion_detection(self):
        """Initialize emotion detection model"""
        try:
            from fer import FER
            self.fer_model = FER(mtcnn=True)
            logger.info("FER emotion detection model loaded successfully")
        except ImportError:
            logger.warning("FER not available, using simplified emotion detection")
            self.fer_model = None
        except Exception as e:
            logger.error(f"Error loading FER model: {e}")
            self.fer_model = None
    
    def detect_emotion(self, face_img):
        """Detect emotion in a face image using FER or simplified detection"""
        try:
            if self.fer_model:
                # Use FER for emotion detection
                result = self.fer_model.predict(face_img)
                if result:
                    dominant_emotion = result[0]['emotions']
                    # Convert FER format to our format
                    emotion_scores = {}
                    for emotion in self.emotions:
                        emotion_scores[emotion] = dominant_emotion.get(emotion, 0.0)
                    
                    # Find dominant emotion
                    dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    return dominant, emotion_scores
                else:
                    return 'neutral', {emotion: 0.0 for emotion in self.emotions}
            else:
                # Simplified emotion detection (fallback)
                return self.detect_emotion_simple(face_img)
                
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return 'neutral', {emotion: 0.0 for emotion in self.emotions}
    
    def detect_emotion_simple(self, face_img):
        """Simplified emotion detection for fallback"""
        import random
        
        # Simple heuristic-based emotion detection
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Simple brightness-based emotion detection
        if brightness > 150:
            dominant_emotion = 'happy'
        elif brightness < 100:
            dominant_emotion = 'sad'
        else:
            dominant_emotion = 'neutral'
        
        # Generate emotion scores
        emotion_scores = {}
        for emotion in self.emotions:
            if emotion == dominant_emotion:
                emotion_scores[emotion] = random.uniform(0.6, 0.9)
            else:
                emotion_scores[emotion] = random.uniform(0.0, 0.3)
        
        return dominant_emotion, emotion_scores
    
    def process_frame(self, frame):
        """Process a single frame for face detection and emotion recognition"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        processed_frame = frame.copy()
        emotions_detected = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            dominant_emotion, emotion_scores = self.detect_emotion(face_roi)
            emotions_detected.append(dominant_emotion)
            
            # Update emotion counts
            self.emotion_counts[dominant_emotion] += 1
            
            # Draw rectangle around face
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add emotion label
            label = f"{dominant_emotion.upper()}"
            cv2.putText(processed_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Log emotion data
            self.log_emotion(dominant_emotion, emotion_scores)
        
        return processed_frame, emotions_detected
    
    def log_emotion(self, dominant_emotion, emotion_scores):
        """Log emotion data to session storage"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = {
            'timestamp': timestamp,
            'dominant_emotion': dominant_emotion,
            **emotion_scores
        }
        self.session_data.append(log_entry)
    
    def save_session_log(self):
        """Save session data to CSV file"""
        if not self.session_data:
            return None
        
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/emotion_session_web_{timestamp}.csv"
        
        df = pd.DataFrame(self.session_data)
        df.to_csv(filename, index=False)
        logger.info(f"Session log saved to {filename}")
        return filename
    
    def start_detection(self):
        """Start the emotion detection process"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            self.is_running = True
            logger.info("Web emotion detection started")
            return True
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            return False
    
    def stop_detection(self):
        """Stop the emotion detection process"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("Web emotion detection stopped")
    
    def get_frame(self):
        """Get current frame for web streaming"""
        if not self.is_running or not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        processed_frame, emotions = self.process_frame(frame)
        self.current_frame = processed_frame
        return processed_frame

# Global detector instance
detector = WebEmotionDetector()

def generate_frames():
    """Generate video frames for web streaming"""
    while detector.is_running:
        frame = detector.get_frame()
        if frame is not None:
            # Encode frame for web streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start emotion detection"""
    try:
        success = detector.start_detection()
        if success:
            return jsonify({'status': 'success', 'message': 'Detection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start detection'})
    except Exception as e:
        logger.error(f"Error in start_detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop emotion detection"""
    try:
        detector.stop_detection()
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        logger.error(f"Error in stop_detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_stats')
def get_stats():
    """Get current emotion statistics"""
    try:
        return jsonify({
            'status': 'success',
            'emotion_counts': detector.emotion_counts,
            'total_detections': sum(detector.emotion_counts.values())
        })
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/save_log', methods=['POST'])
def save_log():
    """Save session log"""
    try:
        filename = detector.save_session_log()
        if filename:
            return jsonify({
                'status': 'success', 
                'message': 'Session log saved',
                'filename': filename
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'No session data to save'
            })
    except Exception as e:
        logger.error(f"Error in save_log: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset emotion statistics"""
    try:
        detector.emotion_counts = {emotion: 0 for emotion in detector.emotions}
        detector.session_data = []
        return jsonify({'status': 'success', 'message': 'Statistics reset'})
    except Exception as e:
        logger.error(f"Error in reset_stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🌐 Starting EmoScan Web Interface")
    print("=" * 50)
    print("Access the application at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Error: {e}")
    finally:
        detector.stop_detection()
