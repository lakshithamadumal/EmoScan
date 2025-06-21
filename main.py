#!/usr/bin/env python3
"""
EmoScan: Real-Time Facial Emotion Recognition System
Main application script with Tkinter UI and real-time emotion detection
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from datetime import datetime
from deepface import DeepFace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionDetector:
    """Main emotion detection class with real-time processing capabilities"""
    
    def __init__(self):
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        self.emotion_counts = {emotion: 0 for emotion in self.emotions}
        self.session_data = []
        self.is_running = False
        self.cap = None
        self.face_cascade = None
        self.setup_face_detection()
        
    def setup_face_detection(self):
        """Initialize face detection cascade classifier"""
        try:
            # Try to load the cascade file from OpenCV
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            logger.info("Face detection cascade loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            messagebox.showerror("Error", "Failed to load face detection model")
    
    def detect_emotion(self, face_img):
        """Detect emotion in a face image using DeepFace"""
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            emotion_scores = result[0]['emotion']
            return dominant_emotion, emotion_scores
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return 'neutral', {emotion: 0.0 for emotion in self.emotions}
    
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
            return
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/emotion_session_{timestamp}.csv"
        
        # Save to CSV
        df = pd.DataFrame(self.session_data)
        df.to_csv(filename, index=False)
        logger.info(f"Session log saved to {filename}")
        return filename

class EmoScanUI:
    """Tkinter-based user interface for EmoScan"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EmoScan: Real-Time Facial Emotion Recognition")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        self.detector = EmotionDetector()
        self.video_thread = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface components"""
        # Main title
        title_label = tk.Label(self.root, text="EmoScan", 
                              font=("Arial", 24, "bold"), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(self.root, text="Real-Time Facial Emotion Recognition", 
                                 font=("Arial", 12), 
                                 fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack(pady=5)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(pady=20)
        
        # Start button
        self.start_button = tk.Button(control_frame, text="Start Detection", 
                                     command=self.start_detection,
                                     font=("Arial", 12, "bold"),
                                     bg='#27ae60', fg='white',
                                     width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        # Stop button
        self.stop_button = tk.Button(control_frame, text="Stop Detection", 
                                    command=self.stop_detection,
                                    font=("Arial", 12, "bold"),
                                    bg='#e74c3c', fg='white',
                                    width=15, height=2,
                                    state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Save log button
        self.save_button = tk.Button(control_frame, text="Save Session Log", 
                                    command=self.save_log,
                                    font=("Arial", 12, "bold"),
                                    bg='#3498db', fg='white',
                                    width=15, height=2)
        self.save_button.pack(side=tk.LEFT, padx=10)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Video frame
        video_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_label = tk.Label(video_frame, text="Video Feed", 
                              font=("Arial", 14, "bold"), 
                              fg='#ecf0f1', bg='#34495e')
        video_label.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, bg='black', width=640, height=480)
        self.video_label.pack(pady=10)
        
        # Statistics frame
        stats_frame = tk.Frame(content_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        stats_label = tk.Label(stats_frame, text="Emotion Statistics", 
                              font=("Arial", 14, "bold"), 
                              fg='#ecf0f1', bg='#34495e')
        stats_label.pack(pady=10)
        
        # Emotion counters
        self.emotion_labels = {}
        for emotion in self.detector.emotions:
            frame = tk.Frame(stats_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            label = tk.Label(frame, text=f"{emotion.title()}:", 
                           font=("Arial", 10), 
                           fg='#ecf0f1', bg='#34495e', width=10, anchor='w')
            label.pack(side=tk.LEFT)
            
            count_label = tk.Label(frame, text="0", 
                                 font=("Arial", 10, "bold"), 
                                 fg='#f39c12', bg='#34495e', width=8, anchor='w')
            count_label.pack(side=tk.RIGHT)
            
            self.emotion_labels[emotion] = count_label
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start emotion detection")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W,
                             font=("Arial", 10), 
                             fg='#ecf0f1', bg='#34495e')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def start_detection(self):
        """Start the emotion detection process"""
        try:
            self.detector.cap = cv2.VideoCapture(0)
            if not self.detector.cap.isOpened():
                raise Exception("Could not open webcam")
            
            self.detector.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Emotion detection started")
            
            # Start video processing thread
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            
            logger.info("Emotion detection started")
            
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop the emotion detection process"""
        self.detector.is_running = False
        if self.detector.cap:
            self.detector.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Emotion detection stopped")
        
        logger.info("Emotion detection stopped")
    
    def process_video(self):
        """Process video frames in a separate thread"""
        while self.detector.is_running:
            ret, frame = self.detector.cap.read()
            if not ret:
                break
            
            # Process frame for emotion detection
            processed_frame, emotions = self.detector.process_frame(frame)
            
            # Convert frame for Tkinter display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.resize(rgb_frame, (640, 480))
            
            # Update video display
            img = tk.PhotoImage(data=cv2.imencode('.ppm', rgb_frame)[1].tobytes())
            self.video_label.configure(image=img)
            self.video_label.image = img
            
            # Update emotion counters
            self.update_emotion_counts()
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
    
    def update_emotion_counts(self):
        """Update emotion count displays"""
        for emotion, count in self.detector.emotion_counts.items():
            self.emotion_labels[emotion].config(text=str(count))
    
    def save_log(self):
        """Save the current session log"""
        if not self.detector.session_data:
            messagebox.showwarning("Warning", "No session data to save")
            return
        
        filename = self.detector.save_session_log()
        if filename:
            messagebox.showinfo("Success", f"Session log saved to:\n{filename}")
        else:
            messagebox.showerror("Error", "Failed to save session log")
    
    def run(self):
        """Start the UI main loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self.stop_detection()

def main():
    """Main entry point"""
    print("🚀 Starting EmoScan: Real-Time Facial Emotion Recognition System")
    print("=" * 60)
    
    try:
        app = EmoScanUI()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()
