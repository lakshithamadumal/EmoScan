#!/usr/bin/env python3
"""
EmoScan: Windows Installation Script
Handles Windows-specific installation issues including long path problems
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """Print installation banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 EmoScan: Windows Installation Helper                    ║
    ║                                                              ║
    ║  Resolving Windows-specific installation issues             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_windows_version():
    """Check Windows version and long path support"""
    print("🔍 Checking Windows configuration...")
    
    # Check Windows version
    win_version = platform.platform()
    print(f"   Windows Version: {win_version}")
    
    # Check if running on Windows
    if not platform.system() == "Windows":
        print("❌ This script is designed for Windows systems only.")
        return False
    
    return True

def enable_long_paths():
    """Provide instructions to enable long path support"""
    print("\n⚠️  Windows Long Path Issue Detected")
    print("=" * 50)
    print("To resolve this issue, you need to enable long path support:")
    print()
    print("Method 1: Using Registry Editor (Recommended)")
    print("1. Press Win + R, type 'regedit', press Enter")
    print("2. Navigate to: HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem")
    print("3. Find 'LongPathsEnabled' and set it to 1")
    print("4. Restart your computer")
    print()
    print("Method 2: Using PowerShell (Run as Administrator)")
    print("1. Open PowerShell as Administrator")
    print("2. Run: New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
    print("3. Restart your computer")
    print()
    print("Method 3: Alternative Installation (No restart required)")
    print("We'll use lighter alternatives that don't require long path support")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    return choice

def install_alternative_dependencies():
    """Install alternative dependencies that work better on Windows"""
    print("\n📦 Installing alternative dependencies...")
    
    # Create alternative requirements file
    alt_requirements = """# Alternative requirements for Windows (no long path issues)
opencv-python-headless==4.8.1.78
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
Pillow==10.0.0
scikit-learn==1.3.0
seaborn==0.12.2
flask==2.3.2
flask-cors==4.0.0
gunicorn==21.2.0
tensorflow-cpu==2.12.0
fer==22.4.0
tkinter-tooltip==2.0.0
"""
    
    with open('requirements_windows.txt', 'w') as f:
        f.write(alt_requirements)
    
    try:
        # Install alternative requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_windows.txt"
        ], check=True)
        print("✅ Alternative dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing alternative dependencies: {e}")
        return False

def install_with_pip_options():
    """Install with pip options to handle long paths"""
    print("\n📦 Installing with pip options to handle long paths...")
    
    try:
        # Install with --no-cache-dir to avoid long path issues
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--user", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully with pip options!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing with pip options: {e}")
        return False

def create_simple_emotion_detector():
    """Create a simplified emotion detector that doesn't require DeepFace"""
    print("\n🔧 Creating simplified emotion detector...")
    
    simple_detector = '''#!/usr/bin/env python3
"""
EmoScan: Simplified Emotion Detector for Windows
Alternative implementation that doesn't require DeepFace
"""

import cv2
import numpy as np
import random
from datetime import datetime

class SimpleEmotionDetector:
    """Simplified emotion detector for Windows compatibility"""
    
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
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            print("✅ Face detection cascade loaded successfully")
        except Exception as e:
            print(f"❌ Error loading face cascade: {e}")
    
    def detect_emotion_simple(self, face_img):
        """Simplified emotion detection (simulation for demo)"""
        # This is a simplified version that simulates emotion detection
        # In a real implementation, you would use a lighter emotion recognition model
        
        # Simulate emotion detection based on image characteristics
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Simple heuristic-based emotion detection
        if brightness > 150:
            dominant_emotion = 'happy'
        elif brightness < 100:
            dominant_emotion = 'sad'
        else:
            dominant_emotion = 'neutral'
        
        # Generate random emotion scores for demonstration
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
            dominant_emotion, emotion_scores = self.detect_emotion_simple(face_roi)
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
        
        import os
        import pandas as pd
        
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/emotion_session_simple_{timestamp}.csv"
        
        df = pd.DataFrame(self.session_data)
        df.to_csv(filename, index=False)
        print(f"✅ Session log saved to {filename}")
        return filename

if __name__ == "__main__":
    print("🚀 EmoScan Simple Emotion Detector")
    print("This is a simplified version for Windows compatibility")
    print("It simulates emotion detection for demonstration purposes")
    
    detector = SimpleEmotionDetector()
    print("✅ Simple emotion detector created successfully!")
'''
    
    with open('simple_emotion_detector.py', 'w') as f:
        f.write(simple_detector)
    
    print("✅ Simple emotion detector created!")

def main():
    """Main installation function"""
    print_banner()
    
    if not check_windows_version():
        return
    
    print("\n🔧 Windows Installation Helper")
    print("=" * 40)
    print("This script will help you resolve Windows-specific installation issues.")
    print()
    
    choice = enable_long_paths()
    
    if choice == "1":
        print("\n📋 Please follow the registry editor instructions above.")
        print("After enabling long paths and restarting, run:")
        print("   pip install -r requirements.txt")
        
    elif choice == "2":
        print("\n📋 Please follow the PowerShell instructions above.")
        print("After enabling long paths and restarting, run:")
        print("   pip install -r requirements.txt")
        
    elif choice == "3":
        print("\n🔄 Installing alternative dependencies...")
        
        # Try alternative installation methods
        if install_alternative_dependencies():
            create_simple_emotion_detector()
            print("\n✅ Alternative installation completed!")
            print("\n📋 Next steps:")
            print("1. Run: python simple_emotion_detector.py")
            print("2. Or try: python run.py main")
        else:
            print("\n❌ Alternative installation failed.")
            print("Please try enabling long path support (options 1 or 2).")
    
    else:
        print("\n❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 