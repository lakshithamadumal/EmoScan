#!/usr/bin/env python3
"""
EmoScan: Launcher Script
Easy launcher for different EmoScan components
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the EmoScan banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 EmoScan: Real-Time Facial Emotion Recognition System    ║
    ║                                                              ║
    ║  AI that feels. Tech that connects. Emotion matters.        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'opencv-python',
        'deepface',
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_main_app():
    """Run the main Tkinter application"""
    print("🎯 Starting EmoScan Main Application (Tkinter UI)...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running main application: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Main application stopped by user")

def run_web_app():
    """Run the Flask web application"""
    print("🌐 Starting EmoScan Web Interface...")
    print("📱 Access the application at: http://localhost:5000")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web application: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Web application stopped by user")

def run_dashboard():
    """Run the emotion visualization dashboard"""
    print("📊 Starting EmoScan Emotion Dashboard...")
    try:
        subprocess.run([sys.executable, "ui/emotion_visualizer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")

def download_models():
    """Download required models"""
    print("🤖 Downloading EmoScan models...")
    try:
        subprocess.run([sys.executable, "models/download_models.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading models: {e}")

def validate_config():
    """Validate configuration"""
    print("⚙️  Validating EmoScan configuration...")
    try:
        subprocess.run([sys.executable, "config.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Configuration validation failed: {e}")

def install_dependencies():
    """Install project dependencies"""
    print("📦 Installing EmoScan dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")

def show_help():
    """Show help information"""
    help_text = """
EmoScan Launcher - Available Commands:

🎯  main     - Start the main Tkinter application
🌐  web      - Start the Flask web interface
📊  dashboard - Start the emotion visualization dashboard
🤖  models   - Download required models
⚙️   config   - Validate configuration
📦  install  - Install dependencies
❓  help     - Show this help message

Examples:
  python run.py main
  python run.py web
  python run.py dashboard
  python run.py models
  python run.py install

For more information, see README.md
    """
    print(help_text)

def main():
    """Main launcher function"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="EmoScan: Real-Time Facial Emotion Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py main     # Start main application
  python run.py web      # Start web interface
  python run.py models   # Download models
        """
    )
    
    parser.add_argument(
        'command',
        choices=['main', 'web', 'dashboard', 'models', 'config', 'install', 'help'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies before running command'
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
    
    # Execute command
    if args.command == 'main':
        run_main_app()
    elif args.command == 'web':
        run_web_app()
    elif args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'models':
        download_models()
    elif args.command == 'config':
        validate_config()
    elif args.command == 'install':
        install_dependencies()
    elif args.command == 'help':
        show_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1) 