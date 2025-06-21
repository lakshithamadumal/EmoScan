#!/usr/bin/env python3
"""
EmoScan: Emotion Visualizer Component
Real-time emotion visualization with charts and graphs
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from collections import deque
import threading
import time

class EmotionVisualizer:
    """Real-time emotion visualization component"""
    
    def __init__(self, parent_frame, max_points=100):
        self.parent = parent_frame
        self.max_points = max_points
        self.emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6', '#f39c12', '#9b59b6', '#e67e22']
        
        # Data storage for real-time plotting
        self.time_data = deque(maxlen=max_points)
        self.emotion_data = {emotion: deque(maxlen=max_points) for emotion in self.emotions}
        
        # Matplotlib setup
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.patch.set_facecolor('#2c3e50')
        
        # Setup plots
        self.setup_plots()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Animation
        self.ani = None
        self.is_running = False
        
    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Line plot for emotion trends
        self.ax1.set_facecolor('#34495e')
        self.ax1.set_title('Real-Time Emotion Trends', color='white', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Time', color='white')
        self.ax1.set_ylabel('Emotion Score', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Bar plot for current emotion distribution
        self.ax2.set_facecolor('#34495e')
        self.ax2.set_title('Current Emotion Distribution', color='white', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Emotions', color='white')
        self.ax2.set_ylabel('Count', color='white')
        self.ax2.tick_params(colors='white')
        
        # Set emotion labels
        self.ax2.set_xticks(range(len(self.emotions)))
        self.ax2.set_xticklabels([emotion.title() for emotion in self.emotions], rotation=45)
        
        plt.tight_layout()
        
    def add_data_point(self, timestamp, emotion_scores):
        """Add a new data point for visualization"""
        self.time_data.append(timestamp)
        
        for emotion in self.emotions:
            score = emotion_scores.get(emotion, 0.0)
            self.emotion_data[emotion].append(score)
    
    def update_plots(self, frame):
        """Update the plots with new data"""
        if not self.is_running or len(self.time_data) < 2:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Update line plot
        self.ax1.set_facecolor('#34495e')
        self.ax1.set_title('Real-Time Emotion Trends', color='white', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Time', color='white')
        self.ax1.set_ylabel('Emotion Score', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot emotion lines
        for i, emotion in enumerate(self.emotions):
            if len(self.emotion_data[emotion]) > 0:
                self.ax1.plot(list(self.time_data), list(self.emotion_data[emotion]), 
                            color=self.colors[i], label=emotion.title(), linewidth=2)
        
        self.ax1.legend(facecolor='#34495e', edgecolor='white', fontsize=10)
        
        # Update bar plot
        self.ax2.set_facecolor('#34495e')
        self.ax2.set_title('Current Emotion Distribution', color='white', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Emotions', color='white')
        self.ax2.set_ylabel('Count', color='white')
        self.ax2.tick_params(colors='white')
        
        # Get current emotion counts
        current_counts = []
        for emotion in self.emotions:
            if len(self.emotion_data[emotion]) > 0:
                current_counts.append(self.emotion_data[emotion][-1])
            else:
                current_counts.append(0)
        
        # Create bar plot
        bars = self.ax2.bar(range(len(self.emotions)), current_counts, 
                           color=self.colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, count in zip(bars, current_counts):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{count:.1f}', ha='center', va='bottom', color='white', fontweight='bold')
        
        self.ax2.set_xticks(range(len(self.emotions)))
        self.ax2.set_xticklabels([emotion.title() for emotion in self.emotions], rotation=45)
        
        plt.tight_layout()
    
    def start_animation(self):
        """Start the real-time animation"""
        if not self.is_running:
            self.is_running = True
            self.ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                             interval=1000, blit=False)
            self.canvas.draw()
    
    def stop_animation(self):
        """Stop the real-time animation"""
        self.is_running = False
        if self.ani:
            self.ani.event_source.stop()
    
    def clear_data(self):
        """Clear all visualization data"""
        self.time_data.clear()
        for emotion in self.emotions:
            self.emotion_data[emotion].clear()
    
    def save_chart(self, filename):
        """Save the current chart as an image"""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='#2c3e50', edgecolor='none')
            return True
        except Exception as e:
            print(f"Error saving chart: {e}")
            return False

class EmotionDashboard:
    """Complete emotion dashboard with multiple visualization components"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("EmoScan: Emotion Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        self.setup_ui()
        self.visualizer = None
        
    def setup_ui(self):
        """Setup the dashboard UI"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = tk.Label(title_frame, text="EmoScan Emotion Dashboard", 
                              font=("Arial", 24, "bold"), 
                              fg='#ecf0f1', bg='#2c3e50')
        title_label.pack()
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#2c3e50')
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Control buttons
        self.start_btn = tk.Button(control_frame, text="Start Visualization", 
                                  command=self.start_visualization,
                                  font=("Arial", 12, "bold"),
                                  bg='#27ae60', fg='white',
                                  width=15, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Visualization", 
                                 command=self.stop_visualization,
                                 font=("Arial", 12, "bold"),
                                 bg='#e74c3c', fg='white',
                                 width=15, height=2,
                                 state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(control_frame, text="Clear Data", 
                                  command=self.clear_data,
                                  font=("Arial", 12, "bold"),
                                  bg='#3498db', fg='white',
                                  width=15, height=2)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(control_frame, text="Save Chart", 
                                 command=self.save_chart,
                                 font=("Arial", 12, "bold"),
                                 bg='#f39c12', fg='white',
                                 width=15, height=2)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#34495e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create visualizer
        self.visualizer = EmotionVisualizer(content_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start visualization")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W,
                             font=("Arial", 10), 
                             fg='#ecf0f1', bg='#34495e')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_visualization(self):
        """Start the emotion visualization"""
        try:
            self.visualizer.start_animation()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Visualization active")
            
            # Start data simulation (for demo purposes)
            self.start_data_simulation()
            
        except Exception as e:
            print(f"Error starting visualization: {e}")
            self.status_var.set("Error starting visualization")
    
    def stop_visualization(self):
        """Stop the emotion visualization"""
        try:
            self.visualizer.stop_animation()
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("Visualization stopped")
            
            # Stop data simulation
            self.stop_data_simulation()
            
        except Exception as e:
            print(f"Error stopping visualization: {e}")
            self.status_var.set("Error stopping visualization")
    
    def clear_data(self):
        """Clear all visualization data"""
        try:
            self.visualizer.clear_data()
            self.status_var.set("Data cleared")
        except Exception as e:
            print(f"Error clearing data: {e}")
            self.status_var.set("Error clearing data")
    
    def save_chart(self):
        """Save the current chart"""
        try:
            import os
            from datetime import datetime
            
            os.makedirs('logs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/emotion_chart_{timestamp}.png"
            
            if self.visualizer.save_chart(filename):
                self.status_var.set(f"Chart saved: {filename}")
            else:
                self.status_var.set("Error saving chart")
                
        except Exception as e:
            print(f"Error saving chart: {e}")
            self.status_var.set("Error saving chart")
    
    def start_data_simulation(self):
        """Start simulated data generation for demo purposes"""
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self.simulate_data, daemon=True)
        self.simulation_thread.start()
    
    def stop_data_simulation(self):
        """Stop simulated data generation"""
        self.simulation_running = False
    
    def simulate_data(self):
        """Simulate emotion data for demonstration"""
        import random
        
        while self.simulation_running:
            timestamp = time.time()
            
            # Generate random emotion scores
            emotion_scores = {}
            for emotion in self.visualizer.emotions:
                # Simulate realistic emotion patterns
                base_score = random.uniform(0, 0.3)
                if random.random() < 0.1:  # 10% chance of high emotion
                    base_score = random.uniform(0.5, 1.0)
                emotion_scores[emotion] = base_score
            
            # Add data point
            self.visualizer.add_data_point(timestamp, emotion_scores)
            
            time.sleep(2)  # Update every 2 seconds
    
    def run(self):
        """Start the dashboard"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Dashboard interrupted by user")
        finally:
            self.stop_visualization()

def main():
    """Main function for standalone dashboard"""
    print("📊 Starting EmoScan Emotion Dashboard")
    print("=" * 50)
    
    try:
        root = tk.Tk()
        dashboard = EmotionDashboard(root)
        dashboard.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 