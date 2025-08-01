#!/usr/bin/env python3
"""
Web Dashboard for Lane Comparison Tool
Real-time progress monitoring via Flask web interface
"""

import os
import json
import time
import threading
import subprocess
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import logging

# Configure logging for the web dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
template_folder = os.path.join(project_root, 'templates')

app = Flask(__name__, template_folder=template_folder)
app.config['SECRET_KEY'] = 'lane_comparison_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for progress tracking
progress_state = {
    'status': 'idle',
    'current_phase': 'Waiting to start...',
    'total_images': 0,
    'processed_images': 0,
    'current_image': '',
    'algorithms': [],
    'algorithm_progress': {},
    'start_time': None,
    'estimated_completion': None,
    'current_fps': 0,
    'log_messages': [],
    'preview_image': None,
    'preview_video': None,
    'metrics': {}
}

# Processing control state
processing_control = {
    'process': None,
    'is_running': False,
    'should_stop': False
}

class ProgressTracker:
    """Class to track and broadcast progress updates"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        global progress_state
        progress_state.update({
            'status': 'idle',
            'current_phase': 'Waiting to start...',
            'total_images': 0,
            'processed_images': 0,
            'current_image': '',
            'algorithms': [],
            'algorithm_progress': {},
            'start_time': None,
            'estimated_completion': None,
            'current_fps': 0,
            'log_messages': [],
            'preview_image': None,
            'metrics': {}
        })
    
    def start_processing(self, total_images, algorithms):
        global progress_state
        progress_state.update({
            'status': 'running',
            'current_phase': 'Initializing...',
            'total_images': total_images,
            'processed_images': 0,
            'algorithms': algorithms,
            'algorithm_progress': {algo: 0 for algo in algorithms},
            'start_time': time.time(),
            'log_messages': []
        })
        self.broadcast_update()
    
    def update_phase(self, phase_name):
        global progress_state
        progress_state['current_phase'] = phase_name
        self.add_log(f"Phase: {phase_name}")
        self.broadcast_update()
    
    def update_image_progress(self, current_image, processed_count):
        global progress_state
        progress_state['current_image'] = current_image
        progress_state['processed_images'] = processed_count
        
        # Calculate progress percentage
        if progress_state['total_images'] > 0:
            progress_percent = (processed_count / progress_state['total_images']) * 100
            
            # Estimate completion time
            if progress_state['start_time'] and processed_count > 0:
                elapsed = time.time() - progress_state['start_time']
                estimated_total = elapsed * (progress_state['total_images'] / processed_count)
                estimated_remaining = estimated_total - elapsed
                progress_state['estimated_completion'] = estimated_remaining
                progress_state['current_fps'] = processed_count / elapsed
        
        self.broadcast_update()
    
    def update_algorithm_progress(self, algorithm, progress):
        global progress_state
        progress_state['algorithm_progress'][algorithm] = progress
        self.broadcast_update()
    
    def add_log(self, message, level='info'):
        global progress_state
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        progress_state['log_messages'].append(log_entry)
        
        # Keep only last 100 messages
        if len(progress_state['log_messages']) > 100:
            progress_state['log_messages'] = progress_state['log_messages'][-100:]
        
        self.broadcast_update()
    
    def update_preview_image(self, image_path):
        """Update the preview image with the current comparison result"""
        try:
            if os.path.exists(image_path):
                # Read and encode image as base64
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize for web display
                    height, width = img.shape[:2]
                    if width > 800:
                        scale = 800 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    
                    # Encode to base64
                    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    global progress_state
                    progress_state['preview_image'] = img_base64
                    self.broadcast_update()
        except Exception as e:
            logger.error(f"Error updating preview image: {e}")
    
    def update_preview_video(self, video_path):
        """Update the preview with a video file"""
        try:
            if os.path.exists(video_path):
                # Convert to relative path for web serving
                relative_path = os.path.relpath(video_path, project_root)
                video_url = f"/video/{relative_path.replace(os.sep, '/')}"
                
                global progress_state
                progress_state['preview_video'] = video_url
                progress_state['preview_image'] = None  # Clear image when showing video
                self.broadcast_update()
        except Exception as e:
            logger.error(f"Error updating preview video: {e}")
    
    def update_metrics(self, metrics):
        global progress_state
        progress_state['metrics'] = metrics
        self.broadcast_update()
    
    def complete_processing(self):
        global progress_state
        progress_state['status'] = 'completed'
        progress_state['current_phase'] = 'Completed!'
        self.add_log("Processing completed successfully!", 'success')
        self.broadcast_update()
    
    def error_occurred(self, error_message):
        global progress_state
        progress_state['status'] = 'error'
        progress_state['current_phase'] = 'Error occurred'
        self.add_log(f"Error: {error_message}", 'error')
        self.broadcast_update()
    
    def broadcast_update(self):
        """Broadcast progress update to all connected clients"""
        socketio.emit('progress_update', progress_state)

# Global tracker instance
tracker = ProgressTracker()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/progress')
def get_progress():
    return jsonify(progress_state)

@app.route('/api/start', methods=['POST'])
def start_processing():
    """API endpoint to start processing (called by main script)"""
    data = request.json
    total_images = data.get('total_images', 0)
    algorithms = data.get('algorithms', [])
    
    tracker.start_processing(total_images, algorithms)
    return jsonify({'status': 'started'})

@app.route('/api/update', methods=['POST'])
def update_progress():
    """API endpoint for progress updates from main script"""
    data = request.json
    update_type = data.get('type')
    
    if update_type == 'phase':
        tracker.update_phase(data.get('phase'))
    elif update_type == 'image':
        tracker.update_image_progress(data.get('image'), data.get('processed'))
    elif update_type == 'algorithm':
        tracker.update_algorithm_progress(data.get('algorithm'), data.get('progress'))
    elif update_type == 'log':
        tracker.add_log(data.get('message'), data.get('level', 'info'))
    elif update_type == 'preview':
        tracker.update_preview_image(data.get('image_path'))
    elif update_type == 'video':
        tracker.update_preview_video(data.get('video_path'))
    elif update_type == 'metrics':
        tracker.update_metrics(data.get('metrics'))
    elif update_type == 'complete':
        tracker.complete_processing()
    elif update_type == 'error':
        tracker.error_occurred(data.get('message'))
    
    return jsonify({'status': 'updated'})

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """API endpoint to start lane detection processing"""
    try:
        data = request.json
        algorithms = data.get('algorithms', ['hough_transform', 'advanced_sliding_window'])
        start_index = data.get('start_index', 0)
        num_images = data.get('num_images', 100)
        
        # Check if already processing
        if processing_control['is_running']:
            return jsonify({'success': False, 'message': 'Processing already running'})
        
        # Build command
        python_script = os.path.join(project_root, 'src', 'lane_comparison', 'run_comparison.py')
        cmd = [
            'python', python_script,
            '--start_index', str(start_index),
            '--num_images', str(num_images),
            '--dashboard'
        ]
        
        # Add algorithm parameters
        for algo in algorithms:
            cmd.extend(['--algorithm', algo])
        
        # Start processing in background
        def run_processing():
            global processing_control
            processing_control['is_running'] = True
            processing_control['should_stop'] = False
            
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processing_control['process'] = process
                
                # Wait for process to complete
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    tracker.add_log("Processing completed successfully", "success")
                    tracker.complete_processing()
                else:
                    tracker.add_log(f"Processing failed: {stderr}", "error")
                    tracker.error_occurred(stderr)
                    
            except Exception as e:
                tracker.add_log(f"Error running process: {e}", "error")
                tracker.error_occurred(str(e))
            finally:
                processing_control['is_running'] = False
                processing_control['process'] = None
        
        # Start in background thread
        thread = threading.Thread(target=run_processing)
        thread.daemon = True
        thread.start()
        
        # Initialize progress
        tracker.update_phase("Starting processing...")
        tracker.add_log(f"Starting processing with {len(algorithms)} algorithms", "info")
        
        return jsonify({'success': True, 'message': 'Processing started'})
        
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop_processing', methods=['POST'])
def stop_processing():
    """API endpoint to stop lane detection processing"""
    try:
        if processing_control['is_running'] and processing_control['process']:
            processing_control['should_stop'] = True
            processing_control['process'].terminate()
            
            # Wait briefly for clean shutdown
            try:
                processing_control['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                processing_control['process'].kill()
            
            processing_control['is_running'] = False
            processing_control['process'] = None
            
            tracker.add_log("Processing stopped by user", "warning")
            tracker.update_phase("Stopped")
            
            return jsonify({'success': True, 'message': 'Processing stopped'})
        else:
            return jsonify({'success': False, 'message': 'No processing running'})
            
    except Exception as e:
        logger.error(f"Error stopping processing: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files for preview"""
    try:
        video_path = os.path.join(project_root, filename)
        if os.path.exists(video_path):
            from flask import send_file
            return send_file(video_path, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        logger.error(f"Error serving video: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    emit('progress_update', progress_state)

@socketio.on('disconnect')
def handle_disconnect():
    pass

def run_dashboard(host='localhost', port=5000):
    """Run the web dashboard"""
    print(f"Starting web dashboard at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_dashboard()
