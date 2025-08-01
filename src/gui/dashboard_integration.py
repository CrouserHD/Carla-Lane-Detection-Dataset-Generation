"""
Dashboard Integration for Lane Comparison Tool
This module provides integration between the main processing and the web dashboard.
"""

import requests
import json
import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DashboardIntegration:
    """
    Handles communication between the main processing and the web dashboard.
    """
    
    def __init__(self, dashboard_url: str = "http://localhost:5002", enabled: bool = True):
        self.dashboard_url = dashboard_url.rstrip('/')
        self.enabled = enabled
        self.session = requests.Session()
        self.session.timeout = 1.0  # Short timeout for non-blocking updates
        
        # Test connection
        if self.enabled:
            try:
                response = self.session.get(f"{self.dashboard_url}/api/progress", timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"Dashboard connection established: {self.dashboard_url}")
                else:
                    logger.warning(f"Dashboard not responding properly: {response.status_code}")
                    self.enabled = False
            except Exception as e:
                logger.warning(f"Dashboard not available: {e}")
                self.enabled = False
    
    def _send_update(self, update_data: Dict):
        """Send update to dashboard API"""
        if not self.enabled:
            return
        
        try:
            response = self.session.post(
                f"{self.dashboard_url}/api/update",
                json=update_data,
                timeout=1.0
            )
            if response.status_code != 200:
                logger.debug(f"Dashboard update failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"Dashboard update error: {e}")
    
    def start_processing(self, total_images: int, algorithms: List[str]):
        """Notify dashboard that processing has started"""
        self._send_update({
            'type': 'start',
            'total_images': total_images,
            'algorithms': algorithms
        })
    
    def update_phase(self, phase_name: str):
        """Update current processing phase"""
        self._send_update({
            'type': 'phase',
            'phase': phase_name
        })
    
    def update_image_progress(self, current_image: str, processed_count: int):
        """Update image processing progress"""
        self._send_update({
            'type': 'image',
            'image': current_image,
            'processed': processed_count
        })
    
    def update_algorithm_progress(self, algorithm: str, progress: float):
        """Update algorithm-specific progress"""
        self._send_update({
            'type': 'algorithm',
            'algorithm': algorithm,
            'progress': progress
        })
    
    def add_log(self, message: str, level: str = 'info'):
        """Add a log message to the dashboard"""
        self._send_update({
            'type': 'log',
            'message': message,
            'level': level
        })
    
    def update_preview_image(self, image_path: str):
        """Update the preview image"""
        self._send_update({
            'type': 'preview',
            'image_path': image_path
        })
    
    def update_preview_video(self, video_path: str):
        """Update the preview video"""
        self._send_update({
            'type': 'video',
            'video_path': video_path
        })
    
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        self._send_update({
            'type': 'metrics',
            'metrics': metrics
        })
    
    def processing_complete(self):
        """Notify dashboard that processing is complete"""
        self._send_update({
            'type': 'complete'
        })
    
    def processing_error(self, error_message: str):
        """Notify dashboard of an error"""
        self._send_update({
            'type': 'error',
            'message': error_message
        })

class DashboardLogger(logging.Handler):
    """
    Custom logging handler that sends log messages to the dashboard.
    """
    
    def __init__(self, dashboard_integration: DashboardIntegration):
        super().__init__()
        self.dashboard = dashboard_integration
        self.setLevel(logging.INFO)
    
    def emit(self, record):
        try:
            level_map = {
                'DEBUG': 'info',
                'INFO': 'info',
                'WARNING': 'warning',
                'ERROR': 'error',
                'CRITICAL': 'error'
            }
            
            level = level_map.get(record.levelname, 'info')
            message = self.format(record)
            
            self.dashboard.add_log(message, level)
        except Exception:
            pass  # Don't let dashboard logging interfere with main processing

# Global dashboard instance
_dashboard_instance: Optional[DashboardIntegration] = None

def initialize_dashboard(dashboard_url: str = "http://localhost:5000", enabled: bool = True) -> DashboardIntegration:
    """Initialize the global dashboard instance"""
    global _dashboard_instance
    _dashboard_instance = DashboardIntegration(dashboard_url, enabled)
    return _dashboard_instance

def get_dashboard() -> Optional[DashboardIntegration]:
    """Get the global dashboard instance"""
    return _dashboard_instance

def setup_dashboard_logging(logger_instance: logging.Logger):
    """Setup dashboard logging for the given logger"""
    if _dashboard_instance and _dashboard_instance.enabled:
        handler = DashboardLogger(_dashboard_instance)
        handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        logger_instance.addHandler(handler)
