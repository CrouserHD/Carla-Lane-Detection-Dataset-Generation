#!/usr/bin/env python3
"""
Main entry point for the lane comparison tool with optional web dashboard.
Usage:
    python main.py                  # Run without dashboard
    python main.py --dashboard      # Run with web dashboard
    python main.py --dashboard --port 8080  # Run with dashboard on custom port
    python main.py --start_index 100 --num_images 50  # Standard processing args
"""

import sys
import argparse
import threading
import time
import webbrowser
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Lane Comparison Tool')
    
    # Dashboard arguments
    parser.add_argument('--dashboard', action='store_true', help='Start web dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port (default: 5000)')
    parser.add_argument('--host', default='localhost', help='Dashboard host (default: localhost)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t auto-open browser')
    
    # Processing arguments (pass-through to run_comparison.py)
    parser.add_argument('--start_index', type=int, help='0-based index of first image to process')
    parser.add_argument('--num_images', type=int, help='Number of images to process')
    
    return parser.parse_args()

def start_dashboard(host='localhost', port=5000, open_browser=True):
    """Start the web dashboard in a separate thread"""
    try:
        from src.gui.web_dashboard import run_dashboard
        
        def run_server():
            run_dashboard(host=host, port=port)
        
        # Start dashboard in background thread
        dashboard_thread = threading.Thread(target=run_server, daemon=True)
        dashboard_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open browser
        if open_browser:
            dashboard_url = f"http://{host}:{port}"
            print(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        
        return True
        
    except ImportError as e:
        print(f"Dashboard dependencies not installed: {e}")
        print("To use the dashboard, install: pip install flask flask-socketio")
        return False
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return False

def main():
    args = parse_args()
    
    dashboard_started = False
    if args.dashboard:
        print("Starting web dashboard...")
        dashboard_started = start_dashboard(
            host=args.host, 
            port=args.port,
            open_browser=not args.no_browser
        )
        
        if dashboard_started:
            print(f"Dashboard running at http://{args.host}:{args.port}")
            print("Dashboard is ready - you can now control processing from the web interface")
            print("Press Ctrl+C to stop the dashboard")
            
            # Keep the dashboard running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down dashboard...")
                return
        else:
            print("Continuing without dashboard...")
    
    # If no dashboard or dashboard failed, run processing directly
    if not dashboard_started:
        # Prepare arguments for run_comparison.py by filtering out dashboard-specific args
        processing_args = []
        if args.start_index is not None:
            processing_args.extend(['--start_index', str(args.start_index)])
        if args.num_images is not None:
            processing_args.extend(['--num_images', str(args.num_images)])
        
        # Temporarily modify sys.argv to pass only processing arguments
        original_argv = sys.argv[:]
        sys.argv = ['main.py'] + processing_args
        
        try:
            from src.lane_comparison.run_comparison import main_comparison_orchestrator
            
            # Run the main comparison
            main_comparison_orchestrator()
                
        except Exception as e:
            print(f"Error running lane comparison: {e}")
            sys.exit(1)
        finally:
            # Restore original argv
            sys.argv = original_argv

if __name__ == "__main__":
    main()
