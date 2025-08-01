#!/usr/bin/env python3
"""
Simple Lane Comparison Tool Launcher
Automatically detects ob Dashboard-Dependencies verfügbar sind und startet entsprechend.
"""

import sys
import os
import subprocess
import importlib.util

def check_dashboard_dependencies():
    """Check if dashboard dependencies are available"""
    required_packages = ['flask', 'flask_socketio', 'requests']
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False, package
    return True, None

def install_dashboard_dependencies():
    """Install dashboard dependencies"""
    print("📦 Installing dashboard dependencies...")
    packages = ['flask>=2.0.0', 'flask-socketio>=5.0.0', 'eventlet>=0.33.0', 'requests>=2.25.0']
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚗 Lane Comparison Tool Launcher")
    print("=" * 50)
    
    # Check if dashboard dependencies are available
    deps_available, missing_package = check_dashboard_dependencies()
    
    if not deps_available:
        print(f"⚠️  Dashboard dependency '{missing_package}' not found.")
        print("\nOptions:")
        print("1. Install dependencies and use dashboard (recommended)")
        print("2. Run without dashboard")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            
            if choice == '1':
                if install_dashboard_dependencies():
                    print("✅ Dependencies installed successfully!")
                    deps_available = True
                    break
                else:
                    print("❌ Failed to install dependencies. Running without dashboard.")
                    deps_available = False
                    break
            elif choice == '2':
                deps_available = False
                break
            elif choice == '3':
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Prepare command based on availability
    if deps_available:
        print("\n🌐 Starting with web dashboard...")
        print("Dashboard will be available at: http://localhost:5000")
        cmd = [sys.executable, 'main.py', '--dashboard']
    else:
        print("\n⚡ Starting without dashboard...")
        cmd = [sys.executable, 'main.py']
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    try:
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running lane comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
