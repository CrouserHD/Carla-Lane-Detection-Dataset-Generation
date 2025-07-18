import os
import sys
from gui.app import App

def main():
    """
    Main entry point for the application.
    Launches the GUI.
    """
    app = App()
    app.mainloop()

if __name__ == "__main__":
    # Add the project root to the Python path to allow for absolute imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main()
