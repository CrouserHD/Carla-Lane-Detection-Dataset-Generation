import cv2
import os
import numpy as np
import argparse  # Added for command-line arguments

# Attempt to import the configuration file
try:
    import lane_comparison_config as cfg
except ImportError:
    print("ERROR: Could not import 'lane_comparison_config.py'.")
    print("Make sure this script is in the same directory as 'lane_comparison_config.py',")
    print("or that 'lane_comparison_config.py' is in your Python path.")
    # Define fallback paths if config is not available, though this is not ideal
    # For the script to be truly useful, lane_comparison_config.py should be importable
    # to get IMAGE_SOURCE_DIR.
    class FallbackConfig:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\John\Desktop\Masterarbeit_Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\Carla-Lane-Detection-Dataset-Generation"))
        IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "debug", "Town03_Opt")

    cfg = FallbackConfig()
    print(f"Warning: Using fallback IMAGE_SOURCE_DIR: {cfg.IMAGE_SOURCE_DIR}")


# Global variables
points = []
original_image = None
image_to_show = None

def calculate_and_print_ratios(clicked_points, H, W):
    """
    Calculates and prints the ROI ratios based on the clicked points.
    Points are expected in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    if len(clicked_points) != 4:
        print("Error: Exactly 4 points are required.")
        return

    P_tl, P_tr, P_br, P_bl = clicked_points

    if H == 0 or W == 0:
        print("Error: Image height or width is zero.")
        return

    # Calculate ratios
    # Top Y is the average of the Y-coordinates of the top-left and top-right points
    roi_y_ratio = (P_tl[1] + P_tr[1]) / (2 * H)
    # Top-left X
    roi_x_start_ratio = P_tl[0] / W
    # Top-right X
    roi_x_end_ratio = P_tr[0] / W
    # Bottom Y is the average of the Y-coordinates of the bottom-left and bottom-right points
    roi_y_end_ratio = (P_bl[1] + P_br[1]) / (2 * H)

    # Calculate ROI_BOTTOM_WIDTH_FACTOR_OFFSET
    # This factor determines how much wider the bottom edge is compared to the top edge, relative to ROI_X_START_RATIO and ROI_X_END_RATIO.
    # offset_from_left = (Expected top_left_x_at_bottom_level) - (Actual clicked bottom_left_x)
    # offset_from_left_ratio = ( (P_tl[0]/W) - (P_bl[0]/W) )
    offset_left_factor = (P_tl[0] - P_bl[0]) / W
    
    # offset_from_right = (Actual clicked bottom_right_x) - (Expected top_right_x_at_bottom_level)
    # offset_from_right_ratio = ( (P_br[0]/W) - (P_tr[0]/W) )
    offset_right_factor = (P_br[0] - P_tr[0]) / W
    
    # Average the two factors, as the config uses a single symmetric offset
    roi_bottom_width_factor_offset = (offset_left_factor + offset_right_factor) / 2

    print("\\n--- Calculated ROI Parameters ---")
    print(f"ROI_Y_RATIO = {roi_y_ratio:.4f}")
    print(f"ROI_X_START_RATIO = {roi_x_start_ratio:.4f}")
    print(f"ROI_X_END_RATIO = {roi_x_end_ratio:.4f}")
    print(f"ROI_Y_END_RATIO = {roi_y_end_ratio:.4f}")
    print(f"ROI_BOTTOM_WIDTH_FACTOR_OFFSET = {roi_bottom_width_factor_offset:.4f}")
    print("\\nCopy these values into your 'lane_comparison_config.py'.")

    print("\\n--- Clicked Points (for reference) ---")
    print(f"  Top-Left:     ({P_tl[0]}, {P_tl[1]})")
    print(f"  Top-Right:    ({P_tr[0]}, {P_tr[1]})")
    print(f"  Bottom-Right: ({P_br[0]}, {P_br[1]})")
    print(f"  Bottom-Left:  ({P_bl[0]}, {P_bl[1]})")
    
    # Verification: Reconstruct trapezoid points using calculated ratios
    # This helps to see if the calculated ratios correctly represent the user's clicks
    # according to the logic in lane_comparison_config.py
    v_top_y = int(H * roi_y_ratio)
    v_bottom_y = int(H * roi_y_end_ratio)
    v_top_left_x = int(W * roi_x_start_ratio)
    v_top_right_x = int(W * roi_x_end_ratio)
    v_bottom_offset_pixels = int(W * roi_bottom_width_factor_offset)
    v_bottom_left_x = v_top_left_x - v_bottom_offset_pixels
    v_bottom_right_x = v_top_right_x + v_bottom_offset_pixels

    print("\\n--- Verification: Trapezoid from calculated ratios ---")
    print(f"  Top-Left:     ({v_top_left_x}, {v_top_y})")
    print(f"  Top-Right:    ({v_top_right_x}, {v_top_y})")
    print(f"  Bottom-Left:  ({v_bottom_left_x}, {v_bottom_y})")
    print(f"  Bottom-Right: ({v_bottom_right_x}, {v_bottom_y})")
    print("---------------------------------")


def mouse_callback(event, x, y, flags, param):
    global points, image_to_show, original_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # Draw a circle at the clicked point
            cv2.circle(image_to_show, (x, y), 5, (0, 255, 0), -1)
            
            # If 4 points are collected, draw the polygon and calculate ratios
            if len(points) == 4:
                # Draw the trapezoid (polygon)
                # Convert points to numpy array for polylines
                poly_points = np.array(points, dtype=np.int32)
                cv2.polylines(image_to_show, [poly_points], isClosed=True, color=(0, 255, 255), thickness=2)
                
                H, W = original_image.shape[:2]
                calculate_and_print_ratios(points, H, W)
        else:
            print("4 points already selected. Press 'r' to reset.")

def main():
    global points, original_image, image_to_show

    # Determine image to load
    image_name = "0167.jpg" # Default image
    try:
        if not os.path.exists(cfg.IMAGE_SOURCE_DIR):
            print(f"Warning: IMAGE_SOURCE_DIR '{cfg.IMAGE_SOURCE_DIR}' does not exist.")
            image_path = image_name # Try loading default from current dir if path fails
        else:
            image_files = [f for f in os.listdir(cfg.IMAGE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                #image_name = image_files[0]
                print(f"Loading image: {os.path.join(cfg.IMAGE_SOURCE_DIR, image_name)}")
            else:
                print(f"No images found in {cfg.IMAGE_SOURCE_DIR}. Attempting to load default '{image_name}'.")
            image_path = os.path.join(cfg.IMAGE_SOURCE_DIR, image_name)

    except AttributeError: # If cfg object (even fallback) doesn't have IMAGE_SOURCE_DIR
        print("Error: IMAGE_SOURCE_DIR not found in config. Cannot load image.")
        return
    except Exception as e:
        print(f"Error accessing IMAGE_SOURCE_DIR or listing images: {e}")
        image_path = image_name # Fallback to trying to load default image name

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from '{image_path}'.")
        print("Please ensure the image exists and the path in 'lane_comparison_config.py' (IMAGE_SOURCE_DIR) is correct.")
        return

    image_to_show = original_image.copy()
    points = [] # Reset points list for each run

    window_name = "ROI Selector - Click 4 points (TL, TR, BR, BL). 'r' to reset, 'q' to quit."
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\\n--- ROI Selector Instructions ---")
    print("1. Click 4 points on the image to define the ROI trapezoid in the following order:")
    print("   - Top-Left (TL)")
    print("   - Top-Right (TR)")
    print("   - Bottom-Right (BR)")
    print("   - Bottom-Left (BL)")
    print("2. After 4 clicks, the calculated ratios will be printed to the console.")
    print("3. Press 'r' to reset the points and the image, then click again.")
    print("4. Press 'q' to close the window and quit the script.")
    print("---------------------------------")

    while True:
        cv2.imshow(window_name, image_to_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            points = []
            image_to_show = original_image.copy()
            print("\\nPoints reset. Click 4 new points on the image.")
            # Optionally, redraw instructions or clear previous ratio output from console if possible

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
