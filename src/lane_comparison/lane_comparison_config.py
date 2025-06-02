"""
Lane Comparison Configuration File
"""
import os
import cv2 # For color definitions if needed directly here

# --- Project Paths ---
# Assuming this config file is in src/lane_comparison/
# PROJECT_ROOT will be two levels up from this file's directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- LaneNet Specific Paths ---
# Path to the LaneNet project directory
LANE_NET_PROJECT_ROOT_CONFIG = r"c:\\Users\\John\\Desktop\\Masterarbeit_Carla\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\LaneNet_Detection\\lanenet-lane-detection"
# Path to the pre-trained LaneNet model weights checkpoint file (prefix, without .ckpt)
LANE_NET_WEIGHTS_PATH_CONFIG = os.path.join(LANE_NET_PROJECT_ROOT_CONFIG, 'weights', 'tusimple_lanenet', 'tusimple_lanenet.ckpt')

JSON_GT_PATH = os.path.join(PROJECT_ROOT, "data", "dataset", "Town03_Opt", "train_gt_tmp.json")
IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "debug", "Town03_Opt")
OUTPUT_VIS_DIR = os.path.join(PROJECT_ROOT, "data", "comparison_results_modular")
OUTPUT_VIDEO_FILENAME = "comparison_video.mp4"
METRICS_SUMMARY_FILENAME = "metrics_summary.json" # Added missing configuration

# --- Processing Settings ---
#MAX_IMAGES_TO_PROCESS = 100  # Set to None or 0 to process all images
MAX_IMAGES_FOR_OPTIMIZER_EVAL = 200 # Max images to use for each optimizer evaluation trial
NUM_IMAGES_FOR_OPTIMIZATION_PHASE = 50 # Total unique images for the entire optimization phase of one algorithm
OPTIMIZER_MAX_ITERATIONS_OVERALL = 10 # Max iterations for the optimizer (e.g. hill climbing passes)

SKIP_IMAGES_WITHOUT_GT = True
VIDEO_FPS = 20.0
AUTO_PLAY_VIDEO_ON_COMPLETION = True # Added new flag
START_IMAGE_INDEX = 925  # 0-based index of the first image to process
PROCESS_NUM_IMAGES = 70 # Number of images to process from START_IMAGE_INDEX. None or 0 means use MAX_IMAGES_TO_PROCESS or all.
RESIZE_PROCESSING_FACTOR = 0.5 # Verarbeitet Bilder mit 50% ihrer Originalgröße


# --- Algorithm Configuration ---
# Define which algorithms to run and their properties
# 'module_name': The name of the .py file in the 'algorithms' subfolder (without .py)
# 'function_name': The name of the detection function within that module
# 'active': Boolean, whether to run this algorithm
# 'color': BGR tuple for visualization
# 'display_name': Name to show in legend/metrics

# Placeholder for algorithm imports - these will be dynamically imported in run_comparison.py
# from .algorithms.hough_transform import detect_lanes_hough
# from .algorithms.dl_placeholder import detect_lanes_dl_placeholder
# from .algorithms.sliding_window import detect_lanes_sliding_window # To be created

ALGORITHMS_TO_RUN = [
    {
        "module_name": "hough_transform",
        "function_name": "detect_lanes_hough",
        "active": True,  # Activate
        "color": (255, 0, 0),
        "display_name": "Hough Transform"
    },
    {
        "module_name": "advanced_sliding_window",
        "function_name": "detect_lanes_advanced_sliding_window",
        "active": True, # Activate
        "color": (255, 165, 0),
        "display_name": "Adv. Sliding Window"
    },
    {
        "module_name": "carnd_pipeline_algorithm",
        "function_name": "detect_lanes_carnd",
        "active": False,
        "color": (0, 255, 255),  # Cyan oder eine andere Farbe
        "display_name": "CarND Pipeline"
    },
    {
        "module_name": "lanenet_algorithm",
        "function_name": "detect_lanes_lanenet",
        "active": False,
        "color": (0, 0, 0), # Changed to black for better visibility
        "display_name": "LaneNet",
        "params": { # Algorithm-specific parameters
            "LANE_NET_WEIGHTS_PATH": LANE_NET_WEIGHTS_PATH_CONFIG,
            "LANE_NET_USE_GPU": True # Set to False to force CPU
        }
    }
]

# --- ROI Parameters ---
# These are ratios of image height/width
ROI_Y_RATIO = 0.3382    # Top of the ROI (y-coordinate)
ROI_X_START_RATIO = 0.3352  # Left side of the ROI (x-coordinate)
ROI_X_END_RATIO = 0.5958 # Increased from 0.5758    # Right side of the ROI (x-coordinate)
ROI_Y_END_RATIO = 0.8458    # Bottom of the ROI (y-coordinate)
ROI_BOTTOM_WIDTH_FACTOR_OFFSET = 0.3457 #  # Width of the bottom of the ROI (x-coordinate) - this is a factor of the image width

# --- Dynamically Calculated Source Ratios for Perspective Transforms ---
# These will be used by algorithms like Advanced Sliding Window and CarND Pipeline.
# Format: [[Top-Left_X, Top-Left_Y], [Top-Right_X, Top-Right_Y], 
#          [Bottom-Left_X, Bottom-Left_Y], [Bottom-Right_X, Bottom-Right_Y]]

# Top-Left
_src_tl_x = ROI_X_START_RATIO
_src_tl_y = ROI_Y_RATIO
# Top-Right
_src_tr_x = ROI_X_END_RATIO
_src_tr_y = ROI_Y_RATIO
# Bottom-Left
_src_bl_x = max(0.0, ROI_X_START_RATIO - ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
_src_bl_y = ROI_Y_END_RATIO
# Bottom-Right
_src_br_x = min(1.0, ROI_X_END_RATIO + ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
_src_br_y = ROI_Y_END_RATIO

# Universal Source Ratios based on the primary ROI parameters
# This ensures all algorithms use a consistent source trapezoid derived from the main ROI.
UNIVERSAL_SRC_RATIOS = [
    [_src_tl_x, _src_tl_y],  # Top-Left
    [_src_tr_x, _src_tr_y],  # Top-Right
    [_src_bl_x, _src_bl_y],  # Bottom-Left
    [_src_br_x, _src_br_y]   # Bottom-Right
]

# --- Ground Truth Visualization ---
GT_COLOR = (0, 255, 0)  # Green for Ground Truth
GT_THICKNESS = 2 # Reset to a visible thickness

# --- Algorithm Visualization ---
ALGO_LANE_THICKNESS = 3 # Increased Algo lane thickness for max visibility
DEFAULT_ALGO_COLOR = (255, 255, 255)  # Default color (white) for algorithms if not specified

# --- Canny Edge Detection Parameters (example, if needed globally or passed to algos) ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- Hough Transform Parameters (example, if needed globally or passed to algos) ---
HOUGH_RHO = 1 # Updated from sweep: prev 2
HOUGH_THETA_DIVISOR = 180 # Updated from sweep: prev 180 (np.pi / 180)
HOUGH_THRESHOLD = 20 # Updated from sweep: prev 50 
HOUGH_MIN_LINE_LENGTH = 17 # Updated from sweep: prev 50 
HOUGH_MAX_LINE_GAP = 5 # Updated from sweep: prev 100 
# New slope parameters for Hough Transform
HOUGH_SLOPE_LEFT_MIN = -5.0  # Min slope for left lane lines (more negative = steeper) 
HOUGH_SLOPE_LEFT_MAX = -0.4  # Max slope for left lane lines (less negative = less steep)
HOUGH_SLOPE_RIGHT_MIN = 0.4   # Min slope for right lane lines (less positive = less steep)
HOUGH_SLOPE_RIGHT_MAX = 5.0   # Max slope for right lane lines (more positive = steeper)
# Additional parameters for refined Hough Transform
HOUGH_GAUSSIAN_BLUR_KERNEL = (5,5) # Kernel size for Gaussian Blur
HOUGH_GAUSSIAN_BLUR_SIGMA_X = 0 # SigmaX for Gaussian Blur
HOUGH_CANNY_LOW_THRESHOLD = 25 # Updated from sweep: prev 50
HOUGH_CANNY_HIGH_THRESHOLD = 95 # Updated from sweep: prev 150
HOUGH_MAX_HORIZONTAL_SLOPE_DEVIATION = 0.15 # Max abs(slope) to be considered horizontal (and thus filtered out)
HOUGH_MIN_SEGMENTS_FOR_LANE_FIT = 2 # Min raw Hough segments to consider fitting a polynomial for a lane side
HOUGH_MIN_POINTS_FOR_POLYFIT = 5  # Minimum number of points in a segment to attempt polynomial fitting

# --- Advanced Sliding Window Parameters (ASW prefix) ---
ASW_NWINDOWS = 9                 # Number of sliding windows
ASW_MARGIN = 80                  # Width of the windows +/- margin
ASW_MINPIX = 40                  # Minimum number of pixels found to recenter window
ASW_POLY_DEGREE = 2              # Degree of the polynomial to fit to lane lines
ASW_S_THRESH_MIN = 170           # Min threshold for S channel (HLS)
ASW_S_THRESH_MAX = 255           # Max threshold for S channel (HLS)
ASW_SOBEL_KERNEL_SIZE = 5        # Kernel size for Sobel operator
ASW_SOBEL_THRESH_MIN = 30        # Min threshold for Sobel X gradient
ASW_SOBEL_THRESH_MAX = 150       # Max threshold for Sobel X gradient
# Source points for perspective transform (ratios of image width/height)
# Derived from ROI_ parameters:
# Top-left X: ROI_X_START_RATIO
# Top-left Y: ROI_Y_RATIO
# Top-right X: ROI_X_END_RATIO
# Top-right Y: ROI_Y_RATIO
# Bottom-left X: max(0.0, ROI_X_START_RATIO - ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
# Bottom-left Y: ROI_Y_END_RATIO
# Bottom-right X: min(1.0, ROI_X_END_RATIO + ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
# Bottom-right Y: ROI_Y_END_RATIO
ASW_SRC_RATIOS = [
    [0.3352, 0.3382], [0.5958, 0.3382], # Top-left, Top-right
    [0.0, 0.8458], [0.9215, 0.8458]  # Bottom-left, Bottom-right
]
ASW_SRC_RATIOS = UNIVERSAL_SRC_RATIOS # Use the dynamically calculated universal ratios !!!!!!!!!!!!!!THIS IS OVERWRITING THE PREVIOUS ONE
# Destination points for perspective transform (ratios of warped image width/height)
ASW_DST_RATIOS = [
    [0.25, 0.0], [0.75, 0.0],   # Top-left, Top-right
    [0.25, 1.0], [0.75, 1.0]    # Bottom-left, Bottom-right
]

ASW_MIN_LANE_DIST_WARPED = 30    # Minimum distance between left and right lanes in warped view (pixels)

# --- CarND Pipeline Specific Parameters ---
# IMPORTANT: You MUST tune these source and destination ratios for the CarND algorithm.
# The quality of the perspective transform is CRITICAL for this pipeline.
# The order for SRC_RATIOS and DST_RATIOS is: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]
# These points define a trapezoid in the original image (SRC) that should correspond to a
# rectangular region in the warped bird\'s-eye view (DST).
# 
# Example ratios for a 1280x720 image (adjust to your image dimensions and lane appearance):
# Find points that form a trapezoid around a straight section of lane lines.
# The top of the trapezoid should be near the horizon where lanes appear to converge.
# The bottom of the trapezoid should be near the bottom of the image, capturing the lanes broadly.

# Define default thresholds for CarND pipeline before they are used in OPTIMIZATION_SETTINGS
CARND_RGB_THRESH = (190, 255) # Example default, adjust as needed
CARND_HLS_THRESH = (190, 255) # Example default, adjust as needed

CARND_SRC_RATIOS = [
    [0.453, 0.35],  # Top-Left (Y adjusted from 0.639)
    [0.547, 0.35],  # Top-Right (Y adjusted from 0.639)
    [0.150, 0.950],  # Bottom-Left
    [0.875, 0.950]   # Bottom-Right
]
CARND_SRC_RATIOS = UNIVERSAL_SRC_RATIOS # Use the dynamically calculated universal ratios !!!!!!!!!!!!!!THIS IS OVERWRITING THE PREVIOUS ONE

CARND_DST_RATIOS = [ # These are often a good default
    [0.25, 0.0],    # Top-Left
    [0.75, 0.0],    # Top-Right
    [0.25, 1.0],    # Bottom-Left
    [0.75, 1.0]     # Bottom-Right
]

# Quantitative Evaluation Metrics Configuration
GROUND_TRUTH_JSON_FILE = os.path.join(PROJECT_ROOT, "data", "dataset", "Town03_Opt", "train_gt.json")  # Path to the ground truth JSON file (TuSimple format)
LANE_METRICS_THRESHOLD_PX = 10  # Pixel threshold for considering a detected point correct against ground truth
H_SAMPLES = list(range(240, 1080, 50)) # Y-coordinates for evaluating lane accuracy, adjust as per your dataset

OPTIMIZATION_SETTINGS = {
    "hough_transform": {
        "enabled": True,
        "evaluation_image_subset_count": 30, # Number of images for quick F1 eval
        "parameters_to_tune": {
            "HOUGH_THRESHOLD": {"min": 10, "max": 50, "step": 1, "initial_guess": 20},
            "HOUGH_MIN_LINE_LENGTH": {"min": 10, "max": 40, "step": 1, "initial_guess": 18},
            # ... other integer params for Hough
        }
    },
    "advanced_sliding_window": {
        "enabled": True,
        "evaluation_image_subset_count": 30,
        "parameters_to_tune": {
            "ASW_NWINDOWS": {"min": 5, "max": 15, "step": 1, "initial_guess": 9},
            "ASW_MARGIN": {"min": 20, "max": 100, "step": 5, "initial_guess": 50},
            "ASW_MINPIX": {"min": 20, "max": 80, "step": 5, "initial_guess": 40},
        }
    },
    "carnd_pipeline_algorithm": {
        "enabled": True,
        "evaluation_image_subset_count": 30,
        "parameters_to_tune": {
            # For tuple params like (min_thresh, max_thresh), we'd optimize one part.
            # Need a way to specify which part, e.g. by a special key or convention.
            # Let's assume we optimize the 'min' part of threshold tuples.
            "CARND_RGB_THRESH": {"min": 150, "max": 220, "step": 1, "initial_guess": 190, "tuple_index_to_tune": 0},
            "CARND_HLS_THRESH": {"min": 150, "max": 220, "step": 1, "initial_guess": 190, "tuple_index_to_tune": 0}
            # The 'tuple_index_to_tune' would tell the optimizer to modify only that element of the tuple.
        }
    }
    # No entry for "lanenet" as per your request
}