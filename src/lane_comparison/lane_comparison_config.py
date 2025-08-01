"""
Lane Comparison Configuration File
"""
import os

# --- Project Paths ---
# Assuming this config file is in src/lane_comparison/
# PROJECT_ROOT will be two levels up from this file's directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- LaneNet Specific Paths ---
LANE_NET_PROJECT_ROOT_CONFIG = r"c:\\Users\\John\\Desktop\\Masterarbeit_Carla\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\LaneNet_Detection\\lanenet-lane-detection"
LANE_NET_WEIGHTS_PATH_CONFIG = os.path.join(LANE_NET_PROJECT_ROOT_CONFIG, 'weights', 'tusimple_lanenet', 'tusimple_lanenet.ckpt')

# --- Data Paths ---
JSON_GT_PATH = os.path.join(PROJECT_ROOT, "data", "dataset", "Town03_Opt", "train_gt_tmp.json")
IMAGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "debug", "Town03_Opt")
OUTPUT_VIS_DIR = os.path.join(PROJECT_ROOT, "data", "comparison_results_modular")
GROUND_TRUTH_JSON_FILE = os.path.join(PROJECT_ROOT, "data", "dataset", "Town03_Opt", "train_gt.json")

# --- Output Settings ---
OUTPUT_VIDEO_FILENAME = "comparison_video.mp4"
METRICS_SUMMARY_FILENAME = "metrics_summary.json"
VIDEO_FPS = 20.0
AUTO_PLAY_VIDEO_ON_COMPLETION = True

# --- Processing Settings ---
START_IMAGE_INDEX = 0
PROCESS_NUM_IMAGES = 10000 # Process a large number of images by default
RESIZE_PROCESSING_FACTOR = 0.5
SKIP_IMAGES_WITHOUT_GT = True

# --- Optimization Settings ---
MAX_IMAGES_FOR_OPTIMIZER_EVAL = 200
NUM_IMAGES_FOR_OPTIMIZATION_PHASE = 50
OPTIMIZER_MAX_ITERATIONS_OVERALL = 10

# --- Algorithm Configuration ---
# Master dictionary for all available algorithms.
ALGORITHMS = {
    "hough_transform": {
        "module_name": "hough_transform",
        "function_name": "detect_lanes_hough",
        "display_name": "Hough Transform",
        "color": (255, 0, 0),
        "params": {}
    },
    "advanced_sliding_window": {
        "module_name": "advanced_sliding_window",
        "function_name": "detect_lanes_advanced_sliding_window",
        "display_name": "Adv. Sliding Window",
        "color": (255, 165, 0),
        "params": {}
    },
    "carnd_pipeline_algorithm": {
        "module_name": "carnd_pipeline_algorithm",
        "function_name": "detect_lanes_carnd",
        "display_name": "CarND Pipeline",
        "color": (0, 255, 255),
        "params": {}
    },
    "lanenet": {
        "module_name": "lanenet_algorithm",
        "function_name": "detect_lanes_lanenet",
        "display_name": "LaneNet",
        "color": (0, 0, 0),
        "params": {
            "LANE_NET_WEIGHTS_PATH": LANE_NET_WEIGHTS_PATH_CONFIG,
            "LANE_NET_USE_GPU": True
        }
    }
}

# List of algorithm keys to run by default. Can be overridden by the web dashboard.
ALGORITHMS_TO_RUN_KEYS = [
    "hough_transform",
    "advanced_sliding_window",
    "lanenet",
]

# --- Parameter Optimization Settings ---
# Defines the search space for tuning algorithm parameters.
OPTIMIZATION_SETTINGS = {
    "hough_transform": {
        "enabled": False,
        "parameters_to_tune": {
            "HOUGH_THRESHOLD": {"min": 10, "max": 50, "step": 1, "initial_guess": 20},
            "HOUGH_MIN_LINE_LENGTH": {"min": 10, "max": 40, "step": 1, "initial_guess": 18},
        }
    },
    "advanced_sliding_window": {
        "enabled": False,
        "parameters_to_tune": {
            "ASW_NWINDOWS": {"min": 5, "max": 15, "step": 1, "initial_guess": 9},
            "ASW_MARGIN": {"min": 20, "max": 100, "step": 5, "initial_guess": 50},
            "ASW_MINPIX": {"min": 20, "max": 80, "step": 5, "initial_guess": 40},
        }
    },
    "carnd_pipeline_algorithm": {
        "enabled": False,
        "parameters_to_tune": {
            "CARND_RGB_THRESH": {"min": 150, "max": 220, "step": 1, "initial_guess": 190, "tuple_index_to_tune": 0},
            "CARND_HLS_THRESH": {"min": 150, "max": 220, "step": 1, "initial_guess": 190, "tuple_index_to_tune": 0}
        }
    }
}

# --- ROI & Perspective Transform Parameters ---
ROI_Y_RATIO = 0.3382
ROI_X_START_RATIO = 0.3352
ROI_X_END_RATIO = 0.5958
ROI_Y_END_RATIO = 0.8458
ROI_BOTTOM_WIDTH_FACTOR_OFFSET = 0.3457

_src_tl_x = ROI_X_START_RATIO
_src_tl_y = ROI_Y_RATIO
_src_tr_x = ROI_X_END_RATIO
_src_tr_y = ROI_Y_RATIO
_src_bl_x = max(0.0, ROI_X_START_RATIO - ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
_src_bl_y = ROI_Y_END_RATIO
_src_br_x = min(1.0, ROI_X_END_RATIO + ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
_src_br_y = ROI_Y_END_RATIO

UNIVERSAL_SRC_RATIOS = [
    [_src_tl_x, _src_tl_y], [_src_tr_x, _src_tr_y],
    [_src_bl_x, _src_bl_y], [_src_br_x, _src_br_y]
]

UNIVERSAL_DST_RATIOS = [
    [0.25, 0.0], [0.75, 0.0],
    [0.25, 1.0], [0.75, 1.0]
]

# --- Visualization Parameters ---
GT_COLOR = (0, 255, 0)
GT_THICKNESS = 2
ALGO_LANE_THICKNESS = 3
DEFAULT_ALGO_COLOR = (255, 255, 255)

# --- Quantitative Evaluation Metrics ---
LANE_METRICS_THRESHOLD_PX = 10
H_SAMPLES = list(range(240, 1080, 50))

# --- Canny Edge Detection Parameters ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- Hough Transform Parameters ---
HOUGH_RHO = 1
HOUGH_THETA_DIVISOR = 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 17
HOUGH_MAX_LINE_GAP = 5
HOUGH_SLOPE_LEFT_MIN = -5.0
HOUGH_SLOPE_LEFT_MAX = -0.4
HOUGH_SLOPE_RIGHT_MIN = 0.4
HOUGH_SLOPE_RIGHT_MAX = 5.0
HOUGH_GAUSSIAN_BLUR_KERNEL = (5, 5)
HOUGH_GAUSSIAN_BLUR_SIGMA_X = 0
HOUGH_CANNY_LOW_THRESHOLD = 25
HOUGH_CANNY_HIGH_THRESHOLD = 95
HOUGH_MAX_HORIZONTAL_SLOPE_DEVIATION = 0.15
HOUGH_MIN_SEGMENTS_FOR_LANE_FIT = 2
HOUGH_MIN_POINTS_FOR_POLYFIT = 5

# --- Advanced Sliding Window Parameters ---
ASW_NWINDOWS = 9
ASW_MARGIN = 80
ASW_MINPIX = 40
ASW_POLY_DEGREE = 2
ASW_S_THRESH_MIN = 170
ASW_S_THRESH_MAX = 255
ASW_SOBEL_KERNEL_SIZE = 5
ASW_SOBEL_THRESH_MIN = 30
ASW_SOBEL_THRESH_MAX = 150
ASW_SRC_RATIOS = UNIVERSAL_SRC_RATIOS
ASW_DST_RATIOS = UNIVERSAL_DST_RATIOS
ASW_MIN_LANE_DIST_WARPED = 30

# --- CarND Pipeline Specific Parameters ---
CARND_RGB_THRESH = (190, 255)
CARND_HLS_THRESH = (190, 255)
CARND_SRC_RATIOS = UNIVERSAL_SRC_RATIOS
CARND_DST_RATIOS = UNIVERSAL_DST_RATIOS