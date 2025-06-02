import logging  # Add logging
import cv2
import numpy as np
import os
from . import utils
from . import lane_comparison_config as CFG

# Configure a simple logger for this module
logger = logging.getLogger(__name__)  # Gets a logger named: src.lane_comparison.visualizer
#turn off logger
logger.disabled = True
# Check if handlers are already attached to prevent duplicate logs
if not logger.handlers:
    handler = logging.StreamHandler()  # Outputs to stderr by default
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')  # Added process ID
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Set to INFO for key messages, can be DEBUG for more
    logger.propagate = True  # Ensure logs propagate if a root logger is configured

def draw_legend(image, algorithms_data, font_scale=0.5, thickness=1, text_color=(255, 255, 255)):
    """Draws a legend on the image with algorithm names and their colors."""
    start_x = 20  # Adjust to position legend on the left
    start_y = 20  # Adjust as needed
    line_height = 20
    box_size = 15

    for i, algo_data in enumerate(algorithms_data):
        if not isinstance(algo_data, dict):
            logger.warning(f"Legend: Skipping invalid algo_data item: {algo_data}")
            continue
        
        algo_name = algo_data.get('name', 'Unknown Algorithm')
        algo_color = algo_data.get('color', (128, 128, 128)) # Default to gray if no color

        # Ensure color is BGR tuple for OpenCV
        if not (isinstance(algo_color, tuple) and len(algo_color) == 3):
            logger.warning(f"Legend: Invalid color format for {algo_name}: {algo_color}. Defaulting to gray.")
            algo_color = (128, 128, 128)

        y_pos = start_y + i * line_height

        # Draw color box
        cv2.rectangle(image, (start_x, y_pos - box_size + 2), (start_x + box_size, y_pos + 2), 
                      algo_color, -1) # Filled rectangle

        # Draw algorithm name
        cv2.putText(image, algo_name, (start_x + box_size + 5, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return image

def create_comparison_image(original_image_cv2, detected_lanes_by_algo, gt_lanes_for_image, image_filename_for_log=""):
    logger.info(f"Visualizer ({image_filename_for_log}): ENTERING create_comparison_image.")
    if original_image_cv2 is None:
        logger.error(f"Visualizer ({image_filename_for_log}): original_image_cv2 is None. Cannot proceed. Returning None.")
        return None  # Caller must handle this

    overlay_image = original_image_cv2.copy()
    logger.debug(f"Visualizer ({image_filename_for_log}): Initial overlay_image shape {overlay_image.shape}, type {overlay_image.dtype}. Copied from original.")

    # Draw Ground Truth Lanes
    if gt_lanes_for_image:
        num_gt_lanes = len(gt_lanes_for_image)
        gt_sample_data = "N/A"
        if num_gt_lanes > 0 and gt_lanes_for_image[0] is not None:
            if isinstance(gt_lanes_for_image[0], (list, np.ndarray)) and len(gt_lanes_for_image[0]) > 0:
                if isinstance(gt_lanes_for_image[0][0], (list, np.ndarray)) and len(gt_lanes_for_image[0][0]) >= 2:
                    gt_sample_data = f"First GT lane, first point: {gt_lanes_for_image[0][0]}"
                else:
                    gt_sample_data = f"First GT lane's first point has unexpected format: {type(gt_lanes_for_image[0][0]) if len(gt_lanes_for_image[0]) > 0 else 'empty lane'}"
            else:
                gt_sample_data = f"First GT lane has no points or unexpected format: {type(gt_lanes_for_image[0])}"
        
        logger.info(f"Visualizer ({image_filename_for_log}): Attempting to draw {num_gt_lanes} GT lanes. {gt_sample_data}")
        try:
            overlay_image = utils.draw_single_lane_set(
                overlay_image, 
                gt_lanes_for_image, 
                CFG.GT_COLOR, 
                CFG.GT_THICKNESS
            )
            logger.debug(f"Visualizer ({image_filename_for_log}): GT lanes drawing attempted.")
        except Exception as e:
            logger.error(f"Visualizer ({image_filename_for_log}): ERROR drawing GT lanes: {e}", exc_info=True)
    else:
        logger.info(f"Visualizer ({image_filename_for_log}): No GT lanes to draw.")

    # Draw Detected Lanes for each algorithm
    logger.debug(f"Visualizer ({image_filename_for_log}): Iterating detected_lanes_by_algo (type: {type(detected_lanes_by_algo)}). Content: {detected_lanes_by_algo}")
    for algo_result in detected_lanes_by_algo: # Iterate through the list of results
        if algo_result and isinstance(algo_result, dict):
            algo_display_name = algo_result.get('name')
            algo_lanes = algo_result.get('lanes')
            algo_color = algo_result.get('color', CFG.DEFAULT_ALGO_COLOR) # Use default if color not in result

            if algo_display_name:
                logger.info(f"Visualizer ({image_filename_for_log}): Processing detected lanes for {algo_display_name}.")
                if algo_lanes and isinstance(algo_lanes, list):
                    num_algo_lanes = len(algo_lanes)
                    algo_sample_data = "N/A"
                    if num_algo_lanes > 0 and algo_lanes[0] is not None:
                        if isinstance(algo_lanes[0], (list, np.ndarray)) and len(algo_lanes[0]) > 0:
                            if isinstance(algo_lanes[0][0], (list, np.ndarray)) and len(algo_lanes[0][0]) >= 2:
                                algo_sample_data = f"First {algo_display_name} lane, first point: {algo_lanes[0][0]}"
                            else:
                                algo_sample_data = f"First {algo_display_name} lane's first point has unexpected format: {type(algo_lanes[0][0]) if len(algo_lanes[0]) > 0 else 'empty lane'}"
                        else:
                             algo_sample_data = f"First {algo_display_name} lane has no points or unexpected format: {type(algo_lanes[0])}"
                    
                    logger.info(f"Visualizer ({image_filename_for_log}): Attempting to draw {num_algo_lanes} lanes for {algo_display_name} with color {algo_color}. {algo_sample_data}")
                    try:
                        overlay_image = utils.draw_single_lane_set(
                            overlay_image,
                            algo_lanes,
                            algo_color, # Use color from algo_result
                            CFG.ALGO_LANE_THICKNESS
                        )
                        logger.debug(f"Visualizer ({image_filename_for_log}): {algo_display_name} lanes drawing attempted.")
                    except Exception as e:
                        logger.error(f"Visualizer ({image_filename_for_log}): ERROR drawing {algo_display_name} lanes: {e}", exc_info=True)
                else:
                    logger.info(f"Visualizer ({image_filename_for_log}): No 'lanes' data or 'lanes' is not a list for {algo_display_name} (lanes data: {repr(algo_lanes)[:100]}, type: {type(algo_lanes)}).")
            else:
                logger.warning(f"Visualizer ({image_filename_for_log}): Found an algo_result without a 'name': {algo_result}")
        else:
            logger.warning(f"Visualizer ({image_filename_for_log}): Invalid algo_result found in detected_lanes_by_algo: {algo_result}")

    # Draw the legend
    try:
        # Prepare data for legend: only active algorithms that provided lanes or are configured to show
        # For simplicity, we'll pass all `detected_lanes_by_algo` as it contains name and color.
        # If an algorithm didn't run or produce lanes, it might still appear if its data is in `detected_lanes_by_algo`.
        # This can be refined if needed to only show algos that actually drew something.
        if detected_lanes_by_algo: # Ensure there's something to draw a legend for
            overlay_image = draw_legend(overlay_image, detected_lanes_by_algo)
            logger.info(f"Visualizer ({image_filename_for_log}): Legend drawing attempted.")
        else:
            logger.info(f"Visualizer ({image_filename_for_log}): No algorithm data provided for legend.")
    except Exception as e:
        logger.error(f"Visualizer ({image_filename_for_log}): ERROR drawing legend: {e}", exc_info=True)

    logger.info(f"Visualizer ({image_filename_for_log}): EXITING create_comparison_image.")
    return overlay_image

def save_image(image, output_dir, filename):
    if image is None:
        logger.error(f"Error: Attempted to save a None image as {filename} in {output_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}", exc_info=True)
        if hasattr(image, 'shape') and hasattr(image, 'dtype'):
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
        else:
            logger.debug("Image shape or dtype not available.")
