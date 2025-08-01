#print("DEBUG: run_comparison.py script started") # DEBUG PRINT 0

import cv2
import os
import json
import numpy as np
import importlib
import types
from types import SimpleNamespace
import copy
import multiprocessing
import logging
import time
import random # Added for sampling images for optimization
from . import utils as lane_utils # Changed alias for clarity
from . import lane_comparison_config as cfg # For main orchestrator
from .utils import (
    load_ground_truth_entry, 
    convert_culane_gt_to_points, 
    calculate_lane_metrics, 
    parse_command_line_arguments, 
    define_roi_vertices_from_config,
    save_and_print_metrics_summary
)
from .visualizer import create_comparison_image, save_image
from .parameter_sweeper import (
    generate_hough_transform_parameter_sets,
    generate_advanced_sliding_window_parameter_sets,
    generate_carnd_pipeline_parameter_sets, 
    expand_algorithms_for_sweeps,
    DEFAULT_VARIANT_COLORS,
    optimize_parameters_for_algorithm # Import the optimizer function
)

# Try to import dashboard integration
try:
    from ..gui.dashboard_integration import initialize_dashboard, get_dashboard, setup_dashboard_logging
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

#print("DEBUG: Imports in run_comparison.py completed") # DEBUG PRINT 1

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format) # Changed level to INFO
logger = logging.getLogger(__name__)

# --- Top-level worker function for multiprocessing ---
# This function processes all images for a single algorithm configuration.
def _mp_worker_process_single_algo_all_images(
    algo_config_worker, 
    image_filenames_worker,
    global_cfg_dict_serializable_worker # Contains picklable config values
):
    # Initialize logger for this specific worker process
    # Using a distinct logger name can help differentiate logs if needed.
    worker_logger = logging.getLogger(f"{__name__}.mp_worker.{algo_config_worker['display_name']}")
    
    algo_name_worker = algo_config_worker['display_name']
    local_exec_time_sum = 0.0
    local_frames_processed = 0
    
    metrics_summary_for_this_algo = []
    detected_lanes_for_this_algo = {} # image_filename -> {"lanes": ..., "color": ...}

    # Reconstruct a SimpleNamespace for configuration within the worker
    # This ensures the worker uses a consistent config view derived from picklable data.
    current_run_cfg_dict_worker_base = global_cfg_dict_serializable_worker.copy()
    
    # Apply algorithm-specific parameter overrides
    if "param_overrides" in algo_config_worker:
        overrides = algo_config_worker["param_overrides"]
        if "LOWER_WHITE_L_COMPONENT" in overrides:
            if 'LOWER_WHITE_HLS' in current_run_cfg_dict_worker_base and \
               isinstance(current_run_cfg_dict_worker_base['LOWER_WHITE_HLS'], list) and \
               len(current_run_cfg_dict_worker_base['LOWER_WHITE_HLS']) == 3:
                
                modified_lower_white_hls = list(current_run_cfg_dict_worker_base['LOWER_WHITE_HLS'])
                modified_lower_white_hls[1] = overrides["LOWER_WHITE_L_COMPONENT"]
                current_run_cfg_dict_worker_base['LOWER_WHITE_HLS'] = modified_lower_white_hls
            else:
                worker_logger.warning(f"LOWER_WHITE_HLS not found or invalid format in base config for {algo_name_worker}. Cannot apply L_COMPONENT override.")
            # Apply other overrides, excluding the one just handled
            temp_overrides = {k: v for k, v in overrides.items() if k != "LOWER_WHITE_L_COMPONENT"}
            current_run_cfg_dict_worker_base.update(temp_overrides)
        else:
            current_run_cfg_dict_worker_base.update(overrides)
            
    current_run_cfg_worker = SimpleNamespace(**current_run_cfg_dict_worker_base)

    # Dynamically import the algorithm module and function
    module_path = f".algorithms.{algo_config_worker['module_name']}"
    try:
        # Assuming this script (run_comparison.py) is part of the 'src.lane_comparison' package
        algo_module = importlib.import_module(module_path, package='src.lane_comparison')
        detect_function = getattr(algo_module, algo_config_worker['function_name'])
    except Exception as e:
        #worker_logger.error(f"Error loading algorithm {algo_name_worker}: {e}")
        # Return empty results if algorithm loading fails
        return algo_name_worker, [], {}, 0, 0

    for idx, image_filename_worker in enumerate(image_filenames_worker):
        # Print progress from the worker process
        #print(f"[{algo_name_worker}] Processing image {idx + 1}/{len(image_filenames_worker)}: {image_filename_worker}")

        image_path_worker = os.path.join(current_run_cfg_worker.IMAGE_SOURCE_DIR, image_filename_worker)
        base_image_array_worker = cv2.imread(image_path_worker)

        if base_image_array_worker is None:
            worker_logger.error(f"Could not load image: {image_path_worker}, skipping for {algo_name_worker}.")
            continue

        # Image resizing for performance
        image_to_process_for_algo = base_image_array_worker
        original_shape_for_scaling_back = base_image_array_worker.shape
        was_resized = False
        resize_factor = getattr(current_run_cfg_worker, 'RESIZE_PROCESSING_FACTOR', 1.0)
        
        # Determine if resizing should be applied for the current algorithm
        # For example, LaneNet might be sensitive to input size changes.
        apply_resize_for_this_algo = True # Default to true
        # Check if the algorithm is LaneNet (assuming 'lanenet' is in its module_name)
        if 'lanenet' in algo_config_worker.get('module_name', '').lower():
            apply_resize_for_this_algo = False
            #worker_logger.info(f"[{algo_name_worker}] Skipping image resizing as it might be LaneNet.")

        if apply_resize_for_this_algo and 0.0 < resize_factor < 1.0:
            worker_logger.debug(f"[{algo_name_worker}] Resizing image {image_filename_worker} by factor {resize_factor}")
            original_height, original_width = image_to_process_for_algo.shape[:2]
            new_width = int(original_width * resize_factor)
            new_height = int(original_height * resize_factor)
            if new_width > 0 and new_height > 0:
                image_to_process_for_algo = cv2.resize(image_to_process_for_algo, (new_width, new_height), interpolation=cv2.INTER_AREA)
                was_resized = True
            else:
                worker_logger.warning(f"[{algo_name_worker}] Invalid new dimensions after applying resize_factor {resize_factor} to image {image_filename_worker}. Skipping resize.")
        
        current_image_h_samples_for_metrics_worker = getattr(current_run_cfg_worker, 'H_SAMPLES', [])
        gt_lanes_points_worker = []
        gt_entry_worker = load_ground_truth_entry(current_run_cfg_worker.JSON_GT_PATH, image_filename_worker)

        if gt_entry_worker:
            gt_lanes_culane_worker = gt_entry_worker.get("lanes", [])
            h_samples_from_gt_entry_worker = gt_entry_worker.get("h_samples", [])
            if h_samples_from_gt_entry_worker:
                current_image_h_samples_for_metrics_worker = h_samples_from_gt_entry_worker
            gt_lanes_points_worker = convert_culane_gt_to_points(gt_lanes_culane_worker, current_image_h_samples_for_metrics_worker)
        
        # Define ROI based on the current image (possibly resized) and configuration
        roi_vertices_worker = define_roi_vertices_from_config(image_to_process_for_algo.shape, current_run_cfg_worker)
        
        detected_lanes_worker = []
        exec_time_worker = 0
        start_time_worker = time.time()
        try:
            if algo_config_worker['module_name'] == "advanced_sliding_window":
                detected_lanes_worker = detect_function(image_to_process_for_algo, roi_vertices_worker, {}, current_run_cfg_worker)
            elif algo_config_worker['module_name'] == "carnd_pipeline_algorithm":
                if hasattr(current_run_cfg_worker, 'CARND_SRC_RATIOS') and hasattr(current_run_cfg_worker, 'CARND_DST_RATIOS'):
                    param_overrides_worker = algo_config_worker.get("param_overrides", {})
                    detected_lanes_worker = detect_function(
                        image_to_process_for_algo,
                        roi_vertices_worker, 
                        current_run_cfg_worker.CARND_SRC_RATIOS,
                        current_run_cfg_worker.CARND_DST_RATIOS,
                        param_overrides_worker
                    )
                else:
                    worker_logger.error(f"[{algo_name_worker}] Perspective ratios not found for CarND on {image_filename_worker}.")
                    detected_lanes_worker = [] 
            else: # Default call signature for other algorithms
                detected_lanes_worker = detect_function(image_to_process_for_algo, roi_vertices_worker, current_run_cfg_worker)
        except Exception as e:
            worker_logger.error(f"Error running {algo_name_worker} on {image_filename_worker}: {e}")
            pass # Continue processing other images for this algorithm
        end_time_worker = time.time()
        exec_time_worker = end_time_worker - start_time_worker
        local_exec_time_sum += exec_time_worker
        local_frames_processed += 1

        # Scale detected lanes back to original image size if resizing occurred
        if apply_resize_for_this_algo and was_resized and detected_lanes_worker:
            scaled_back_lanes = []
            resized_shape = image_to_process_for_algo.shape # Shape of the image used for detection
            scale_x = original_shape_for_scaling_back[1] / resized_shape[1]
            scale_y = original_shape_for_scaling_back[0] / resized_shape[0]
            
            for lane in detected_lanes_worker:
                if not lane: # Handle empty lanes
                    scaled_back_lanes.append([])
                    continue
                scaled_lane = []
                for point in lane:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        scaled_x_val = int(point[0] * scale_x) # Renamed to avoid conflict if point[0] was also 'scaled_x'
                        scaled_y_val = int(point[1] * scale_y) # Renamed to avoid conflict
                        scaled_lane.append([scaled_x_val, scaled_y_val])
                    else:
                        # Non-point data or unexpected format, append as is or log warning
                        scaled_lane.append(point) 
                        worker_logger.debug(f"[{algo_name_worker}] Non-point data found in lane for scaling: {point}")
                scaled_back_lanes.append(scaled_lane)
            detected_lanes_worker = scaled_back_lanes
            worker_logger.debug(f"[{algo_name_worker}] Scaled back detected lanes for {image_filename_worker}") # Corrected variable name

        metrics_worker = {}
        # Use SKIP_IMAGES_WITHOUT_GT from the effective config for this run
        skip_img_if_no_gt_worker = getattr(current_run_cfg_worker, 'SKIP_IMAGES_WITHOUT_GT', True)
        if gt_lanes_points_worker or not skip_img_if_no_gt_worker:
            metrics_worker = calculate_lane_metrics(
                gt_lanes_points_worker,
                detected_lanes_worker,
                current_image_h_samples_for_metrics_worker,
                current_run_cfg_worker.LANE_METRICS_THRESHOLD_PX # From effective config
            )
        
        should_store_metrics = False
        if metrics_worker:
            should_store_metrics = True
        elif gt_lanes_points_worker: 
            should_store_metrics = True
        elif not skip_img_if_no_gt_worker: # If we are not skipping images without GT
            should_store_metrics = True
        
        if should_store_metrics:
            metrics_summary_for_this_algo.append({
                "image": image_filename_worker,
                "exec_time": exec_time_worker,
                **metrics_worker
            })
        
        detected_lanes_for_this_algo[image_filename_worker] = {
            "lanes": detected_lanes_worker,
            "color": algo_config_worker['color']
        }

    # Safe FPS calculation for worker log
    fps_for_log = 0
    if local_frames_processed > 0 and local_exec_time_sum > 0:
        fps_for_log = local_frames_processed / local_exec_time_sum
    worker_logger.info(f"Algorithm '{algo_name_worker}' completed processing {local_frames_processed} images in {local_exec_time_sum:.2f}s (FPS: {fps_for_log:.2f})")
    return (
        algo_name_worker, 
        metrics_summary_for_this_algo, 
        detected_lanes_for_this_algo, 
        local_exec_time_sum, 
        local_frames_processed
    )
# --- End of top-level worker function ---


def _process_images_and_write_video(
    image_filenames_to_process,
    cfg_obj, # The main config object (e.g., imported 'cfg' module)
    final_loaded_algorithms_configs,
    dashboard=None  # Dashboard integration
):
    overall_start_time = time.time()

    video_writer = None
    first_image_for_video = True
    
    batch_metrics_summary = {algo['display_name']: [] for algo in final_loaded_algorithms_configs}
    all_detected_lanes_by_algo_then_image = {algo['display_name']: {} for algo in final_loaded_algorithms_configs}
    algo_exec_times = {algo['display_name']: 0.0 for algo in final_loaded_algorithms_configs}
    algo_frame_counts = {algo['display_name']: 0 for algo in final_loaded_algorithms_configs}
    
    images_processed_count_for_output = 0 # For output generation phase

    # Create a picklable version of the configuration for worker processes
    ALLOWED_CONFIG_TYPES = (str, int, float, bool, list, dict, tuple, type(None))
    global_cfg_serializable = {
        k: v for k, v in cfg_obj.__dict__.items()
        if not k.startswith('__') and isinstance(v, ALLOWED_CONFIG_TYPES)
    }
    
    # --- Phase 1: Algorithm Processing (Multiprocessing) ---
    #logger.info(f"Starting algorithm processing for {len(image_filenames_to_process)} images "
    #            f"with {len(final_loaded_algorithms_configs)} algorithms using multiprocessing.")
    
    if dashboard:
        dashboard.update_phase("Algorithm Processing")

    pool_args = []
    for algo_config in final_loaded_algorithms_configs:
        pool_args.append(
            (
                algo_config,
                image_filenames_to_process, # Each algo gets all images
                global_cfg_serializable # Pass the picklable config dict
            )
        )

    # Determine number of processes for the pool
    # Cap at number of CPUs or number of algorithms, whichever is smaller.
    # Consider if LaneNet (if active) has specific process limitations.
    # For now, assume all algos can run in parallel up to CPU limits.
    # The original LaneNet check was for an inner pool, not this outer algo-level pool.
    num_algo_processes = len(final_loaded_algorithms_configs)
    if hasattr(cfg_obj, 'MAX_PARALLEL_ALGORITHMS') and cfg_obj.MAX_PARALLEL_ALGORITHMS > 0 :
        num_algo_processes = min(num_algo_processes, cfg_obj.MAX_PARALLEL_ALGORITHMS)
    else: # Default to number of CPU cores if not specified, or if too many algos
        num_algo_processes = min(num_algo_processes, multiprocessing.cpu_count())
    
    num_algo_processes = max(1, num_algo_processes) # Ensure at least one process

    logger.info(f"Using a pool of {num_algo_processes} processes for running algorithms.")

    results_from_algo_pool = []
    if pool_args:
        # Corrected: Use try-finally to ensure pool closure, or use context manager if available for Pool
        # For starmap, it blocks until all results are ready.
        with multiprocessing.Pool(processes=num_algo_processes) as algo_pool:
            results_from_algo_pool = algo_pool.starmap(_mp_worker_process_single_algo_all_images, pool_args)

    # Collect results from all algorithm processes
    for result_tuple in results_from_algo_pool:
        if result_tuple:
            (algo_name_res, metrics_list_res, lanes_dict_res, 
             exec_time_res, frames_res) = result_tuple
            
            # Ensure the list for metrics exists before extending
            if algo_name_res not in batch_metrics_summary:
                 batch_metrics_summary[algo_name_res] = [] # Should already exist from initialization
            batch_metrics_summary[algo_name_res].extend(metrics_list_res)
            
            all_detected_lanes_by_algo_then_image[algo_name_res] = lanes_dict_res
            algo_exec_times[algo_name_res] = exec_time_res
            algo_frame_counts[algo_name_res] = frames_res
            
            # Update dashboard with algorithm completion
            if dashboard:
                dashboard.update_algorithm_progress(algo_name_res, 100.0)
                dashboard.add_log(f"Algorithm '{algo_name_res}' completed processing {frames_res} images")
        else:
            logger.error("Received an empty result from an algorithm process. This should not happen if worker returns tuple.")


    logger.info("All algorithm processing has completed.")

    # --- Phase 2: Image/Video Generation (Sequential) ---
    logger.info("Starting image/video generation phase...")
    if dashboard:
        dashboard.update_phase("Image/Video Generation")
    
    # Check if any images were processed in Phase 1
    # A simple check: if the sum of frames processed by algos is zero, but we had input images.
    total_frames_by_all_algos = sum(algo_frame_counts.values())
    if total_frames_by_all_algos == 0 and image_filenames_to_process:
        logger.warning("No images seem to have been processed by algorithms in Phase 1, though input images were provided.")
    elif not image_filenames_to_process:
        logger.info("No images were in the input list to process.")
        return batch_metrics_summary, 0 # No images to process, return early


    for i, image_filename in enumerate(image_filenames_to_process):
        # This print is for the main process, indicating output generation progress
        #print(f"[MainProcess] Generating output for image {images_processed_count_for_output + 1}/{len(image_filenames_to_process)}: {image_filename}...")
        
        # Update dashboard with current image progress
        if dashboard:
            dashboard.update_image_progress(image_filename, images_processed_count_for_output + 1)

        image_path = os.path.join(cfg_obj.IMAGE_SOURCE_DIR, image_filename)
        base_image = cv2.imread(image_path)
        if base_image is None:
            logger.error(f"Could not load image for output generation: {image_path}, skipping.")
            continue

        # Ground truth loading for display
        current_image_h_samples_for_gt_display = getattr(cfg_obj, 'H_SAMPLES', [])
        gt_lanes_points_for_display = []
        gt_entry = load_ground_truth_entry(cfg_obj.JSON_GT_PATH, image_filename)
        if gt_entry:
            gt_lanes_culane = gt_entry.get("lanes", [])
            h_samples_from_gt = gt_entry.get("h_samples", [])
            if h_samples_from_gt:
                current_image_h_samples_for_gt_display = h_samples_from_gt
            gt_lanes_points_for_display = convert_culane_gt_to_points(gt_lanes_culane, current_image_h_samples_for_gt_display)
        elif cfg_obj.SKIP_IMAGES_WITHOUT_GT:
            logger.debug(f"Image {image_filename} has no GT and SKIP_IMAGES_WITHOUT_GT is True. GT lanes will be empty for visualization.")
            pass # Proceed with empty GT lanes for visualization


        if first_image_for_video: # Initialize video writer with the first valid image
            height, width, _ = base_image.shape
            video_output_path = os.path.join(cfg_obj.OUTPUT_VIS_DIR, cfg_obj.OUTPUT_VIDEO_FILENAME)
            if not os.path.exists(cfg_obj.OUTPUT_VIS_DIR):
                try:
                    os.makedirs(cfg_obj.OUTPUT_VIS_DIR, exist_ok=True)
                    logger.info(f"Successfully created output directory: {cfg_obj.OUTPUT_VIS_DIR}")
                except Exception as e:
                    logger.error(f"Failed to create output directory {cfg_obj.OUTPUT_VIS_DIR}: {e}")
            
            fourcc_str = getattr(cfg_obj, 'VIDEO_CODEC', 'mp4v') # Default to mp4v
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            video_writer = cv2.VideoWriter(video_output_path, fourcc, float(cfg_obj.VIDEO_FPS), (width, height))
            
            if not video_writer.isOpened():
                logger.error(f"VideoWriter failed to open for '{video_output_path}'. Check codec ('{fourcc_str}'), permissions, and path.")
                video_writer = None 
            else:
                logger.info(f"VideoWriter opened successfully for '{video_output_path}'.")
            first_image_for_video = False

        # Collect detected lanes for this specific image from all algorithms
        detected_lanes_data_for_image = []
        for algo_conf in final_loaded_algorithms_configs: # Iterate in defined order for consistent overlay
            algo_name_collect = algo_conf['display_name']
            # Check if this algo processed this image and has data for it
            if algo_name_collect in all_detected_lanes_by_algo_then_image and \
               image_filename in all_detected_lanes_by_algo_then_image[algo_name_collect]:
                
                result_for_algo_image = all_detected_lanes_by_algo_then_image[algo_name_collect][image_filename]
                detected_lanes_data_for_image.append({
                    "name": algo_name_collect,
                    "lanes": result_for_algo_image["lanes"],
                    "color": result_for_algo_image["color"]
                })
            # else: This image might not have been processed by this algo, or no lanes found.
            # create_comparison_image should handle cases where "lanes" might be empty or algo data missing.

        comparison_frame = create_comparison_image(
            original_image_cv2=base_image, 
            gt_lanes_for_image=gt_lanes_points_for_display, 
            detected_lanes_by_algo=detected_lanes_data_for_image,
            image_filename_for_log=image_filename
        )
            
        if comparison_frame is not None and comparison_frame.size > 0:
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(comparison_frame)
            
            output_image_filename = f"comp_{image_filename}"
            output_image_path = save_image(comparison_frame, cfg_obj.OUTPUT_VIS_DIR, output_image_filename)
            
            # Update dashboard with preview image
            if dashboard and output_image_path:
                dashboard.update_preview_image(output_image_path)
                
        else:
            logger.warning(f"Comparison frame is None or empty for image {image_filename}. Skipping write.")
        
        images_processed_count_for_output += 1

    # --- Finalization ---
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
        video_path = os.path.join(cfg_obj.OUTPUT_VIS_DIR, cfg_obj.OUTPUT_VIDEO_FILENAME)
        logger.info(f"Video successfully saved: {video_path}")
        
        # Update dashboard with the created video
        if dashboard:
            dashboard.update_preview_video(video_path)
            
    elif images_processed_count_for_output > 0 and (video_writer is None or not video_writer.isOpened()) and not first_image_for_video:
        # This case means we processed images for output, but video writer had an issue (wasn't opened or closed prematurely)
        logger.warning("Images processed for output, but video writer was not properly initialized or failed. No video saved.")
    elif images_processed_count_for_output == 0 and image_filenames_to_process: # Had images to process, but none made it to output generation
        logger.warning("No images were successfully processed in the output generation phase, though input images were provided.")
    # If images_processed_count_for_output is 0 and image_filenames_to_process is also empty, no warning needed.


    overall_end_time = time.time()
    total_script_execution_time = overall_end_time - overall_start_time

    num_images_for_overall_fps = images_processed_count_for_output

    if num_images_for_overall_fps > 0:
        logger.info("\\n--- Performance Metrics ---")
        overall_fps_val = num_images_for_overall_fps / total_script_execution_time if total_script_execution_time > 0 else 0
        logger.info(f"Overall pipeline FPS (total wall time): {overall_fps_val:.2f} "
                    f"(across {num_images_for_overall_fps} images for output, total time: {total_script_execution_time:.2f}s)")

        for algo_name_fps, total_time_fps in algo_exec_times.items():
            frame_count_fps = algo_frame_counts[algo_name_fps]
            if frame_count_fps > 0 and total_time_fps > 0:
                fps = frame_count_fps / total_time_fps
                logger.info(f"Algorithm '{algo_name_fps}' average processing FPS: {fps:.2f} "
                            f"(processed {frame_count_fps} images in {total_time_fps:.2f}s)")
            elif frame_count_fps > 0 : # Processed frames but time was negligible or zero
                 logger.info(f"Algorithm '{algo_name_fps}' processed {frame_count_fps} images but had zero or negligible execution time.")
            else: # Did not process any frames
                logger.info(f"Algorithm '{algo_name_fps}' did not run or process any frames.")
        logger.info("-------------------------")
    elif image_filenames_to_process : # Had input images, but none were processed to calculate FPS.
        logger.warning("No images were processed successfully to calculate performance metrics.")
    
    return batch_metrics_summary, images_processed_count_for_output


def _evaluation_callback_for_optimizer(
    current_params_for_eval,    # Parameters being tested by the optimizer
    base_algo_config,           # Original algorithm config (e.g., cfg.ALGORITHMS["carnd_algorithm"])
    algo_module_name,           # E.g., "carnd_algorithm_modular"
    algo_fn_name,               # E.g., "detect_lanes_carnd_modular"
    image_path_subset,          # List of image paths for this evaluation run
    logger_callback,            # Logger instance from the optimizer
    cfg_callback,               # Config module (cfg) from the optimizer
    ground_truth_data_dict,     # The main ground_truth_data dictionary, keyed by image filename
    h_samples_from_cfg          # Fallback cfg.H_SAMPLES from the main config
):
    """
    Callback function for the optimizer to evaluate a given set of parameters.
    It runs the detection algorithm on a subset of images and returns an F1 score.
    """
    logger_callback.debug(f"Optimizer eval callback: Algo: {algo_module_name}, Params: {current_params_for_eval}, Images: {len(image_path_subset)}")

    try:
        # Assuming src.lane_detection_algorithms is the correct path relative to where the script is run or Python's path
        algo_module = importlib.import_module(f"src.lane_detection_algorithms.{algo_module_name}")
        detection_function = getattr(algo_module, algo_fn_name)
    except Exception as e:
        logger_callback.error(f"Optimizer eval: Failed to load algorithm {algo_module_name}.{algo_fn_name}: {e}")
        return 0.0  # Cannot evaluate

    f1_scores = []
    num_images_processed_for_eval = 0
    eval_run_params = current_params_for_eval # These are the specific parameters to test for the algo

    for image_path in image_path_subset:
        if num_images_processed_for_eval >= cfg_callback.MAX_IMAGES_FOR_OPTIMIZER_EVAL:
            logger_callback.info(f"Optimizer eval: Reached MAX_IMAGES_FOR_OPTIMIZER_EVAL ({cfg_callback.MAX_IMAGES_FOR_OPTIMIZER_EVAL}). Stopping eval for this iteration.")
            break
        
        image_filename = os.path.basename(image_path)
        try:
            img_for_eval = cv2.imread(image_path)
            if img_for_eval is None:
                logger_callback.warning(f"Optimizer eval: Failed to load image {image_path}. Skipping.")
                continue

            # Prepare the configuration for the detection function
            # Start with a copy of the algorithm's base parameters (if any defined in cfg.ALGORITHMS)
            temp_algo_specific_config = base_algo_config.get("parameters", {}).copy()
            # Update with the current parameters being tuned by the optimizer
            temp_algo_specific_config.update(eval_run_params)

            # Construct the full config dict that some modular functions might expect
            current_full_config_for_eval = {
                "IMAGE_WIDTH": img_for_eval.shape[1],
                "IMAGE_HEIGHT": img_for_eval.shape[0],
                "ROI_VERTICES_RATIO": cfg_callback.ROI_VERTICES_RATIO, # from main cfg
                # Add other general config items if needed by the algo, e.g., H_SAMPLES if not passed elsewhere
            }
            # Merge the tuned algorithm-specific parameters into this full config
            current_full_config_for_eval.update(temp_algo_specific_config)


            processed_result = None
            if "carnd_algorithm" in algo_module_name or "hough_transform_algorithm" in algo_module_name:
                 processed_result = detection_function(
                    img_for_eval,
                    config=current_full_config_for_eval, # Pass the combined and tuned config
                    logger_instance=logger_callback,
                    debug_mode=False # No debug visuals during optimization trials
                )
            # Add elif for other optimizable algorithms if their call signature differs
            else:
                logger_callback.warning(f"Optimizer eval: Algorithm {algo_module_name} is not explicitly supported by this evaluation callback structure for parameter passing. Skipping detection.")
                processed_result = None


            if processed_result and 'detected_lanes_pixels' in processed_result:
                gt_entry = ground_truth_data_dict.get(image_filename)
                if not gt_entry:
                    logger_callback.warning(f"Optimizer eval: GT data not found for {image_filename} in ground_truth_data_dict. Skipping metrics for this image.")
                    f1_scores.append(0.0) # Or continue, depending on desired strictness
                    continue

                gt_lanes_culane_format = gt_entry.get("lanes")
                h_samples_for_this_gt = gt_entry.get("h_samples")

                if gt_lanes_culane_format is None or h_samples_for_this_gt is None:
                    logger_callback.warning(f"Optimizer eval: Incomplete GT data for {image_filename} (lanes or h_samples missing). Skipping metrics.")
                    f1_scores.append(0.0)
                    continue
                
                if processed_result['detected_lanes_pixels'] is None:
                    logger_callback.debug(f"Optimizer eval: No lanes detected by {algo_module_name} for {image_filename}. F1 will be 0.")
                    f1_scores.append(0.0)
                else:
                    # Convert GT from CULane list-of-x-coords to list-of-points format
                    # Uses convert_culane_gt_to_points imported from .utils
                    gt_lanes_points_for_metric = lane_utils.convert_culane_gt_to_points(gt_lanes_culane_format, h_samples_for_this_gt)
                    
                    # Uses calculate_lane_metrics imported from .utils
                    metrics = lane_utils.calculate_lane_metrics(
                        gt_lanes_points=gt_lanes_points_for_metric,
                        detected_lanes_polylines=processed_result['detected_lanes_pixels'],
                        h_samples=h_samples_for_this_gt, # Use h_samples specific to this GT entry
                        threshold_px=cfg_callback.LANE_METRICS_THRESHOLD_PX
                    )
                    if metrics and 'f1_score' in metrics:
                        f1_scores.append(metrics['f1_score'])
                    else:
                        logger_callback.warning(f"Optimizer eval: Metrics calculation failed or F1 score missing for {image_filename}.")
                        f1_scores.append(0.0)
            else:
                logger_callback.debug(f"Optimizer eval: No 'detected_lanes_pixels' in result for {image_filename} or processed_result is None.")
                f1_scores.append(0.0) # No detection or bad result format

            num_images_processed_for_eval += 1

        except Exception as e:
            logger_callback.error(f"Optimizer eval: Error processing {image_path} with {algo_module_name}: {e}", exc_info=True)
            f1_scores.append(0.0) # Penalize errors

    if not f1_scores: # Handles case where image_path_subset was empty or all images failed before scoring
        logger_callback.warning("Optimizer eval: No F1 scores were calculated in this evaluation run (e.g., no images, all failed).")
        return 0.0
    
    average_f1 = sum(f1_scores) / len(f1_scores)
    logger_callback.info(f"Optimizer eval callback: Algo: {algo_module_name}, Params: {current_params_for_eval}, Avg F1: {average_f1:.4f} over {len(f1_scores)} images ({num_images_processed_for_eval} processed for eval).")
    return average_f1


def main_comparison_orchestrator():
    # Initialize dashboard if available
    dashboard = None
    if DASHBOARD_AVAILABLE:
        try:
            dashboard = initialize_dashboard("http://localhost:5000", enabled=True)
            setup_dashboard_logging(logger)
            logger.info("Dashboard integration initialized")
        except Exception as e:
            logger.warning(f"Dashboard initialization failed: {e}")
            dashboard = None
    
    args = parse_command_line_arguments(cfg)

    # Override algorithm selection from command line/dashboard if provided
    if args.algorithm:
        logger.info(f"Overriding algorithm selection from command line/dashboard: {args.algorithm}")
        selected_keys = []
        for selected_name in args.algorithm:
            found_key = None
            # Match against the config dict key or module_name suffix
            if selected_name in cfg.ALGORITHMS:
                found_key = selected_name
            else:
                # Match against module_name or display_name
                for key, config in cfg.ALGORITHMS.items():
                    module_name = config.get('module_name')
                    if module_name == selected_name or config.get('display_name') == selected_name or key == selected_name:
                        found_key = key
                        break
            if found_key:
                selected_keys.append(found_key)
            else:
                logger.warning(f"Dashboard requested algorithm '{selected_name}', but it was not found in cfg.ALGORITHMS.")
        cfg.ALGORITHMS_TO_RUN_KEYS = selected_keys
        logger.info(f"Updated algorithms to run: {cfg.ALGORITHMS_TO_RUN_KEYS}")

    if args.dashboard:
        logger.info("Dashboard mode enabled")
        if dashboard:
            dashboard.add_log("Dashboard mode enabled", "info")

    # --- Configuration & Setup ---
    PERFORM_OPTIMIZATION = getattr(cfg, "PERFORM_OPTIMIZATION_PHASE", False)

    ground_truth_data = {}
    if os.path.exists(cfg.JSON_GT_PATH):
        try:
            with open(cfg.JSON_GT_PATH, 'r') as f:
                logger.info(f"Attempting to load ground truth data from: {cfg.JSON_GT_PATH}")
                if dashboard:
                    dashboard.update_phase("Loading ground truth data...")
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        gt_entry = json.loads(line)
                        raw_file_path = gt_entry.get("raw_file")
                        if raw_file_path:
                            image_filename = os.path.basename(raw_file_path)
                            ground_truth_data[image_filename] = {
                                "lanes": gt_entry.get("lanes"),
                                "h_samples": gt_entry.get("h_samples")
                            }
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON line from {cfg.JSON_GT_PATH}")
        except Exception as e:
            logger.error(f"Failed to read/process ground truth file {cfg.JSON_GT_PATH}: {e}")
    else:
        logger.warning(f"Ground truth file not found: {cfg.JSON_GT_PATH}.")

    # --- Image List Preparation ---
    start_index = args.start_index
    num_to_process_from_arg = args.num_images

    if not os.path.exists(cfg.OUTPUT_VIS_DIR):
        os.makedirs(cfg.OUTPUT_VIS_DIR, exist_ok=True)

    try:
        available_image_files = sorted([f for f in os.listdir(cfg.IMAGE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        logger.info(f"Found {len(available_image_files)} image files.")
    except FileNotFoundError:
        logger.error(f"Image directory not found: {cfg.IMAGE_SOURCE_DIR}"); return
    if not available_image_files:
        logger.warning(f"No images found in {cfg.IMAGE_SOURCE_DIR}."); return

    if start_index < 0: start_index = 0
    if start_index >= len(available_image_files):
        logger.error(f"start_index ({start_index}) out of bounds."); return

    images_to_consider_all = available_image_files[start_index:]
    num_actually_to_process = len(images_to_consider_all)
    if num_to_process_from_arg is not None and num_to_process_from_arg > 0:
        num_actually_to_process = min(len(images_to_consider_all), num_to_process_from_arg)
    
    image_list_for_main_run = images_to_consider_all[:num_actually_to_process]
    logger.info(f"Main run will process {len(image_list_for_main_run)} images.")

    if dashboard:
        active_algo_names = [cfg.ALGORITHMS[key]['display_name'] for key in cfg.ALGORITHMS_TO_RUN_KEYS if key in cfg.ALGORITHMS]
        dashboard.start_processing(len(image_list_for_main_run), active_algo_names)

    # --- Parameter Optimization Phase ---
    optimized_algo_params = {}
    if PERFORM_OPTIMIZATION:
        logger.info("--- Starting Parameter Optimization Phase ---")
        if dashboard: dashboard.update_phase("Parameter Optimization Phase")
        
        num_images_for_opt = min(len(available_image_files), cfg.NUM_IMAGES_FOR_OPTIMIZATION_PHASE)
        if num_images_for_opt > 0 and available_image_files:
            image_paths_for_opt = random.sample([os.path.join(cfg.IMAGE_SOURCE_DIR, f) for f in available_image_files], num_images_for_opt)
            logger.info(f"Selected {len(image_paths_for_opt)} images for optimization.")
        else:
            logger.warning("Not enough images for optimization. Skipping phase.")
            PERFORM_OPTIMIZATION = False

        if PERFORM_OPTIMIZATION and not ground_truth_data:
            logger.warning("No ground truth data. Skipping optimization.")
            PERFORM_OPTIMIZATION = False

        if PERFORM_OPTIMIZATION:
            for algo_key, algo_config in cfg.ALGORITHMS.items():
                opt_settings = cfg.OPTIMIZATION_SETTINGS.get(algo_key)
                if not (opt_settings and opt_settings.get("enabled")):
                    logger.info(f"Optimization not enabled for {algo_key}. Using default parameters.")
                    optimized_algo_params[algo_key] = algo_config.get("params", {}).copy()
                    continue

                logger.info(f"Starting optimization for {algo_key}...")
                if dashboard: dashboard.add_log(f"Optimizing {algo_key}...")
                
                try:
                    best_params, best_score = optimize_parameters_for_algorithm(
                        algorithm_name_passed_in=algo_key,
                        base_algo_config=algo_config,
                        algo_module_name=algo_config["module_name"],
                        algo_fn_name=algo_config["function_name"],
                        parameters_to_tune_config=opt_settings["parameters_to_tune"],
                        image_path_subset=image_paths_for_opt,
                        evaluation_callback=_evaluation_callback_for_optimizer,
                        logger_instance=logger,
                        cfg_module=cfg,
                        eval_callback_additional_args=(ground_truth_data, cfg.H_SAMPLES)
                    )
                    logger.info(f"Optimization for {algo_key} complete. Best F1: {best_score:.4f}. Params: {best_params}")
                    if dashboard: dashboard.add_log(f"Optimization for {algo_key} complete: F1={best_score:.4f}")
                    optimized_algo_params[algo_key] = best_params
                except Exception as e:
                    logger.error(f"Error during optimization for {algo_key}: {e}", exc_info=True)
                    if dashboard: dashboard.add_log(f"Optimization failed for {algo_key}: {e}", 'error')
                    optimized_algo_params[algo_key] = algo_config.get("params", {}).copy()

    # --- Prepare Algorithm Configurations for Main Run ---
    final_algorithms_for_main_run = []
    for algo_key in cfg.ALGORITHMS_TO_RUN_KEYS:
        if algo_key not in cfg.ALGORITHMS:
            logger.warning(f"Algorithm key '{algo_key}' not in ALGORITHMS dict. Skipping.")
            continue
        
        algo_config = copy.deepcopy(cfg.ALGORITHMS[algo_key])
        
        # Apply optimized parameters if they exist, otherwise use defaults
        if PERFORM_OPTIMIZATION and algo_key in optimized_algo_params:
            logger.info(f"Applying optimized parameters to {algo_key} for the main run.")
            # The worker expects 'param_overrides' to be flat key-value pairs.
            algo_config["param_overrides"] = optimized_algo_params[algo_key]
        else:
            # Use default params from the config
            algo_config["param_overrides"] = algo_config.get("params", {})

        final_algorithms_for_main_run.append(algo_config)

    if not final_algorithms_for_main_run:
        logger.warning("No algorithms configured to run. Exiting.")
        return

    logger.info(f"--- Starting Main Comparison Run with {len(final_algorithms_for_main_run)} algorithm configurations ---")
    if dashboard: dashboard.update_phase("Main Comparison Run")
    for algo_conf in final_algorithms_for_main_run:
        logger.info(f"Will run: {algo_conf.get('display_name')}, Params: {algo_conf.get('param_overrides', 'Defaults')}")

    all_metrics_summary, images_processed_count = _process_images_and_write_video(
        image_filenames_to_process=image_list_for_main_run,
        cfg_obj=cfg,
        final_loaded_algorithms_configs=final_algorithms_for_main_run,
        dashboard=dashboard
    )

    # --- Finalization and Reporting ---
    logger.info(f"Processing and video writing completed for {images_processed_count} images.")
    if dashboard:
        dashboard.update_phase("Finalizing Results")
        dashboard.update_metrics(all_metrics_summary)
    
    if images_processed_count > 0 and all_metrics_summary:
        save_and_print_metrics_summary(
            all_metrics_summary,
            cfg.OUTPUT_VIS_DIR,
            metrics_filename=cfg.METRICS_SUMMARY_FILENAME
        )
        # The print_metrics_table_to_console function needs the final list of algorithms that were run
        lane_utils.print_metrics_table_to_console(all_metrics_summary, final_algorithms_for_main_run)

    if cfg.AUTO_PLAY_VIDEO_ON_COMPLETION:
        video_file_path = os.path.join(cfg.OUTPUT_VIS_DIR, cfg.OUTPUT_VIDEO_FILENAME)
        if os.path.exists(video_file_path):
            try:
                logger.info(f"Attempting to auto-play video: {video_file_path}")
                os.startfile(video_file_path)
            except Exception as e:
                logger.error(f"Failed to auto-play video: {e}")
            # Send the final video to the dashboard preview
            if dashboard:
                dashboard.update_preview_video(video_file_path)
        else:
            logger.warning(f"Video auto-play enabled, but file not found: {video_file_path}")

    logger.info("Lane comparison script completed.")
    if dashboard:
        dashboard.processing_complete()
        dashboard.add_log("Lane comparison script completed successfully!", 'success')


if __name__ == "__main__":
    #print("DEBUG: Inside if __name__ == '__main__' block") # DEBUG PRINT 2
    multiprocessing.freeze_support()
    #print("DEBUG: multiprocessing.freeze_support() called") # DEBUG PRINT 3
    main_comparison_orchestrator()
    #print("DEBUG: main_comparison_orchestrator() finished") # DEBUG PRINT 5

