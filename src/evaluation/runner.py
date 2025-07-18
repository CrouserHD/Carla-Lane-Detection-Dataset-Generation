#print("DEBUG: run_comparison.py script started") # DEBUG PRINT 0

import cv2
import config as cfg
import os
import json
import numpy as np
import importlib
from types import SimpleNamespace
import copy
import multiprocessing
import logging
import time
import random # Added for sampling images for optimization
import matplotlib.pyplot as plt
from . import metrics as lane_utils # Changed alias for clarity
from .metrics import (
    load_ground_truth_entry, 
    convert_culane_gt_to_points, 
    calculate_lane_metrics, 
    parse_command_line_arguments, 
    define_roi_vertices_from_config,
    save_and_print_metrics_summary,
    print_metrics_table_to_console
)
from .visualizer import create_comparison_image, save_image
from .optimizer import (
    optimize_parameters_for_algorithm # Import the optimizer function
)

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
    # --- Path Correction for Worker Process ---
    # Add project root to path to ensure 'src' can be found for module imports
    import sys
    # This ensures that the 'src' module can be found by the worker process
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # --- End Path Correction ---

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
    module_path = f"src.algorithms.{algo_config_worker['module_name']}"
    try:
        # Assuming this script (run_comparison.py) is part of the 'src.evaluation' package
        algo_module = importlib.import_module(module_path)
        detect_function = getattr(algo_module, algo_config_worker['function_name'])
    except Exception as e:
        worker_logger.error(f"Error loading algorithm {algo_name_worker}: {e}")
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
            worker_logger.debug(f"[{algo_name_worker}] Scaled back detected lanes for {image_filename_worker}")

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
    final_loaded_algorithms_configs
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
        else:
            logger.error("Received an empty result from an algorithm process. This should not happen if worker returns tuple.")


    logger.info("All algorithm processing has completed.")

    # --- Phase 2: Image/Video Generation (Sequential) ---
    logger.info("Starting image/video generation phase...")
    
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
            save_image(comparison_frame, cfg_obj.OUTPUT_VIS_DIR, output_image_filename)
        else:
            logger.warning(f"Comparison frame is None or empty for image {image_filename}. Skipping write.")
        
        images_processed_count_for_output += 1

    # --- Finalization ---
    if video_writer is not None and video_writer.isOpened():
        video_writer.release()
        logger.info(f"Video successfully saved: {os.path.join(cfg_obj.OUTPUT_VIS_DIR, cfg_obj.OUTPUT_VIDEO_FILENAME)}")
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
    # --- Path Correction for Optimizer Callback ---
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # --- End Path Correction ---

    logger_callback.debug(f"Optimizer eval callback: Algo: {algo_module_name}, Params: {current_params_for_eval}, Images: {len(image_path_subset)}")

    try:
        # Assuming src.algorithms is the correct path relative to where the script is run or Python's path
        algo_module = importlib.import_module(f"src.algorithms.{algo_module_name}")
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
                # Add other general config items if needed by the algo, e.g., H_SAMPLES if not passed elsewhere
            }
            # Merge the tuned algorithm-specific parameters into this full config
            current_full_config_for_eval.update(temp_algo_specific_config)


            processed_result = None
            if "carnd_algorithm" in algo_module_name or "hough_transform" in algo_module_name:
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


def plot_optimization_histories(histories, output_dir, logger_instance):
    """
    Plots the F1 score history for each algorithm from the optimization phase.
    """
    logger_instance.info("--- Plotting Optimization Histories ---")
    if not histories:
        logger_instance.info("No optimization histories to plot.")
        return

    for algo_name, history in histories.items():
        if not history:
            logger_instance.warning(f"No history data for {algo_name} to plot.")
            continue

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history, marker='o', linestyle='-', label=f'F1 Score per Iteration')
            plt.title(f'Optimization F1 Score History for {algo_name}')
            plt.xlabel('Evaluation Iteration')
            plt.ylabel('Average F1 Score')
            plt.grid(True)
            plt.legend()
            
            # Find best score and mark it
            if history:
                best_score = max(history)
                best_iter = history.index(best_score)
                plt.scatter(best_iter, best_score, color='red', zorder=5, label=f'Best F1: {best_score:.4f}')
                plt.legend()

            plot_filename = f"optimization_history_{algo_name.replace(' ', '_')}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            
            plt.savefig(plot_filepath)
            plt.close() # Close the figure to free memory
            logger_instance.info(f"Saved optimization plot for {algo_name} to {plot_filepath}")

        except Exception as e:
            logger_instance.error(f"Failed to plot optimization history for {algo_name}: {e}", exc_info=True)


def _load_ground_truth(gt_path, logger_instance):
    """Loads ground truth data from the specified JSON file."""
    ground_truth_data = {}
    if not os.path.exists(gt_path):
        logger_instance.warning(f"Ground truth file not found: {gt_path}.")
        return ground_truth_data

    try:
        with open(gt_path, 'r') as f:
            logger_instance.info(f"Attempting to load ground truth data from: {gt_path}")
            lines_read, entries_added = 0, 0
            for line_number, line in enumerate(f):
                lines_read += 1
                line = line.strip()
                if not line: continue
                try:
                    gt_entry = json.loads(line)
                    raw_file_path = gt_entry.get("raw_file")
                    if raw_file_path:
                        image_filename = os.path.basename(raw_file_path)
                        gt_lanes_data = gt_entry.get("lanes")
                        gt_h_samples_data = gt_entry.get("h_samples")
                        if gt_lanes_data is not None and gt_h_samples_data is not None:
                            ground_truth_data[image_filename] = {
                                "lanes": gt_lanes_data, "h_samples": gt_h_samples_data
                            }
                            entries_added += 1
                        else:
                            logger_instance.warning(f"GT entry for '{image_filename}' on line {line_number + 1} missing 'lanes' or 'h_samples'.")
                    else:
                        logger_instance.warning(f"GT entry in {gt_path} on line {line_number + 1} missing 'raw_file' key.")
                except json.JSONDecodeError as e_line:
                    logger_instance.error(f"Error decoding JSON line from {gt_path} on line {line_number + 1}: {e_line}.")
                except Exception as e_entry:
                    logger_instance.error(f"Unexpected error processing GT entry on line {line_number + 1}: {e_entry}.")
            logger_instance.info(f"GT Load: Read {lines_read} lines. Successfully processed {entries_added} entries.")
        if not ground_truth_data:
            logger_instance.warning(f"Ground truth file {gt_path} was read, but no valid entries were processed.")
    except Exception as e_file:
        logger_instance.error(f"Failed to read/process ground truth file {gt_path}: {e_file}.")
    
    return ground_truth_data


def _prepare_image_list(source_dir, args, logger_instance):
    """Prepares the list of image files to process based on source directory and arguments."""
    try:
        logger_instance.info(f"Attempting to list files from IMAGE_SOURCE_DIR: {source_dir}")
        available_image_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        logger_instance.info(f"Found {len(available_image_files)} image files.")
    except FileNotFoundError:
        logger_instance.error(f"Image directory not found: {source_dir}")
        return None, None
    
    if not available_image_files:
        logger_instance.warning(f"No images found in {source_dir}.")
        return [], []

    start_index = max(0, args.start_index)
    if start_index >= len(available_image_files):
        logger_instance.error(f"start_index ({start_index}) out of bounds.")
        return None, None

    images_to_consider = available_image_files[start_index:]
    num_to_process = len(images_to_consider)
    if args.num_images is not None and args.num_images > 0:
        num_to_process = min(len(images_to_consider), args.num_images)
    
    if num_to_process == 0:
        logger_instance.warning("No images to process after applying start_index and num_images.")
        return [], available_image_files
        
    image_list_for_run = images_to_consider[:num_to_process]
    logger_instance.info(f"Main run will process {len(image_list_for_run)} images, starting with {image_list_for_run[0] if image_list_for_run else 'N/A'}.")
    return image_list_for_run, available_image_files


def _run_optimization_phase(cfg_module, all_images, gt_data, logger_instance):
    """Runs the parameter optimization phase for configured algorithms."""
    logger_instance.info("--- Starting Parameter Optimization Phase ---")
    optimized_params = {}
    histories = {}

    num_images_for_opt = min(len(all_images), getattr(cfg_module, "NUM_IMAGES_FOR_OPTIMIZATION_PHASE", 50))
    if num_images_for_opt <= 0 or not all_images:
        logger_instance.warning("Not enough images or NUM_IMAGES_FOR_OPTIMIZATION_PHASE is 0. Skipping optimization.")
        return {}, {}

    if not gt_data:
        logger_instance.warning("No ground truth data loaded. Parameter optimization requires GT and will be skipped.")
        return {}, {}

    if not hasattr(cfg_module, 'ALGORITHMS') or not isinstance(cfg_module.ALGORITHMS, dict):
        logger_instance.error("cfg.ALGORITHMS not defined or not a dictionary. Cannot perform optimization.")
        return {}, {}

    image_paths_for_opt = random.sample(
        [os.path.join(cfg_module.IMAGE_SOURCE_DIR, f) for f in all_images],
        min(num_images_for_opt, len(all_images))
    )
    logger_instance.info(f"Selected {len(image_paths_for_opt)} images randomly for optimization.")

    for algo_key, algo_cfg in cfg_module.ALGORITHMS.items():
        default_params = algo_cfg.get("parameters", {}).copy()
        opt_settings = cfg_module.OPTIMIZATION_SETTINGS.get(algo_key)

        if not (opt_settings and opt_settings.get("enabled") and opt_settings.get("parameters_to_tune")):
            logger_instance.info(f"Optimization not enabled for {algo_key}. Using default parameters.")
            optimized_params[algo_key] = default_params
            continue

        logger_instance.info(f"Starting parameter optimization for {algo_key}...")
        algo_module_name = algo_cfg.get("module")
        algo_fn_name = algo_cfg.get("function")

        if not algo_module_name or not algo_fn_name:
            logger_instance.error(f"Algorithm {algo_key} is missing 'module' or 'function'. Skipping optimization.")
            optimized_params[algo_key] = default_params
            continue

        try:
            best_params, best_score, score_history = optimize_parameters_for_algorithm(
                algorithm_name_passed_in=algo_key,
                base_algo_config=algo_cfg,
                algo_module_name=algo_module_name,
                algo_fn_name=algo_fn_name,
                parameters_to_tune_config=opt_settings["parameters_to_tune"],
                image_path_subset=image_paths_for_opt,
                evaluation_callback=_evaluation_callback_for_optimizer,
                logger_instance=logger_instance,
                cfg_module=cfg_module,
                eval_callback_additional_args=(gt_data, cfg_module.H_SAMPLES)
            )
            logger_instance.info(f"Optimization for {algo_key} complete. Best F1: {best_score:.4f}. Params: {best_params}")
            optimized_params[algo_key] = best_params
            histories[algo_key] = score_history
        except Exception as e_opt:
            logger_instance.error(f"Error during optimization for {algo_key}: {e_opt}", exc_info=True)
            optimized_params[algo_key] = default_params

    return optimized_params, histories


def _prepare_main_run_configs(cfg_module, optimized_algo_params, perform_optimization):
    """Prepares the final list of algorithm configurations for the main run."""
    final_algorithms = []
    for algo_run_config in cfg_module.ALGORITHMS_TO_RUN:
        if not algo_run_config.get("active", False):
            continue

        display_name = algo_run_config.get("display_name")
        if not display_name:
            logger.warning(f"Algorithm config in ALGORITHMS_TO_RUN missing 'display_name'. Skipping.")
            continue
            
        current_config = copy.deepcopy(algo_run_config)
        
        optimization_key = None
        if hasattr(cfg_module, 'ALGORITHMS') and isinstance(cfg_module.ALGORITHMS, dict):
            for opt_key, opt_cfg in cfg_module.ALGORITHMS.items():
                if opt_cfg.get("module") == current_config.get("module_name") and \
                   opt_cfg.get("function") == current_config.get("function_name"):
                    optimization_key = opt_key
                    break
        
        if optimization_key and optimization_key in optimized_algo_params:
            logger.info(f"Applying optimized parameters to {display_name} (key: {optimization_key}).")
            current_config["param_overrides"] = optimized_algo_params[optimization_key]
        elif perform_optimization:
             logger.info(f"No optimized parameters found for {display_name}. Using defaults.")
             if "params" in current_config:
                 current_config["param_overrides"] = current_config["params"]

        final_algorithms.append(current_config)
    
    return final_algorithms


def main():
    args = parse_command_line_arguments(cfg)

    # --- Configuration & Setup ---
    PERFORM_OPTIMIZATION = getattr(cfg, "PERFORM_OPTIMIZATION_PHASE", True)
    if not os.path.exists(cfg.OUTPUT_VIS_DIR):
        os.makedirs(cfg.OUTPUT_VIS_DIR, exist_ok=True)

    # --- Data and Image List Preparation ---
    ground_truth_data = _load_ground_truth(cfg.JSON_GT_PATH, logger)
    image_list_for_main_run, available_image_files = _prepare_image_list(cfg.IMAGE_SOURCE_DIR, args, logger)

    if image_list_for_main_run is None: # Indicates a fatal error like directory not found
        return {}, 0
    if not image_list_for_main_run: # No images to process
        logger.warning("No images selected for the main run. Exiting.")
        return {}, 0

    # --- Parameter Optimization Phase ---
    optimized_algo_params = {}
    if PERFORM_OPTIMIZATION:
        optimized_algo_params, optimization_histories = _run_optimization_phase(cfg, available_image_files, ground_truth_data, logger)
        if optimization_histories:
            plot_optimization_histories(optimization_histories, cfg.OUTPUT_VIS_DIR, logger)
    
    # --- Prepare Algorithm Configurations for Main Run ---
    final_algorithms_for_main_run = _prepare_main_run_configs(cfg, optimized_algo_params, PERFORM_OPTIMIZATION)

    if not final_algorithms_for_main_run:
        logger.warning("No algorithms are configured to run. Exiting.")
        return {}, 0

    logger.info(f"--- Starting Main Comparison Run with {len(final_algorithms_for_main_run)} algorithm configurations ---")
    for algo_conf_log in final_algorithms_for_main_run:
        logger.info(f"Will run: {algo_conf_log.get('display_name')}, Module: {algo_conf_log.get('module_name')}, Params: {algo_conf_log.get('param_overrides', 'Defaults')}")

    # --- Main Processing and Video Generation ---
    all_metrics_summary, images_processed_count = _process_images_and_write_video(
        image_filenames_to_process=image_list_for_main_run,
        cfg_obj=cfg,
        final_loaded_algorithms_configs=final_algorithms_for_main_run
    )

    # --- Finalization and Reporting ---
    logger.info(f"Processing and video writing completed for {images_processed_count} images.")
    
    if images_processed_count > 0 and all_metrics_summary:
        logger.info(f"Results and metrics saved. Check output directory: {cfg.OUTPUT_VIS_DIR}")
        lane_utils.save_and_print_metrics_summary(
            all_metrics_summary, 
            cfg.OUTPUT_VIS_DIR, 
            metrics_filename=getattr(cfg, "METRICS_SUMMARY_FILENAME", "metrics_summary.json")
        )
        lane_utils.print_metrics_table_to_console(all_metrics_summary, cfg.ALGORITHMS_TO_RUN)
    elif images_processed_count == 0:
        logger.warning("No images were processed successfully in the main run.")
    else:
        logger.warning("Images processed, but no metrics summary was generated.")
    
    # --- Script Completion ---
    if hasattr(cfg, 'AUTO_PLAY_VIDEO_ON_COMPLETION') and cfg.AUTO_PLAY_VIDEO_ON_COMPLETION:
        video_file_path = os.path.join(cfg.OUTPUT_VIS_DIR, cfg.OUTPUT_VIDEO_FILENAME)
        if os.path.exists(video_file_path):
            try:
                logger.info(f"Attempting to auto-play video: {video_file_path}")
                os.startfile(video_file_path) # Windows specific
            except AttributeError:
                logger.warning("os.startfile is not available on this system. Cannot auto-play video.")
            except Exception as e:
                logger.error(f"Failed to auto-play video {video_file_path}: {e}")
        else:
            logger.warning(f"Video auto-play enabled, but video file not found: {video_file_path}")

    logger.info("Lane comparison script completed.")
    return all_metrics_summary, images_processed_count


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

