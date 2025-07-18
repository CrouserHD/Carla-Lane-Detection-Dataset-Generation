\
import itertools
import numpy as np
import copy

DEFAULT_VARIANT_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Light Blue
    (128, 255, 0),  # Lime Green
    (255, 0, 128),  # Pink
    (0, 255, 128),  # Teal
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Dark Purple
    (0, 128, 128),  # Dark Teal
    (192, 192, 192),# Silver
    (128, 128, 128),# Gray
    (255, 165, 0),  # Orange (Web)
    (218, 112, 214),# Orchid
    (75, 0, 130),   # Indigo
    (255, 20, 147), # Deep Pink
    (0, 191, 255),  # Deep Sky Blue
    (50, 205, 50),  # Lime Green (Web)
    (255, 105, 180),# Hot Pink
    (255, 215, 0)   # Gold
]

def generate_hough_transform_parameter_sets():
    """Generates parameter sets for Hough Transform sweep."""
    hough_transform_parameter_sets = []
    # --- Parameter Ranges for Hough Transform Sweep (Fine-tuning) ---
    hough_rho_accuracies = [1] 
    hough_theta_accuracies_deg = [1] 
    hough_theta_accuracies_rad = [deg * np.pi / 180 for deg in hough_theta_accuracies_deg]

    hough_thresholds = [20, 21]
    hough_min_line_lengths = [18, 17]
    hough_max_line_gaps = [5]
    hough_canny_low_thresholds = [25]
    hough_canny_high_threshold_offsets = [70]

    param_combinations_hough = itertools.product(
        hough_rho_accuracies,
        hough_theta_accuracies_rad,
        hough_thresholds,
        hough_min_line_lengths,
        hough_max_line_gaps,
        hough_canny_low_thresholds
    )
    set_counter_hough = 0
    for rho, theta_rad, thresh, mll, mlg, clt in param_combinations_hough:
        for offset in hough_canny_high_threshold_offsets:
            cht = clt + offset
            if cht <= clt: continue
            set_counter_hough += 1
            theta_deg = int(round(theta_rad * 180 / np.pi))
            param_set_name = f"HSet{set_counter_hough}_R{rho}_T{theta_deg}_Th{thresh}_L{mll}_G{mlg}_CL{clt}_CH{cht}"
            hough_transform_parameter_sets.append({
                "name": param_set_name,
                "params": {
                    "HOUGH_RHO_ACCURACY": rho, "HOUGH_THETA_ACCURACY_RADIANS": theta_rad,
                    "HOUGH_THRESHOLD": thresh, "HOUGH_MIN_LINE_LENGTH": mll,
                    "HOUGH_MAX_LINE_GAP": mlg, "HOUGH_CANNY_LOW_THRESHOLD": clt,
                    "HOUGH_CANNY_HIGH_THRESHOLD": cht
                }
            })
    if hough_transform_parameter_sets:
        print(f"Generated {len(hough_transform_parameter_sets)} Hough Transform parameter sets.")
    return hough_transform_parameter_sets

def generate_advanced_sliding_window_parameter_sets():
    """Generates parameter sets for Advanced Sliding Window sweep."""
    advanced_sliding_window_parameter_sets = []
    asw_nwindows_range = [2]
    asw_margin_range = [50, 55, 60, 65, 70, 75]
    asw_minpix_range = [40, 45, 50, 55, 60]

    param_combinations_asw = itertools.product(
        asw_nwindows_range,
        asw_margin_range,
        asw_minpix_range
    )
    set_counter_asw = 0
    for nwindows, margin, minpix in param_combinations_asw:
        set_counter_asw += 1
        param_set_name = f"ASWSet{set_counter_asw}_NW{nwindows}_M{margin}_MP{minpix}"
        advanced_sliding_window_parameter_sets.append({
            "name": param_set_name,
            "params": {
                "ASW_NWINDOWS": nwindows,
                "ASW_MARGIN": margin,
                "ASW_MINPIX": minpix
            }
        })
    if advanced_sliding_window_parameter_sets:
        print(f"Generated {len(advanced_sliding_window_parameter_sets)} Advanced Sliding Window parameter sets.")
    return advanced_sliding_window_parameter_sets

def generate_carnd_pipeline_parameter_sets():
    """Generates parameter sets for CarND Pipeline color threshold sweep."""
    carnd_parameter_sets = []
    
    # Define ranges for RGB thresholds (min_val, max_val)
    # Focusing on the lower bound of the threshold as it's often more critical
    #rgb_thresh_min_vals = [i for i in range(188, 194, 1)]
    rgb_thresh_min_vals = [193]
    rgb_thresh_max_val = 255 # Keep max fixed for simplicity, or sweep this too

    # Define ranges for HLS thresholds (min_val, max_val)
    # Focusing on the lower bound
    #hls_thresh_min_vals = [i for i in range(190, 196, 1)]
    hls_thresh_min_vals = [195]
    hls_thresh_max_val = 255 # Keep max fixed

    param_combinations_carnd = itertools.product(
        rgb_thresh_min_vals,
        hls_thresh_min_vals
    )
    set_counter_carnd = 0
    for rgb_min, hls_min in param_combinations_carnd:
        set_counter_carnd += 1
        param_set_name = f"CarNDSet{set_counter_carnd}_RGB{rgb_min}-{rgb_thresh_max_val}_HLS{hls_min}-{hls_thresh_max_val}"
        carnd_parameter_sets.append({
            "name": param_set_name,
            "params": {
                "CARND_RGB_THRESH": (rgb_min, rgb_thresh_max_val),
                "CARND_HLS_THRESH": (hls_min, hls_thresh_max_val)
            }
        })
    if carnd_parameter_sets:
        print(f"Generated {len(carnd_parameter_sets)} CarND Pipeline parameter sets.")
    return carnd_parameter_sets

def expand_algorithms_for_sweeps(
    base_algorithms_to_run,
    perform_hough_sweep, hough_param_sets,
    perform_asw_sweep, asw_param_sets,
    perform_color_seg_sweep, color_seg_param_sets, # Assuming this was a placeholder or for another algo
    perform_carnd_sweep, carnd_param_sets, # Added for CarND
    variant_colors=None
):
    """
    Expands the list of algorithms to run with parameter variants if sweeps are enabled.
    """
    if variant_colors is None:
        variant_colors = DEFAULT_VARIANT_COLORS

    final_loaded_algorithms = []
    color_offset_counter = 0

    for algo_config_original in base_algorithms_to_run:
        if not algo_config_original["active"]:
            continue

        is_expanded = False
        base_display_name = algo_config_original.get('display_name', algo_config_original['module_name'])

        if algo_config_original['module_name'] == "hough_transform" and perform_hough_sweep and hough_param_sets:
            print(f"Expanding Hough Transform for parameter sweep with {len(hough_param_sets)} set(s).")
            for idx, param_set_config in enumerate(hough_param_sets):
                variant_algo_config = copy.deepcopy(algo_config_original)
                variant_algo_config["display_name"] = f"{base_display_name} ({param_set_config['name']})"
                variant_algo_config["param_overrides"] = param_set_config['params']
                variant_algo_config["color"] = variant_colors[(color_offset_counter + idx) % len(variant_colors)]
                final_loaded_algorithms.append(variant_algo_config)
            color_offset_counter += len(hough_param_sets)
            is_expanded = True

        elif algo_config_original['module_name'] == "advanced_sliding_window" and perform_asw_sweep and asw_param_sets:
            print(f"Expanding Advanced Sliding Window for parameter sweep with {len(asw_param_sets)} set(s).")
            for idx, param_set_config in enumerate(asw_param_sets):
                variant_algo_config = copy.deepcopy(algo_config_original)
                variant_algo_config["display_name"] = f"{base_display_name} ({param_set_config['name']})"
                variant_algo_config["param_overrides"] = param_set_config['params']
                variant_algo_config["color"] = variant_colors[(color_offset_counter + idx) % len(variant_colors)]
                final_loaded_algorithms.append(variant_algo_config)
            color_offset_counter += len(asw_param_sets)
            is_expanded = True
        
        elif algo_config_original['module_name'] == "carnd_pipeline_algorithm" and perform_carnd_sweep and carnd_param_sets:
            print(f"Expanding CarND Pipeline for parameter sweep with {len(carnd_param_sets)} set(s).")
            for idx, param_set_config in enumerate(carnd_param_sets):
                variant_algo_config = copy.deepcopy(algo_config_original)
                variant_algo_config["display_name"] = f"{base_display_name} ({param_set_config['name']})"
                variant_algo_config["param_overrides"] = param_set_config['params']
                # Assign a unique color if possible, cycling through variant_colors
                variant_algo_config["color"] = variant_colors[(color_offset_counter + idx) % len(variant_colors)]
                final_loaded_algorithms.append(variant_algo_config)
            color_offset_counter += len(carnd_param_sets)
            is_expanded = True

        if not is_expanded:
            algo_entry = copy.deepcopy(algo_config_original)
            if 'display_name' not in algo_entry:
                algo_entry['display_name'] = algo_entry['module_name']
            
            current_color = algo_config_original.get('color')
            if current_color is None: # Assign a color if not defined in config
                 current_color = variant_colors[color_offset_counter % len(variant_colors)]
                 color_offset_counter +=1
            algo_entry['color'] = current_color
            final_loaded_algorithms.append(algo_entry)

    return final_loaded_algorithms

# --- Self-Adjusting Parameter Optimizer ---

def optimize_parameters_for_algorithm(
    algorithm_name_passed_in, 
    base_algo_config,         
    algo_module_name,         
    algo_fn_name,             
    parameters_to_tune_config,
    image_path_subset,        
    evaluation_callback,      
    logger_instance,          
    cfg_module,               
    eval_callback_additional_args 
):
    """
    Optimizes parameters for a single algorithm using iterative hill climbing (coordinate ascent).

    Args:
        algorithm_name_passed_in (str): Name of the algorithm (e.g., "carnd_algorithm").
        base_algo_config (dict): The base configuration for the algorithm from cfg.ALGORITHMS.
                                 This contains initial 'parameters'.
        algo_module_name (str): The module name, e.g., "carnd_algorithm_modular".
        algo_fn_name (str): The function name, e.g., "detect_lanes_carnd_modular".
        parameters_to_tune_config (dict): Configuration for parameters to tune, from
                                          cfg.OPTIMIZATION_SETTINGS[algo_name]["parameters_to_tune"].
                                          Example: {"CARND_RGB_THRESH": {"min": 130, "max": 230, "step": 2, "initial_guess": 190, "tuple_index_to_tune": 0}}
        image_path_subset (list): List of image paths for evaluation.
        evaluation_callback (function): Function to call to evaluate parameters.
                                        Signature: callback(current_params_for_eval, base_algo_config, algo_module_name, algo_fn_name, 
                                                          image_path_subset, logger_callback, cfg_callback, *additional_args_for_callback)
        logger_instance (logging.Logger): Logger for optimizer messages.
        cfg_module (module): The main configuration module (cfg).
        eval_callback_additional_args (tuple): Additional arguments to pass to the evaluation_callback 
                                               (e.g., ground_truth_data, h_samples).

    Returns:
        tuple: (best_params_found, best_score_achieved)
    """
    logger_instance.info(f"[{algorithm_name_passed_in}] Starting optimization with base config: {base_algo_config.get('parameters', {})}")
    logger_instance.info(f"[{algorithm_name_passed_in}] Parameters to tune: {parameters_to_tune_config}")

    # Initialize current_best_params with the algorithm's default parameters from base_algo_config
    # These are the full set of parameters the algorithm function expects.
    current_best_params = base_algo_config.get("parameters", {}).copy()
    if not current_best_params:
        logger_instance.warning(f"[{algorithm_name_passed_in}] No base 'parameters' found in base_algo_config. Starting with an empty set.")

    # Override with initial guesses if provided in parameters_to_tune_config
    for param_name, tune_config in parameters_to_tune_config.items():
        if "initial_guess" in tune_config:
            initial_guess = tune_config["initial_guess"]
            if tune_config.get("tuple_index_to_tune") is not None: # Parameter is a tuple part
                tuple_index = tune_config["tuple_index_to_tune"]
                # Ensure the parameter exists in current_best_params and is a tuple/list
                if param_name in current_best_params and isinstance(current_best_params[param_name], (list, tuple)) and len(current_best_params[param_name]) > tuple_index:
                    # Create a mutable copy (list) to modify the specific index
                    param_tuple_list = list(current_best_params[param_name])
                    param_tuple_list[tuple_index] = initial_guess
                    current_best_params[param_name] = tuple(param_tuple_list) # Convert back to tuple if it was originally
                    logger_instance.debug(f"[{algorithm_name_passed_in}] Initialized tuple param {param_name}[{tuple_index}] to {initial_guess}. Full: {current_best_params[param_name]}")
                else:
                    logger_instance.warning(f"[{algorithm_name_passed_in}] Cannot set initial guess for tuple param {param_name}[{tuple_index}]. Param not found, not a tuple/list, or index out of bounds.")
            else: # Parameter is a direct value
                current_best_params[param_name] = initial_guess
                logger_instance.debug(f"[{algorithm_name_passed_in}] Initialized param {param_name} to {initial_guess}")
        elif param_name not in current_best_params:
            # If no initial guess and param not in base, it's an issue with config.
            # For safety, one might add a default (e.g., (tune_config['min'] + tune_config['max']) / 2)
            # but it's better to ensure `initial_guess` or base param exists.
            logger_instance.warning(f"[{algorithm_name_passed_in}] Parameter '{param_name}' has no initial guess and is not in base parameters. Optimization for it might be unstable.")


    # Initial evaluation with default/initial_guess parameters
    logger_instance.info(f"[{algorithm_name_passed_in}] Evaluating initial parameters: {current_best_params}")
    current_best_score = evaluation_callback(
        current_best_params, 
        base_algo_config, # Pass the original algo config for context if needed by callback
        algo_module_name, 
        algo_fn_name, 
        image_path_subset, 
        logger_instance, # Logger for the callback
        cfg_module,      # Config module for the callback
        *eval_callback_additional_args
    )
    logger_instance.info(f"[{algorithm_name_passed_in}] Initial score: {current_best_score:.4f}")

    max_iterations = getattr(cfg_module, "OPTIMIZER_MAX_ITERATIONS_OVERALL", 10)
    
    for iteration in range(max_iterations):
        logger_instance.info(f"[{algorithm_name_passed_in}] Optimization Iteration: {iteration + 1}/{max_iterations}")
        improved_in_iteration = False
        
        # Iterate through each parameter specified in parameters_to_tune_config
        for param_name, tune_config in parameters_to_tune_config.items():
            min_val, max_val, step = tune_config["min"], tune_config["max"], tune_config["step"]
            tuple_idx_to_tune = tune_config.get("tuple_index_to_tune")

            original_param_value_for_step = None
            if tuple_idx_to_tune is not None: # Tuning part of a tuple
                if param_name in current_best_params and isinstance(current_best_params[param_name], (list, tuple)) and len(current_best_params[param_name]) > tuple_idx_to_tune:
                    original_param_value_for_step = current_best_params[param_name][tuple_idx_to_tune]
                else:
                    logger_instance.error(f"[{algorithm_name_passed_in}] Misconfiguration for tuple parameter {param_name}. Skipping its optimization.")
                    continue
            else: # Tuning a direct parameter
                if param_name in current_best_params:
                    original_param_value_for_step = current_best_params[param_name]
                else:
                    logger_instance.error(f"[{algorithm_name_passed_in}] Parameter {param_name} not found in current_best_params. Skipping its optimization.")
                    continue
            
            if not isinstance(original_param_value_for_step, (int, float)):
                logger_instance.error(f"[{algorithm_name_passed_in}] Parameter {param_name} (or its part) is not numeric ({original_param_value_for_step}). Cannot optimize. Skipping.")
                continue

            # Try increasing the parameter
            next_val_up = min(original_param_value_for_step + step, max_val)
            if next_val_up != original_param_value_for_step: # Avoid re-evaluating if at max boundary and step makes no change
                params_to_test_up = current_best_params.copy()
                if tuple_idx_to_tune is not None:
                    temp_list = list(params_to_test_up[param_name])
                    temp_list[tuple_idx_to_tune] = next_val_up
                    params_to_test_up[param_name] = tuple(temp_list)
                else:
                    params_to_test_up[param_name] = next_val_up
                
                logger_instance.debug(f"[{algorithm_name_passed_in}] Testing {param_name} = {params_to_test_up[param_name]} (original: {original_param_value_for_step}, direction: up)")
                score_up = evaluation_callback(params_to_test_up, base_algo_config, algo_module_name, algo_fn_name, image_path_subset, logger_instance, cfg_module, *eval_callback_additional_args)
                if score_up > current_best_score:
                    current_best_score = score_up
                    current_best_params = params_to_test_up
                    improved_in_iteration = True
                    logger_instance.info(f"[{algorithm_name_passed_in}] Improvement: {param_name} to {params_to_test_up[param_name]}, New Score: {current_best_score:.4f}")

            # Try decreasing the parameter (if no improvement from increasing or to check other direction)
            # This check should be relative to the potentially updated current_best_params if 'up' was better
            current_value_for_down_check = None
            if tuple_idx_to_tune is not None:
                 current_value_for_down_check = current_best_params[param_name][tuple_idx_to_tune]
            else:
                 current_value_for_down_check = current_best_params[param_name]

            next_val_down = max(current_value_for_down_check - step, min_val)
            if next_val_down != current_value_for_down_check: # Avoid re-evaluating if at min boundary
                params_to_test_down = current_best_params.copy() # Use the latest best params
                if tuple_idx_to_tune is not None:
                    temp_list = list(params_to_test_down[param_name])
                    temp_list[tuple_idx_to_tune] = next_val_down
                    params_to_test_down[param_name] = tuple(temp_list)
                else:
                    params_to_test_down[param_name] = next_val_down

                logger_instance.debug(f"[{algorithm_name_passed_in}] Testing {param_name} = {params_to_test_down[param_name]} (original: {current_value_for_down_check}, direction: down)")
                score_down = evaluation_callback(params_to_test_down, base_algo_config, algo_module_name, algo_fn_name, image_path_subset, logger_instance, cfg_module, *eval_callback_additional_args)
                if score_down > current_best_score:
                    current_best_score = score_down
                    current_best_params = params_to_test_down
                    improved_in_iteration = True
                    logger_instance.info(f"[{algorithm_name_passed_in}] Improvement: {param_name} to {params_to_test_down[param_name]}, New Score: {current_best_score:.4f}")
        
        if not improved_in_iteration:
            logger_instance.info(f"[{algorithm_name_passed_in}] No improvement in iteration {iteration + 1}. Stopping optimization.")
            break
            
    logger_instance.info(f"[{algorithm_name_passed_in}] Optimization finished. Final Best Score: {current_best_score:.4f}, Final Best Params: {current_best_params}")
    return current_best_params, current_best_score

