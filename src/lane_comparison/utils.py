import os
import json
import numpy as np
import cv2
from tabulate import tabulate
import colorama
from colorama import Fore, Style # Import Fore and Style
import argparse

# Initialize colorama
colorama.init(autoreset=True) # Added autoreset=True

GT_COLOR = (0, 255, 0)      # Grün für Ground Truth
ALGO1_HOUGH_COLOR = (255, 0, 0) # Blau
ALGO2_COLOR_COLOR = (0, 0, 255) # Rot
ALGO3_DL_COLOR = (255, 255, 0) # Cyan

def load_ground_truth_entry(gt_json_path, image_filename_to_find):
    try:
        with open(gt_json_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if os.path.basename(entry.get("raw_file", "")) == image_filename_to_find:
                    return entry
    except FileNotFoundError:
        print(f"Fehler: Ground truth Datei nicht gefunden: {gt_json_path}")
    except json.JSONDecodeError as e_json: # Renamed exception variable
        print(f"Fehler: JSON konnte nicht dekodiert werden: {gt_json_path}. Details: {e_json}") # Used renamed variable
        return None
    return None

def convert_culane_gt_to_points(gt_lanes_culane_format, h_samples):
    ground_truth_lanes_points = []
    if not gt_lanes_culane_format or not h_samples:
        return ground_truth_lanes_points
    for lane_x_coords in gt_lanes_culane_format:
        current_lane_points = []
        if len(lane_x_coords) != len(h_samples):
            print(f"Warnung: Inkonsistenz zw. x_coords und h_samples in GT. Überspringe Spur.")
            continue
        for x_coord, y_coord in zip(lane_x_coords, h_samples):
            if x_coord >= 0:
                current_lane_points.append((int(x_coord), int(y_coord)))
        if len(current_lane_points) >= 2:
            ground_truth_lanes_points.append(current_lane_points)
    return ground_truth_lanes_points

def draw_single_lane_set(image, lane_set, color, thickness):
    """
    Draws a single set of lanes (e.g., from one algorithm or ground truth).
    A lane_set can be:
    1. A list of line segments: [ [[x1,y1,x2,y2]], [[x3,y3,x4,y4]], ... ]
    2. A list of polylines (contours/paths): [ [[x1,y1],[x2,y2],...], [[x1b,y1b],[x2b,y2b],...] ]
    """
    if lane_set is None:
        return image # Return the original image if lane_set is None

    for lane_element in lane_set:
        # Skip if lane_element is effectively empty
        is_effectively_empty = False
        if hasattr(lane_element, 'size'): # Checks if it's a NumPy array or similar
            if lane_element.size == 0:
                is_effectively_empty = True
        elif not lane_element: # For lists and other sequences that are empty
            is_effectively_empty = True
        
        if is_effectively_empty:
            continue

        # Check if it's a line segment: [[x1,y1,x2,y2]]
        if isinstance(lane_element, list) and len(lane_element) == 1 and \
           isinstance(lane_element[0], list) and len(lane_element[0]) == 4:
            x1, y1, x2, y2 = lane_element[0]
            try:
                cv2.line(image, (int(x1), int(y1)), (int(x2, int(y2))), color, thickness)
            except Exception as e:
                print(f"Error drawing line: {lane_element} with error {e}")
        
        # Check if it's a polyline: [[x1,y1],[x2,y2],...]
        elif isinstance(lane_element, list) and \
             all(isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2 for pt in lane_element):
            try:
                pts = np.array(lane_element, dtype=np.int32)
                if pts.ndim == 2 and pts.shape[1] == 2: # Should be N x 2
                    pts = pts.reshape((-1, 1, 2)) # Reshape to N x 1 x 2 for cv2.polylines
                    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
                else:
                    print(f"Warning: Polyline data has unexpected shape: {pts.shape}. Data: {lane_element}")
            except Exception as e:
                print(f"ERROR in draw_single_lane_set: {e}")
                import traceback
                print(traceback.format_exc())
                current_lane_element = locals().get('lane_element', 'lane_element not defined in local scope')
                print(f"Problematic lane_element type: {type(current_lane_element)}")
                if hasattr(current_lane_element, 'shape'):
                    print(f"Problematic lane_element shape: {current_lane_element.shape}")
                if hasattr(current_lane_element, '__len__'):
                    print(f"Problematic lane_element length: {len(current_lane_element)}")
                try:
                    print(f"Problematic lane_element (first 5 items/points): {str(current_lane_element[:5]) if hasattr(current_lane_element, '__getitem__') else str(current_lane_element)}")
                except:
                    try:
                        print(f"Problematic lane_element (raw): {str(current_lane_element)}")
                    except:
                        print("Could not print problematic lane_element.")
        
        # Fallback for a simple list of 4 coordinates (less preferred, but for safety)
        elif isinstance(lane_element, (list, tuple, np.ndarray)) and len(lane_element) == 4:
            try:
                x1, y1, x2, y2 = map(int, lane_element)
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
            except Exception as e:
                print(f"Error drawing fallback line: {lane_element} with error {e}")
    return image # Ensure the image is always returned

def get_x_at_y(polyline_points, y_target):
    """
    Interpolates the x-coordinate on a polyline at a given y_target.
    Assumes polyline_points is a list or array of (x,y) tuples or lists, sorted by y if possible,
    but will iterate through segments regardless of order.
    Returns None if y_target is outside the polyline's y-range or if polyline is empty/invalid.
    """
    # Check if polyline_points is None or has no points (shape[0] < 1 for numpy array)
    if polyline_points is None or polyline_points.shape[0] < 1:
        return None

    # If only one point, cannot form a segment for interpolation
    if polyline_points.shape[0] < 2:
        # Check if this single point happens to be at y_target
        if abs(float(polyline_points[0][1]) - float(y_target)) < 1e-6: # Compare y-coordinate
            return float(polyline_points[0][0]) # Return x-coordinate
        return None

    # Iterate over line segments defined by consecutive points
    for i in range(polyline_points.shape[0] - 1):
        p1 = polyline_points[i]
        p2 = polyline_points[i+1]

        # Ensure points are tuples/lists of two numbers and convert to float
        try:
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
        except (TypeError, ValueError, IndexError):
            continue

        # Check if y_target is within the y-range of the current segment
        y_min_segment = min(y1, y2)
        y_max_segment = max(y1, y2)

        if float(y_target) >= y_min_segment - 1e-6 and float(y_target) <= y_max_segment + 1e-6:
            # Handle horizontal segment: if y1 is very close to y2
            if abs(y1 - y2) < 1e-6:
                continue 

            # Handle vertical segment: if x1 is very close to x2
            if abs(x1 - x2) < 1e-6:
                return x1

            # Interpolate for non-horizontal, non-vertical segments
            interpolated_x = x1 + (x2 - x1) * (float(y_target) - y1) / (y2 - y1)
            return interpolated_x
            
    return None

def calculate_lane_metrics(gt_lanes_points, detected_lanes_polylines, h_samples, threshold_px):
    """
    Calculates quantitative metrics for lane detection.
    - gt_lanes_points: List of ground truth lanes. Each lane is a list of (x,y) points,
                       where y values are from h_samples.
                       Example: [[(x1,y1), (x2,y2)], [(x3,y3), (x4,y4)]]
    - detected_lanes_polylines: List of detected lanes. Each lane is a polyline (list of (x,y) points).
                               Example: [[(x1,y1), (x2,y2), ...], [...]]
    - h_samples: List of y-coordinates at which to evaluate.
    - threshold_px: Pixel distance threshold for considering a point matched.
    Returns a dictionary of metrics.
    """
    num_gt_lanes = 0
    if gt_lanes_points: # Ensure gt_lanes_points is not None
        num_gt_lanes = len(gt_lanes_points)
    
    num_det_lanes = 0
    if detected_lanes_polylines: # Ensure detected_lanes_polylines is not None
        num_det_lanes = len(detected_lanes_polylines)

    tp = 0 # True Positives: GT lanes correctly detected
    matched_det_indices = set() # Indices of detected lanes that have been matched to a GT lane
    
    accumulated_x_diff_sum = 0.0 # Sum of x-differences for all correctly matched points
    accumulated_matched_points_count = 0 # Count of all correctly matched points

    # Pre-sample all detected lanes at h_samples
    det_lanes_sampled_at_h = []
    if detected_lanes_polylines:
        for det_poly_lane_input in detected_lanes_polylines:
            # Ensure det_poly_lane is a numpy array for consistent processing
            if isinstance(det_poly_lane_input, list):
                if not det_poly_lane_input: # Empty list
                    det_poly_lane = np.array([])
                else:
                    det_poly_lane = np.array(det_poly_lane_input, dtype=np.float32)
            elif isinstance(det_poly_lane_input, np.ndarray):
                det_poly_lane = det_poly_lane_input
            else: # Skip if it's an unexpected type or None
                det_lanes_sampled_at_h.append({})
                continue

            sampled_points_map = {}  # y_coord -> x_coord
            if det_poly_lane.size > 0 and det_poly_lane.ndim == 2 and det_poly_lane.shape[1] == 2: # Check if it has points and is Nx2
                for h_val in h_samples:
                    x_interp = get_x_at_y(det_poly_lane, h_val)
                    if x_interp is not None:
                        sampled_points_map[h_val] = x_interp
            det_lanes_sampled_at_h.append(sampled_points_map)

    if gt_lanes_points:
        for i in range(num_gt_lanes):
            gt_lane_points_list = gt_lanes_points[i] 
            if not gt_lane_points_list: 
                continue
                
            gt_lane_map = {int(y): int(x) for x, y in gt_lane_points_list if x is not None and y is not None} # Ensure x,y are not None
            num_points_in_this_gt_lane_at_h = len(gt_lane_map)
            if num_points_in_this_gt_lane_at_h == 0:
                continue

            best_match_det_idx = -1
            max_matched_points_for_this_gt = 0 
            min_avg_x_diff_for_this_gt = float('inf') 
            
            x_diff_sum_for_best_match_with_this_gt = 0
            num_points_for_best_match_with_this_gt = 0

            for j in range(num_det_lanes):
                if j in matched_det_indices: 
                    continue

                det_lane_sampled_map = det_lanes_sampled_at_h[j]
                if not det_lane_sampled_map: 
                    continue

                current_match_x_diff_sum_for_pair = 0.0
                current_match_points_count_for_pair = 0
                
                for h_val, x_gt in gt_lane_map.items(): # h_val is int due to previous conversion
                    if h_val in det_lane_sampled_map:
                        x_det = det_lane_sampled_map[h_val]
                        dist = abs(x_gt - x_det)
                        if dist <= threshold_px:
                            current_match_x_diff_sum_for_pair += dist
                            current_match_points_count_for_pair += 1
                
                if current_match_points_count_for_pair > 0: 
                    if current_match_points_count_for_pair > max_matched_points_for_this_gt:
                        max_matched_points_for_this_gt = current_match_points_count_for_pair
                        min_avg_x_diff_for_this_gt = current_match_x_diff_sum_for_pair / current_match_points_count_for_pair
                        best_match_det_idx = j
                        x_diff_sum_for_best_match_with_this_gt = current_match_x_diff_sum_for_pair
                        num_points_for_best_match_with_this_gt = current_match_points_count_for_pair
                    elif current_match_points_count_for_pair == max_matched_points_for_this_gt:
                        current_avg_x_diff = current_match_x_diff_sum_for_pair / current_match_points_count_for_pair
                        if current_avg_x_diff < min_avg_x_diff_for_this_gt:
                            min_avg_x_diff_for_this_gt = current_avg_x_diff
                            best_match_det_idx = j
                            x_diff_sum_for_best_match_with_this_gt = current_match_x_diff_sum_for_pair
                            num_points_for_best_match_with_this_gt = current_match_points_count_for_pair
            
            is_good_match = False
            if best_match_det_idx != -1 and max_matched_points_for_this_gt >= 1:
                # Match criteria: at least 1 point, and it was the best one found for this GT lane.
                # And, number of matched points >= 5% of GT points (or at least 1).
                if max_matched_points_for_this_gt >= max(1, 0.05 * num_points_in_this_gt_lane_at_h): # Changed 0.1 to 0.05
                     is_good_match = True

            if is_good_match:
                tp += 1
                matched_det_indices.add(best_match_det_idx)
                accumulated_x_diff_sum += x_diff_sum_for_best_match_with_this_gt
                accumulated_matched_points_count += num_points_for_best_match_with_this_gt
                
    fn = num_gt_lanes - tp 
    fp = num_det_lanes - len(matched_det_indices) 

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / num_gt_lanes if num_gt_lanes > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_x_diff_on_matched_pts = accumulated_x_diff_sum / accumulated_matched_points_count \
        if accumulated_matched_points_count > 0 else float('nan')

    return {
        "num_gt_lanes": num_gt_lanes,
        "num_detected_lanes": num_det_lanes,
        "true_positives (matched_gt_lanes)": tp,
        "false_positives (unmatched_det_lanes)": fp,
        "false_negatives (unmatched_gt_lanes)": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_x_diff_matched_points": avg_x_diff_on_matched_pts,
        "total_matched_points_count": accumulated_matched_points_count
    }

def parse_command_line_arguments(cfg_module):
    """
    Parses command-line arguments for the lane comparison script.
    Args:
        cfg_module: The configuration module (e.g., lane_comparison_config).
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run lane detection comparison.")

    default_start_index = getattr(cfg_module, 'START_IMAGE_INDEX', 0)

    # Determine default for --num_images argument based on config
    val_process_num_images = getattr(cfg_module, 'PROCESS_NUM_IMAGES', None)
    val_max_images = getattr(cfg_module, 'MAX_IMAGES_TO_PROCESS', None)
    default_num_images_arg = 0 # Default fallback

    if hasattr(cfg_module, 'PROCESS_NUM_IMAGES'): # Check existence first
        default_num_images_arg = val_process_num_images
    elif hasattr(cfg_module, 'MAX_IMAGES_TO_PROCESS'): # Then this
        default_num_images_arg = val_max_images
    
    # Ensure it's an int if not None, as argparse type=int expects an int default
    if default_num_images_arg is None:
        default_num_images_arg = 0

    parser.add_argument(
        "--start_index",
        type=int,
        default=default_start_index,
        help=(
            "0-based index of the first image to process. "
            f"Default from config: {default_start_index}"
        )
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=default_num_images_arg,
        help=(
            "Number of images to process from start_index. "
            "A value of 0 means process all available images from start_index (or up to a global MAX from config if applicable). "
            f"Current config-derived default for this argument: {default_num_images_arg}. "
            f"(Relevant config values: PROCESS_NUM_IMAGES: {val_process_num_images if val_process_num_images is not None else 'N/A'}, "
            f"MAX_IMAGES_TO_PROCESS: {val_max_images if val_max_images is not None else 'N/A'})"
        )
    )
    return parser.parse_args()

def define_roi_vertices_from_config(image_shape, cfg_module):
    """
    Defines ROI vertices based on image_shape and config parameters.
    Args:
        image_shape: Tuple (height, width, [channels]) of the image.
        cfg_module: The configuration module (e.g., lane_comparison_config).
    Returns:
        A NumPy array of shape (1, 4, 2) defining the ROI vertices.
    """
    height, width = image_shape[:2]
    roi_tl_x = int(width * cfg_module.ROI_X_START_RATIO)
    roi_tl_y = int(height * cfg_module.ROI_Y_RATIO)
    roi_tr_x = int(width * cfg_module.ROI_X_END_RATIO)
    roi_tr_y = int(height * cfg_module.ROI_Y_RATIO)
    roi_bl_x = int(width * (cfg_module.ROI_X_START_RATIO - cfg_module.ROI_BOTTOM_WIDTH_FACTOR_OFFSET))
    roi_bl_y = int(height * cfg_module.ROI_Y_END_RATIO)
    roi_br_x = int(width * (cfg_module.ROI_X_END_RATIO + cfg_module.ROI_BOTTOM_WIDTH_FACTOR_OFFSET))
    roi_br_y = int(height * cfg_module.ROI_Y_END_RATIO)
    
    roi_bl_x = max(0, roi_bl_x)
    roi_br_x = min(width, roi_br_x)
    
    return np.array([[
        (roi_tl_x, roi_tl_y), 
        (roi_tr_x, roi_tr_y), 
        (roi_br_x, roi_br_y), 
        (roi_bl_x, roi_bl_y) 
    ]], dtype=np.int32)

def save_and_print_metrics_summary(all_metrics_summary, output_dir, metrics_filename="metrics_summary.txt"):
    """
    Calculates, prints, and saves the summary of metrics for all algorithms.
    Args:
        all_metrics_summary (dict): A dictionary where keys are algorithm names and
                                    values are lists of metric dicts for each image.
        output_dir (str): Directory to save the metrics summary file.
        metrics_filename (str): Filename for the metrics summary.
    """
    print("\nQuantitative Metrics Summary:")
    metrics_output_path = os.path.join(output_dir, metrics_filename)

    summary_lines_for_file = ["Quantitative Metrics Summary:\n"]
    summary_lines_for_console = ["\nQuantitative Metrics Summary:"]

    for algo_name, metrics_list in all_metrics_summary.items():
        file_algo_lines = [f"  Algorithm: {algo_name}\n"]
        console_algo_lines = [f"  Algorithm: {algo_name}"]

        if metrics_list:
            # Helper to extract and filter valid metric values
            def get_valid_metrics(metric_name):
                return [
                    m[metric_name] for m in metrics_list 
                    if m and metric_name in m and isinstance(m[metric_name], (int, float)) and \
                    (metric_name != "avg_x_diff_matched_points" or not np.isnan(m[metric_name]))
                ]

            f1_scores = get_valid_metrics("f1_score")
            precisions = get_valid_metrics("precision")
            recalls = get_valid_metrics("recall")
            tps = get_valid_metrics("true_positives (matched_gt_lanes)")
            fps = get_valid_metrics("false_positives (unmatched_det_lanes)")
            fns = get_valid_metrics("false_negatives (unmatched_gt_lanes)")
            x_diffs = get_valid_metrics("avg_x_diff_matched_points")

            avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
            avg_precision = np.mean(precisions) if precisions else 0.0
            avg_recall = np.mean(recalls) if recalls else 0.0
            avg_tp = np.mean(tps) if tps else 0.0
            avg_fp = np.mean(fps) if fps else 0.0
            avg_fn = np.mean(fns) if fns else 0.0
            avg_x_diff = np.mean(x_diffs) if x_diffs else float('nan')

            # Calculate average execution time and FPS
            exec_times = [
                m['exec_time'] for m in metrics_list 
                if m and 'exec_time' in m and isinstance(m['exec_time'], (int, float))
            ]
            
            avg_exec_time_per_image = 0.0
            if exec_times:
                avg_exec_time_per_image = np.mean([t for t in exec_times if t > 0]) if any(t > 0 for t in exec_times) else 0.0

            avg_fps = 0.0
            if avg_exec_time_per_image > 0:
                avg_fps = 1.0 / avg_exec_time_per_image
            elif exec_times: # if there were exec_times but all were <=0, avg_fps is 0
                avg_fps = 0.0
            # if no exec_times, avg_fps remains 0.0 (effectively infinite if no time taken, or undefined)


            metric_strings = [
                f"    Avg F1 Score: {avg_f1:.3f}",
                f"    Avg Precision: {avg_precision:.3f}",
                f"    Avg Recall: {avg_recall:.3f}",
                f"    Avg True Positives: {avg_tp:.2f}",
                f"    Avg False Positives: {avg_fp:.2f}",
                f"    Avg False Negatives: {avg_fn:.2f}",
                f"    Avg X Diff (matched pts): {avg_x_diff:.2f} px",
                f"    Avg Execution Time per Image: {avg_exec_time_per_image:.4f} seconds",
                f"    Avg FPS: {avg_fps:.2f}",
                f"    Total Images Processed for this algo: {len(metrics_list)}"
            ]
            
            for s in metric_strings:
                file_algo_lines.append(s + "\n")
                console_algo_lines.append(s)
            file_algo_lines.append("\n") # Extra newline for file per algo

        else:
            no_metrics_msg = "    No metrics calculated (possibly disabled, no GT, or error during processing)."
            file_algo_lines.append(no_metrics_msg + "\n\n")
            console_algo_lines.append(no_metrics_msg)
        
        summary_lines_for_file.extend(file_algo_lines)
        summary_lines_for_console.extend(console_algo_lines)

    try:
        with open(metrics_output_path, "w") as f_metrics:
            f_metrics.writelines(summary_lines_for_file)
        print(f"Metrics summary also saved to: {metrics_output_path}")
    except Exception as e:
        print(f"Error writing metrics summary to file: {e}")

    # Print to console
    for line in summary_lines_for_console:
        print(line)

# ANSI escape codes for colors (now using colorama objects)
class Colors:
    RESET = Style.RESET_ALL
    BLACK = Fore.BLACK
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    BRIGHT_BLACK = Fore.LIGHTBLACK_EX
    BRIGHT_RED = Fore.LIGHTRED_EX
    BRIGHT_GREEN = Fore.LIGHTGREEN_EX
    BRIGHT_YELLOW = Fore.LIGHTYELLOW_EX
    BRIGHT_BLUE = Fore.LIGHTBLUE_EX
    BRIGHT_MAGENTA = Fore.LIGHTMAGENTA_EX
    BRIGHT_CYAN = Fore.LIGHTCYAN_EX
    BRIGHT_WHITE = Fore.LIGHTWHITE_EX

# Map algorithm display names (or a unique key) to console colors
# This is a suggestion; you might need to get the actual algo names and their configured BGR colors
# and map BGR to an appropriate ANSI color name if you want to use the algo's video color.
# For simplicity, let's define a few fixed color assignments for console output.
CONSOLE_COLOR_MAP = {
    "Hough Transform": Colors.CYAN,
    "Adv. Sliding Window": Colors.YELLOW,
    "CarND Pipeline": Colors.MAGENTA,
    "LaneNet": Colors.GREEN,
    # Add more or make this dynamic based on cfg.ALGORITHMS_TO_RUN
}
DEFAULT_CONSOLE_COLOR = Fore.WHITE # Fallback

def print_metrics_table_to_console(all_metrics_summary, algorithm_configs):
    """
    Prints a formatted table of key metrics to the console, with colors.

    Args:
        all_metrics_summary (dict): Metrics summary from save_and_print_metrics_summary.
        algorithm_configs (list): List of algorithm configurations (from cfg.ALGORITHMS_TO_RUN)
                                  to get display names and potentially map to console colors.
    """
    if not all_metrics_summary:
        print("No metrics data available to display in table.")
        return

    print("\nDEBUG: print_metrics_table_to_console called.") # Debug print
    print(f"DEBUG: Received algorithm_configs: {algorithm_configs}") # Debug print

    headers = ["Algorithm", "Avg F1", "Avg FPS", "Avg Precision", "Avg Recall", "Avg TP", "Avg FP", "Avg FN", "Images"]
    
    col_widths = {header: len(header) for header in headers}
    table_data = []

    algo_display_name_to_console_color = {}
    
    # This map should contain all BGR colors used in your ALGORITHMS_TO_RUN config,
    # mapped to the desired Fore.COLOR.
    bgr_to_ansi_map_local = {
        (255,0,0): Fore.BLUE,          # OpenCV BGR Blue (e.g., for Hough Transform)
        (0,255,0): Fore.GREEN,        # OpenCV BGR Green (e.g., for Ground Truth, if ever in this table)
        (0,0,255): Fore.RED,           # OpenCV BGR Red
        (255,255,0): Fore.CYAN,       # OpenCV BGR Teal/Cyan (B=255, G=255, R=0)
        (255,0,255): Fore.MAGENTA,     # OpenCV BGR Magenta (B=255, G=0, R=255)
        (0,255,255): Fore.YELLOW,      # OpenCV BGR Yellow (B=0, G=255, R=255)
        (0,0,0): Fore.LIGHTBLACK_EX,   # Black
        (128,128,128): Fore.LIGHTBLACK_EX, # Gray
        (255,165,0): Fore.CYAN,        # Changed from Fore.LIGHTBLUE_EX for BGR (255,165,0) which is a Teal/Cyan
        # Add any other specific BGR colors from your config here
        # For example, if CarND Pipeline uses (192,192,192) for silver/gray:
        # (192,192,192): Fore.LIGHTBLACK_EX, 
    }
    print(f"DEBUG: bgr_to_ansi_map_local = {bgr_to_ansi_map_local}") # Debug print

    # Fallback map if BGR color is not in bgr_to_ansi_map_local
    CONSOLE_COLOR_MAP_local = {
        "Hough Transform": Fore.CYAN,       # Fallback if BGR (255,0,0) is not specified or matched
        "Adv. Sliding Window": Fore.YELLOW, # Fallback if BGR (0,255,255) is not specified or matched
        "CarND Pipeline": Fore.MAGENTA,
        "LaneNet": Fore.GREEN, # Note: LaneNet is usually excluded from optimizer
    }
    print(f"DEBUG: CONSOLE_COLOR_MAP_local = {CONSOLE_COLOR_MAP_local}") # Debug print
    DEFAULT_CONSOLE_COLOR_local = Fore.WHITE

    for algo_conf in algorithm_configs:
        if not algo_conf.get("active", False):
            continue
        display_name = algo_conf.get("display_name", algo_conf.get("module_name", "Unknown Algo"))
        bgr_color_from_conf = algo_conf.get("color") 
        
        print(f"DEBUG: Processing algo_conf: display_name='{display_name}', configured BGR color={bgr_color_from_conf}")

        current_algo_color = DEFAULT_CONSOLE_COLOR_local
        
        # Ensure bgr_color_from_conf is a tuple for dictionary key lookup
        processed_bgr_color_key = None
        if isinstance(bgr_color_from_conf, list):
            processed_bgr_color_key = tuple(bgr_color_from_conf)
        elif isinstance(bgr_color_from_conf, tuple):
            processed_bgr_color_key = bgr_color_from_conf

        # Priority 1: Use BGR color from config if it's in our map
        if processed_bgr_color_key and processed_bgr_color_key in bgr_to_ansi_map_local:
            current_algo_color = bgr_to_ansi_map_local[processed_bgr_color_key]
            print(f"DEBUG:   SUCCESS: Matched BGR {processed_bgr_color_key} to colorama color repr: {repr(current_algo_color)} for '{display_name}'")
        # Priority 2: Fallback to display name map
        elif display_name in CONSOLE_COLOR_MAP_local:
            current_algo_color = CONSOLE_COLOR_MAP_local[display_name]
            print(f"DEBUG:   FALLBACK: Matched display_name '{display_name}' to colorama color repr: {repr(current_algo_color)} from CONSOLE_COLOR_MAP_local")
        else:
            print(f"DEBUG:   NO MATCH: No BGR or display_name match for '{display_name}'. Using default colorama color repr: {repr(DEFAULT_CONSOLE_COLOR_local)}.")
            
        algo_display_name_to_console_color[display_name] = current_algo_color
    
    print(f"DEBUG: Final algo_display_name_to_console_color map = {algo_display_name_to_console_color}")

    for algo_name, metrics_list in all_metrics_summary.items():
        row = {"Algorithm": algo_name}
        # Retrieve the determined color for this algorithm
        # algo_name here comes from all_metrics_summary keys, should match display_name used above
        retrieved_color_code = algo_display_name_to_console_color.get(algo_name, DEFAULT_CONSOLE_COLOR_local)
        print(f"DEBUG: For algo_name '{algo_name}' (from metrics summary), retrieved console_color repr: {repr(retrieved_color_code)}")


        if metrics_list:
            # ... (metric calculation logic remains the same) ...
            def get_valid_metrics(metric_name):
                return [
                    m[metric_name] for m in metrics_list
                    if m and metric_name in m and isinstance(m[metric_name], (int, float)) and
                    (metric_name != "avg_x_diff_matched_points" or not np.isnan(m[metric_name]))
                ]

            f1_scores = get_valid_metrics("f1_score")
            precisions = get_valid_metrics("precision")
            recalls = get_valid_metrics("recall")
            tps = get_valid_metrics("true_positives (matched_gt_lanes)")
            fps_algo = get_valid_metrics("false_positives (unmatched_det_lanes)") # 'fps' here is False Positives
            fns = get_valid_metrics("false_negatives (unmatched_gt_lanes)")
            
            exec_times = [m['exec_time'] for m in metrics_list if m and 'exec_time' in m and isinstance(m['exec_time'], (int, float))]
            avg_exec_time = np.mean([t for t in exec_times if t > 0]) if any(t > 0 for t in exec_times) else 0
            avg_fps_val = 1.0 / avg_exec_time if avg_exec_time > 0 else 0

            row["Avg F1"] = f"{np.mean(f1_scores):.3f}" if f1_scores else "N/A"
            row["Avg FPS"] = f"{avg_fps_val:.2f}"
            row["Avg Precision"] = f"{np.mean(precisions):.3f}" if precisions else "N/A"
            row["Avg Recall"] = f"{np.mean(recalls):.3f}" if recalls else "N/A"
            row["Avg TP"] = f"{np.mean(tps):.2f}" if tps else "N/A"
            row["Avg FP"] = f"{np.mean(fps_algo):.2f}" if fps_algo else "N/A" 
            row["Avg FN"] = f"{np.mean(fns):.2f}" if fns else "N/A"
            row["Images"] = str(len(metrics_list))
        else:
            for header in headers[1:]: 
                row[header] = "N/A"
        
        table_data.append((row, retrieved_color_code)) # Use the retrieved color
        for header in headers:
            col_widths[header] = max(col_widths[header], len(str(row.get(header, ""))))

    # Print header
    header_str = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    # Use Fore.LIGHTWHITE_EX directly for header, as Colors class might not be in this scope if copy-pasting parts
    print("\n" + Fore.LIGHTWHITE_EX + header_str) # RESET is handled by autoreset=True
    print("-+-".join("-" * col_widths[h] for h in headers))

    # Print data rows
    for row_data, color_code_for_row in table_data:
        row_str = " | ".join(f"{str(row_data.get(h, '')):<{col_widths[h]}}" for h in headers)
        print(color_code_for_row + row_str) # RESET is handled by autoreset=True
    print("\n")

# Make sure to call print_metrics_table_to_console from run_comparison.py
# after all_metrics_summary and cfg.ALGORITHMS_TO_RUN are available.
# Example call in run_comparison.py, near the end of main_comparison_orchestrator:
#
# if images_processed_count > 0 and all_metrics_summary:
#     ...
#     lane_utils.save_and_print_metrics_summary(...)
#     lane_utils.print_metrics_table_to_console(all_metrics_summary, cfg.ALGORITHMS_TO_RUN) # New call
# ...
