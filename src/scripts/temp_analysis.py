import re

# Path to the metrics file
METRICS_FILE_PATH = r"C:\Users\John\Desktop\Masterarbeit_Carla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\Masterarbeit_Lane_Detection\Carla-Lane-Detection-Dataset-Generation\data\comparison_results_modular\metrics_summary.txt"

best_f1_score = -1.0
best_algo_name = ""
current_algo_name = ""

# To store all details of the best algorithm
best_algo_details = {}
# Temporary storage for current algorithm's metrics before confirming it's the best
current_algo_metrics_buffer = {}

try:
    with open(METRICS_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Algorithm:"):
                # Finalize previous algorithm's metrics if it was the best one so far
                if current_algo_name == best_algo_name and best_algo_name != "":
                    for k, v in current_algo_metrics_buffer.items():
                        best_algo_details[k] = v
                
                current_algo_name = line.replace("Algorithm:", "").strip()
                current_algo_metrics_buffer = {} # Reset buffer for new algorithm

            elif line.startswith("Avg F1 Score:") and current_algo_name:
                try:
                    f1_score_str = line.replace("Avg F1 Score:", "").strip()
                    f1_score = float(f1_score_str)
                    current_algo_metrics_buffer["Avg F1 Score"] = f1_score_str # Store as string initially
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_algo_name = current_algo_name
                        # Update best_algo_details basics, detailed metrics will be added from buffer
                        best_algo_details = {"name": best_algo_name, "f1_score_float": best_f1_score}
                except ValueError:
                    pass 
            elif current_algo_name and ":" in line and "Algorithm:" not in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    metric_value = parts[1].strip()
                    current_algo_metrics_buffer[metric_name] = metric_value
        
        # After loop, check if the last processed algorithm was the best
        if current_algo_name == best_algo_name and best_algo_name != "":
            for k, v in current_algo_metrics_buffer.items():
                 # Ensure not to overwrite the float f1_score with string if already set
                if k == "Avg F1 Score" and "f1_score_float" in best_algo_details:
                    best_algo_details[k] = best_algo_details["f1_score_float"] # keep float
                else:
                    best_algo_details[k] = v


    if best_algo_name:
        print(f"Best Algorithm: {best_algo_name}")
        # Ensure f1_score_float is used for printing if available
        f1_to_print = best_algo_details.get("f1_score_float", best_f1_score)
        print(f"Best Avg F1 Score: {f1_to_print:.3f}")

        param_part_match = re.search(r'\((.*?)\)', best_algo_name) # More general regex for content in parentheses
        parsed_parameters = {}

        if param_part_match:
            params_str = param_part_match.group(1) # Content within parentheses, e.g., "ASWSet1_NW7_M60_MP30" or "Set1_R2_T180_TH50_ML50_MG100_CL50_CH150"
            
            print(f"Attempting to parse parameters from: {params_str}") # Debug print

            if "Adv. Sliding Window" in best_algo_name or params_str.startswith("ASWSet"):
                # Parsing for Advanced Sliding Window: e.g., ASWSet1_NW7_M60_MP30
                parts = params_str.split("_")
                try:
                    if len(parts) >= 4 and parts[0].startswith("ASWSet"):
                        parsed_parameters["ASW_SET_NUMBER"] = parts[0].replace("ASWSet", "")
                        parsed_parameters["ASW_NWINDOWS"] = int(parts[1].replace("NW", ""))
                        parsed_parameters["ASW_MARGIN"] = int(parts[2].replace("M", ""))
                        parsed_parameters["ASW_MINPIX"] = int(parts[3].replace("MP", ""))
                        # Add more ASW parameters if they appear in the name string
                    else:
                        print(f"Could not parse ASW parameters from: {params_str}. Expected format like ASWSetX_NWY_MZ_MPW.")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing ASW parameters from name: {best_algo_name} (substring: {params_str}) -> {e}")

            elif "Hough Transform" in best_algo_name or params_str.startswith("Set"): # Assuming old Hough sets start with "Set"
                # Parsing for Hough Transform: e.g., Set1_R2_T180_TH50_ML50_MG100_CL50_CH150
                parts = params_str.split("_")
                try:
                    # Keep your existing Hough parsing logic here
                    # Example: (ensure indices and prefixes match your actual naming convention)
                    if len(parts) >= 8 and parts[0].startswith("Set"):
                        parsed_parameters["HOUGH_SET_NUMBER"] = parts[0].replace("Set", "")
                        parsed_parameters["HOUGH_RHO_ACCURACY"] = int(parts[1][1:]) # Assuming R followed by number
                        theta_deg_key_str = parts[2][1:] # Assuming T followed by number
                        theta_map_to_str = {
                            "180": "1 (corresponds to np.pi / 180)", # Assuming the number is the divisor
                            "90": "2 (corresponds to np.pi / 90)", 
                            "60": "3 (corresponds to np.pi / 60)"
                        }
                        parsed_parameters["HOUGH_THETA_ACCURACY_DEG_NAME"] = theta_map_to_str.get(theta_deg_key_str, f"Unknown T{theta_deg_key_str}")
                        parsed_parameters["HOUGH_THRESHOLD"] = int(parts[3][2:]) # Assuming TH followed by number
                        parsed_parameters["HOUGH_MIN_LINE_LENGTH"] = int(parts[4][2:]) # Assuming ML followed by number
                        parsed_parameters["HOUGH_MAX_LINE_GAP"] = int(parts[5][2:]) # Assuming MG followed by number
                        parsed_parameters["HOUGH_CANNY_LOW_THRESHOLD"] = int(parts[6][2:]) # Assuming CL followed by number
                        parsed_parameters["HOUGH_CANNY_HIGH_THRESHOLD"] = int(parts[7][2:]) # Assuming CH followed by number
                    else:
                        print(f"Could not parse Hough parameters from: {params_str}. Expected format like SetX_RY_TZ_THV_MLV_MGV_CLV_CHV.")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing Hough parameters from name: {best_algo_name} (substring: {params_str}) -> {e}")
            
            elif "Color Segmentation" in best_algo_name or params_str.startswith("CSSet"): # Parameter parsing for Color Segmentation
                # Example from output: CSSet37_A20_S0.3_L100_AR0.05
                # Order: SetNum, Area, Solidity, L-component, AspectRatio
                parts = params_str.split("_")
                try:
                    if len(parts) >= 5 and parts[0].startswith("CSSet"):
                        parsed_parameters["CS_SET_NUMBER"] = parts[0].replace("CSSet", "")
                        parsed_parameters["CS_CONTOUR_MIN_AREA"] = int(parts[1].replace("A", ""))
                        parsed_parameters["CS_MIN_SOLIDITY"] = float(parts[2].replace("S", ""))
                        parsed_parameters["CS_LOWER_WHITE_L_COMPONENT"] = int(parts[3].replace("L", ""))
                        parsed_parameters["CS_MIN_ASPECT_RATIO"] = float(parts[4].replace("AR", ""))
                    else:
                        print(f"Could not parse Color Segmentation parameters from: {params_str}. Expected format like CSSetX_AY_SZ_LV_ARW.")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing Color Segmentation parameters from name: {best_algo_name} (substring: {params_str}) -> {e}")

            else:
                print(f"Unrecognized parameter string format or algorithm type for: {params_str}")

            if parsed_parameters:
                print("\nExtracted Parameters:")
                for key, value in parsed_parameters.items():
                    print(f"  {key}: {value}")
                best_algo_details["parsed_parameters"] = parsed_parameters
            else:
                print(f"No parameters were successfully parsed for {best_algo_name}.")
        else:
            print("Could not parse parameter string from algorithm name.")
            
        print("\nFull details of the best performing set:")
        for key, value in best_algo_details.items():
            if key not in ["parsed_parameters", "name", "f1_score_float", "Avg F1 Score"]:
                 print(f"  {key}: {value}")
            elif key == "Avg F1 Score" and "f1_score_float" not in best_algo_details: # Print string F1 if float not set
                 print(f"  {key}: {value}")


    else:
        print("No F1 scores found or file was empty/unreadable.")

except FileNotFoundError:
    print(f"Error: File not found at {METRICS_FILE_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")
