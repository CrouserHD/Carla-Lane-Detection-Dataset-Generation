import json
import cv2
import os
import numpy as np
import glob

# Configuration
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

JSON_FILE_PATH = os.path.join(WORKSPACE_ROOT, "data", "dataset", "Town03_Opt", "train_gt_tmp.json")
OUTPUT_IMAGE_DIR = os.path.join(WORKSPACE_ROOT, "data", "debug", "Town03_Opt_lanes_visualization")
OUTPUT_VIDEO_PATH_CONFIG = os.path.join(WORKSPACE_ROOT, "data", "Town03_Opt_lanes_visualization.mp4") # Renamed to avoid conflict
VIDEO_FPS = 20
MAX_FRAMES_TO_PROCESS = 1000  # Set to None or 0 to process all frames
DISPLAY_LANE_INDICES = None  #!!!BUGGY!!!  e.g., [0, 1] to display only the first two lanes, or None for all

# Lane colors (BGR format)
LANE_COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (255, 255, 0)  # Cyan
]

# Attempt to import natsort, provide fallback
try:
    from natsort import natsorted
except ImportError:
    print("Warning: 'natsort' library is not installed. Image sequence in video might not be in natural order.")
    print("Please install it using: pip install natsort")
    def natsorted(l, alg=None): # Add alg=None to match natsort signature for this simple fallback
        return sorted(l)

def draw_lanes_on_image(image_path, lanes_data, h_samples_data):
    """Loads an image and draws specified lanes on it, using h_samples for y-coordinates."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    if not h_samples_data:
        print(f"Warning: h_samples_data is missing for {image_path}. Cannot draw lanes.")
        return img # Return original image as is

    lanes_to_draw = []
    if DISPLAY_LANE_INDICES is None or not DISPLAY_LANE_INDICES:
        lanes_to_draw = list(enumerate(lanes_data))
    else:
        for index in DISPLAY_LANE_INDICES:
            if 0 <= index < len(lanes_data):
                lanes_to_draw.append((index, lanes_data[index])) # Keep original index for color consistency if needed
            else:
                print(f"Warning: Lane index {index} is out of bounds for image {image_path}. Max index is {len(lanes_data)-1}. Skipping this index.")

    for original_lane_idx, lane_x_coords in lanes_to_draw: # Use original_lane_idx for consistent coloring if desired
        color = LANE_COLORS[original_lane_idx % len(LANE_COLORS)] # Color based on original index
        
        current_lane_points = []
        if len(lane_x_coords) != len(h_samples_data):
            print(f"Warning: Mismatch between number of x_coords ({len(lane_x_coords)}) and h_samples ({len(h_samples_data)}) for lane {original_lane_idx} in {image_path}. Skipping this lane.")
            continue

        for x_coord, y_coord in zip(lane_x_coords, h_samples_data):
            if x_coord >= 0:  # Valid x-coordinate (CULane uses -2 for invalid/non-existent points)
                current_lane_points.append((int(x_coord), int(y_coord)))
        
        for j in range(len(current_lane_points) - 1):
            p1 = current_lane_points[j]
            p2 = current_lane_points[j+1]
            cv2.line(img, p1, p2, color, thickness=2)
            
    return img

def create_video_from_images(image_folder, video_path_param, fps):
    """Creates a video from images in a folder."""
    # Use the natsorted function (either original or fallback)
    images = natsorted([img for img in glob.glob(os.path.join(image_folder, "*.jpg")) + 
                        glob.glob(os.path.join(image_folder, "*.png"))]) # Include png if any
    
    if not images:
        print(f"No images found in {image_folder} to create video.")
        return None

    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Could not read the first image: {images[0]}")
        return None

    height, width, layers = frame.shape
    
    # Attempt MP4V codec first
    fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path_param, fourcc_mp4v, fps, (width, height))
    
    final_video_path = video_path_param

    if not video.isOpened():
        print(f"Warning: Could not open video writer for path {video_path_param} with codec 'mp4v'.")
        print("Trying with XVID codec and .avi extension.")
        video_path_avi = os.path.splitext(video_path_param)[0] + ".avi"
        fourcc_xvid = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_path_avi, fourcc_xvid, fps, (width, height))
        if not video.isOpened():
            print(f"Error: Could not open video writer for path {video_path_avi} with codec XVID either.")
            return None
        else:
            print(f"Successfully opened video writer with XVID for {video_path_avi}. Output will be an AVI file.")
            final_video_path = video_path_avi
    else:
         print(f"Successfully opened video writer with MP4V for {video_path_param}.")


    for image_file_path in images: # Renamed to avoid conflict
        img = cv2.imread(image_file_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Warning: Skipping image {image_file_path} during video creation as it could not be read.")

    video.release()
    cv2.destroyAllWindows() # Good practice
    print(f"Video saved to {final_video_path}")
    return final_video_path


def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"Created directory: {OUTPUT_IMAGE_DIR}")

    try:
        with open(JSON_FILE_PATH, 'r') as f:
            # Assuming each line is a separate JSON object
            lane_data_list = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: JSON file not found at {JSON_FILE_PATH}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {JSON_FILE_PATH}. Details: {e}")
        return

    if not lane_data_list:
        print(f"No data found or parsed in {JSON_FILE_PATH}")
        return

    # Determine the number of entries to process
    num_entries_to_process = len(lane_data_list)
    if MAX_FRAMES_TO_PROCESS is not None and MAX_FRAMES_TO_PROCESS > 0:
        num_entries_to_process = min(len(lane_data_list), MAX_FRAMES_TO_PROCESS)
        print(f"Processing a maximum of {num_entries_to_process} frames based on MAX_FRAMES_TO_PROCESS setting.")
    else:
        print(f"Processing all {num_entries_to_process} entries from {JSON_FILE_PATH}...")

    processed_image_count = 0
    # Iterate only up to num_entries_to_process
    for i, entry in enumerate(lane_data_list[:num_entries_to_process]):
        image_path_from_json = entry.get("raw_file")
        lanes = entry.get("lanes")
        h_samples = entry.get("h_samples") # Get h_samples

        if not image_path_from_json:
            print(f"Warning: 'raw_file' not found in entry {i}. Skipping.")
            continue
        
        base_filename = os.path.basename(image_path_from_json)
        image_path = os.path.join(WORKSPACE_ROOT, "data", "debug", "Town03_Opt", base_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image path '{image_path}' (derived from '{image_path_from_json}') not found for entry {i}. Skipping.")
            continue

        output_image_path = os.path.join(OUTPUT_IMAGE_DIR, base_filename)

        if not lanes: 
            print(f"Info: No lanes to draw for image {base_filename} (entry {i}). Copying original.")
            try:
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    cv2.imwrite(output_image_path, original_img)
                    processed_image_count += 1
                else:
                    print(f"Warning: Could not read image {image_path} for copying. Skipping.")
            except Exception as e:
                print(f"Error copying image {image_path}: {e}")
            continue 

        if not h_samples:
            print(f"Warning: 'h_samples' not found for entry {i} (image: {base_filename}) though 'lanes' are present. Copying original image.")
            try:
                original_img = cv2.imread(image_path)
                if original_img is not None:
                    cv2.imwrite(output_image_path, original_img)
                    processed_image_count += 1
                else:
                    print(f"Warning: Could not read image {image_path} for copying. Skipping.")
            except Exception as e:
                print(f"Error copying image {image_path}: {e}")
            continue

        img_with_lanes = draw_lanes_on_image(image_path, lanes, h_samples)
        
        if img_with_lanes is None: # Should not happen if image_path is valid, but good check
            print(f"Warning: Failed to process image {image_path}. Skipping.")
            continue

        cv2.imwrite(output_image_path, img_with_lanes)
        processed_image_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_entries_to_process} images...")

    print(f"Finished processing images. {processed_image_count} images saved to {OUTPUT_IMAGE_DIR}")

    if processed_image_count > 0:
        print(f"Creating video from images in {OUTPUT_IMAGE_DIR}...")
        created_video_path = create_video_from_images(OUTPUT_IMAGE_DIR, OUTPUT_VIDEO_PATH_CONFIG, VIDEO_FPS)
        if created_video_path:
            print(f"Video creation process finished. Final video at: {created_video_path}")
        else:
            print("Video creation failed.")
    else:
        print("No images were processed, skipping video creation.")

if __name__ == "__main__":
    main()
