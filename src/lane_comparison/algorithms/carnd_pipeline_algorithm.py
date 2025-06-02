import numpy as np
import cv2
import sys
import os

# --- Helper Functions ---
def combine_color(img_bgr, rgb_thresh_override=None, hls_thresh_override=None):
  """
  Compute binary grayscaled image that captures the lane lines

  Input
  -----
  img_bgr : image in BGR form
  rgb_thresh_override : Optional tuple to override default RGB thresholds
  hls_thresh_override : Optional tuple to override default HLS thresholds

  Output
  -----
  A binary grayscaled image
  """
  # Default thresholds, can be overridden
  rgb_thresh = rgb_thresh_override if rgb_thresh_override is not None else (193, 255) # Made more lenient
  hls_thresh = hls_thresh_override if hls_thresh_override is not None else (195, 255)   # Made more lenient

  # --- BEGIN ADDED DEBUG LOGGING ---
  #print(f"[DEBUG carnd_pipeline_algorithm.combine_color] Using RGB Thresh: {rgb_thresh}, HLS Thresh: {hls_thresh}")
  # --- END ADDED DEBUG LOGGING ---

  img_r = img_bgr[:, :, 2]
  binary_r = np.zeros_like(img_r)
  binary_r[(img_r > rgb_thresh[0]) & (img_r <= rgb_thresh[1])] = 1

  hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
  S = hls[:, :, 2]
  L = hls[:, :, 1]
  binary_s = np.zeros_like(S)
  binary_l = np.zeros_like(L)
  binary_s[(S > hls_thresh[0]) & (S <= hls_thresh[1])] = 1
  binary_l[(L > hls_thresh[0]) & (L <= hls_thresh[1])] = 1

  combined = np.zeros_like(img_r)
  # Use OR logic for broader detection from color channels
  combined[(binary_r == 1) | (binary_s == 1) | (binary_l == 1)] = 1

  return combined

def find_lane_line_pixels(image, window_width, window_height, margin):
  """
  Use convolution to find window centroid. Then the lane line pixels

  Inputs
  -----
  image : presumably it is a binary grayscaled image, with elements of only 0 or 1

  window_width : covolution window width of your choice

  window_height : covolution widnow height of your choice

  margin : the horizontal offset from the window centroids we use to draw a bounding box
           at each level to find lane line pixels. Dont confuse it with covolution window width

  Outputs
  -----
  a 4-element tuple containing x coordinates and y coordinates for left and right lane lines
  """
  window_centroids = [] # Store the (left,right) window centroid positions per level
  window = np.ones(window_width) # Create our window template that we will use for convolutions

  # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
  # and then np.convolve the vertical image slice with the window template

  # Sum quarter bottom of image to get slice, could use a different ratio
  # image is a grayscale, looking at lower left quarter
  l_sum = np.sum(image[int(image.shape[0]/2):,:int(image.shape[1]/2)], axis=0)
  l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
  # the lower right quarter
  r_sum = np.sum(image[int(image.shape[0]/2):,int(image.shape[1]/2):], axis=0)
  # the convolution starts from index 0, so we shift it by half of the width
  r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

  # note l_center and r_center are x coordinates in the image

  # Add what we found for the first layer
  # this is our first window
  window_centroids.append((l_center, r_center))

  # still we need to collect all the nonzero pixels in this window
  # so later we can fit a polynomial
  # here the window width used to collect pixels is 2 * margin
  nonzero = image.nonzero()
  nonzeroy = nonzero[0]
  nonzerox = nonzero[1]

  left_lane_inds = []
  right_lane_inds = []

  win_y_low = image.shape[0] - window_height
  win_y_high = image.shape[0]

  win_x_l_low = l_center - margin
  win_x_l_high = l_center + margin
  win_x_r_low = r_center - margin
  win_x_r_high = r_center + margin

  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

  left_lane_inds.append(good_left_inds)
  right_lane_inds.append(good_right_inds)

  # Go through each layer looking for max pixel locations
  for level in range(1,(int)(image.shape[0]/window_height)):
    # convolve the window into the vertical slice of the image
    # in the loop we go through the entire width
    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    conv_signal = np.convolve(window, image_layer)
    # Find the best left centroid by using past left center as a reference
    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    offset = window_width/2
    # to avoid negative index, use max()
    # it is the index in conv_signal
    l_min_index = int(max(l_center+offset-margin,0))
    # to avoid index larger than width, use min()
    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
    # get the index in original image
    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    # Find the best right centroid by using past right center as a reference
    r_min_index = int(max(r_center+offset-margin,0))
    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    # Add what we found for that layer
    window_centroids.append((l_center,r_center))

    win_y_low = image.shape[0] - (level + 1) * window_height
    win_y_high = image.shape[0] - level * window_height

    win_x_l_low = l_center - margin
    win_x_l_high = l_center + margin
    win_x_r_low = r_center - margin
    win_x_r_high = r_center + margin

    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_x_l_low) & (nonzerox < win_x_l_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_x_r_low) & (nonzerox < win_x_r_high)).nonzero()[0]

    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)
  # centroids we get is the x coordinates for n windows
  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  # Generate x and y values for plotting
  ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  return leftx, lefty, rightx, righty

# --- Line Class (copied from CarND pipeline.py) ---
class Line():
  def __init__(self):
    self.detected = False
    self.recent_xfitted = []
    self.bestx = None
    self.best_fit = None
    self.current_fit = [np.array([False])] # Must be a list containing an array for consistency
    self.radius_of_curvature = None
    self.line_base_pos = None
    self.diffs = np.array([0, 0, 0], dtype='float')
    self.allx = None
    self.ally = None

# --- Global variables ---
perspective_transform_cache = {
    "matP": None,
    "matP_inv": None,
    "img_size": None
}
left_line_global = Line()
right_line_global = Line()

# --- Helper Functions ---
def get_perspective_mat_adapted(img_size, src_ratios_tl_tr_bl_br, dst_ratios_tl_tr_bl_br):
    global perspective_transform_cache
    # img_size is (width, height)
    # Ratios are [ [x_ratio, y_ratio], ... ]
    # Input order for ratios: Top-Left, Top-Right, Bottom-Left, Bottom-Right

    if perspective_transform_cache["matP"] is not None and \
       perspective_transform_cache["matP_inv"] is not None and \
       perspective_transform_cache["img_size"] == img_size:
        return perspective_transform_cache["matP"], perspective_transform_cache["matP_inv"]

    w, h = img_size[0], img_size[1]

    # cv2.getPerspectiveTransform expects points in order: TL, TR, BR, BL
    src_cv2_ordered = np.float32([
        [w * src_ratios_tl_tr_bl_br[0][0], h * src_ratios_tl_tr_bl_br[0][1]], # Top-Left
        [w * src_ratios_tl_tr_bl_br[1][0], h * src_ratios_tl_tr_bl_br[1][1]], # Top-Right
        [w * src_ratios_tl_tr_bl_br[3][0], h * src_ratios_tl_tr_bl_br[3][1]], # Bottom-Right (index 3)
        [w * src_ratios_tl_tr_bl_br[2][0], h * src_ratios_tl_tr_bl_br[2][1]]  # Bottom-Left (index 2)
    ])
    dst_cv2_ordered = np.float32([
        [w * dst_ratios_tl_tr_bl_br[0][0], h * dst_ratios_tl_tr_bl_br[0][1]], # Top-Left
        [w * dst_ratios_tl_tr_bl_br[1][0], h * dst_ratios_tl_tr_bl_br[1][1]], # Top-Right
        [w * dst_ratios_tl_tr_bl_br[3][0], h * dst_ratios_tl_tr_bl_br[3][1]], # Bottom-Right (index 3)
        [w * dst_ratios_tl_tr_bl_br[2][0], h * dst_ratios_tl_tr_bl_br[2][1]]  # Bottom-Left (index 2)
    ])

    matP = cv2.getPerspectiveTransform(src_cv2_ordered, dst_cv2_ordered)
    matP_inv = cv2.getPerspectiveTransform(dst_cv2_ordered, src_cv2_ordered)

    perspective_transform_cache["matP"] = matP
    perspective_transform_cache["matP_inv"] = matP_inv
    perspective_transform_cache["img_size"] = img_size
    return matP, matP_inv

def get_bin_lane_line_img(img_bgr, rgb_thresh_override=None, hls_thresh_override=None):
  return combine_color(img_bgr, rgb_thresh_override=rgb_thresh_override, hls_thresh_override=hls_thresh_override)

def update_line(line, fitx, fit_coeffs):
  line.detected = True
  max_frames_to_average = 10 # Number of frames to average over for smoothing

  # Update recent_xfitted
  if len(line.recent_xfitted) == max_frames_to_average:
    line.recent_xfitted.pop(0)
  line.recent_xfitted.append(fitx)
  line.bestx = np.mean(line.recent_xfitted, axis=0)

  # Update best_fit (polynomial coefficients)
  if line.best_fit is None or not isinstance(line.current_fit[0], np.ndarray): # First valid fit or reset
    line.best_fit = fit_coeffs
  else:
    # Simple moving average for coefficients
    alpha = 1.0 / min(len(line.recent_xfitted), max_frames_to_average)
    line.best_fit = line.best_fit * (1 - alpha) + fit_coeffs * alpha
  
  if isinstance(line.current_fit[0], np.ndarray): # Check if current_fit is valid
      line.diffs = fit_coeffs - line.current_fit[0]
  else:
      line.diffs = np.array([0,0,0], dtype='float') # Reset diffs if current_fit was invalid
  line.current_fit[0] = fit_coeffs


# --- Main Detection Function ---
def detect_lanes_carnd(image, roi_abs_vertices, perspective_src_ratios, perspective_dst_ratios, param_overrides=None):
    """
    Detects lanes in an image using an adapted CarND pipeline.
    Args:
        image: Input BGR image. This image is expected to be at the processing scale (potentially downscaled by the caller).
        roi_abs_vertices: Numpy array of absolute [x,y] pixel coordinates defining the ROI polygon,
                          corresponding to the dimensions of the input `image`.
        perspective_src_ratios: Source points for perspective transform (ratios relative to `image` dimensions).
        perspective_dst_ratios: Destination points for perspective transform (ratios relative to `image` dimensions).
        param_overrides: Optional dictionary for overriding internal parameters (e.g., for sweeps).
                           Expected keys: "CARND_RGB_THRESH", "CARND_HLS_THRESH"
    Returns:
        A list of detected lane lines [[(x,y),...], [(x,y),...]], with coordinates relative to the input `image` dimensions.
    """
    global left_line_global, right_line_global

    # Re-initialize Line objects for each call to ensure independent processing for parameter sweeps
    left_line_global = Line()
    right_line_global = Line()

    # Extract potential overrides for color thresholds
    rgb_thresh_override = None
    hls_thresh_override = None
    if param_overrides:
        rgb_thresh_override = param_overrides.get("CARND_RGB_THRESH")
        hls_thresh_override = param_overrides.get("CARND_HLS_THRESH")

    # img_bgr = image.copy() # Input 'image' is used directly as it's at the processing scale.
    img_height, img_width = image.shape[0], image.shape[1] # Dimensions of the (potentially scaled) input image
    img_size = (img_width, img_height)

    # 1. Apply ROI
    mask = np.zeros_like(image[:,:,0]) # Use shape of input 'image'
    # roi_abs_vertices are already absolute and correspond to the input 'image' dimensions.
    cv2.fillPoly(mask, [np.array(roi_abs_vertices, dtype=np.int32)], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask) # Apply to input 'image'

    # --- BEGIN ADDED DEBUG for ROI Visualization ---
    # ROI debug image should be based on the input 'image' which is at processing scale
    img_with_roi_drawn = image.copy()
    cv2.polylines(img_with_roi_drawn, [np.array(roi_abs_vertices, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2) # Draw ROI in Green

    debug_roi_filename = "debug_carnd_roi_visualization_default.png"
    if param_overrides:
        rgb_thresh_val = param_overrides.get('CARND_RGB_THRESH')
        hls_thresh_val = param_overrides.get('CARND_HLS_THRESH')
        
        rgb_thresh_str = f"{rgb_thresh_val[0]}_{rgb_thresh_val[1]}" if rgb_thresh_val else "defaultRGB"
        hls_thresh_str = f"{hls_thresh_val[0]}_{hls_thresh_val[1]}" if hls_thresh_val else "defaultHLS"
        
        rgb_thresh_str = rgb_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        hls_thresh_str = hls_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        
        debug_roi_filename = f"debug_carnd_roi_visualization_RGB_{rgb_thresh_str}_HLS_{hls_thresh_str}.png"
    
    cv2.imwrite(debug_roi_filename, img_with_roi_drawn)
    # --- END ADDED DEBUG for ROI Visualization ---
    
    # Input 'image' is assumed to be the one for processing (potentially scaled, already undistorted if needed by caller)
    processed_image = masked_image # This is the ROI-masked version of the input 'image'

    # 2. Get binary image, passing potential overrides
    combined_binary = get_bin_lane_line_img(processed_image, 
                                            rgb_thresh_override=rgb_thresh_override, 
                                            hls_thresh_override=hls_thresh_override)

    # --- BEGIN ADDED DEBUG for combined_binary (PRE-WARP) ---
    debug_combined_filename = "debug_combined_binary_PRE_WARP_default.png"
    if param_overrides:
        rgb_thresh_val = param_overrides.get('CARND_RGB_THRESH')
        hls_thresh_val = param_overrides.get('CARND_HLS_THRESH')
        
        rgb_thresh_str = f"{rgb_thresh_val[0]}_{rgb_thresh_val[1]}" if rgb_thresh_val else "defaultRGB"
        hls_thresh_str = f"{hls_thresh_val[0]}_{hls_thresh_val[1]}" if hls_thresh_val else "defaultHLS"
        
        # Sanitize strings for filename
        rgb_thresh_str = rgb_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        hls_thresh_str = hls_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        
        debug_combined_filename = f"debug_combined_binary_PRE_WARP_RGB_{rgb_thresh_str}_HLS_{hls_thresh_str}.png"
    
    # Ensure the output directory from config is used or default to current dir
    # This assumes OUTPUT_VIS_DIR is accessible or files are saved to script's dir
    # For simplicity, let's assume files are saved in the current working directory
    # or where run_comparison.py is executed from, which seems to be the current behavior.
    cv2.imwrite(debug_combined_filename, combined_binary * 255)
    # --- END ADDED DEBUG for combined_binary (PRE-WARP) ---

    # 3. Perspective Transform
    matP, matP_inv = get_perspective_mat_adapted(img_size, perspective_src_ratios, perspective_dst_ratios)
    warped_binary = cv2.warpPerspective(combined_binary, matP, img_size, flags=cv2.INTER_LINEAR)

    # --- BEGIN MODIFICATION: Ensure warped_binary is strictly binary ---
    # Threshold the warped image to ensure it's strictly 0s and 1s.
    # INTER_LINEAR can produce float values.
    warped_binary_thresholded = (warped_binary > 0.5).astype(np.uint8)
    # --- END MODIFICATION ---

    # --- DEBUG: Save the warped binary image for tuning perspective transform ---
    # This image is critical for adjusting CARND_SRC_RATIOS and CARND_DST_RATIOS.
    # Look for 'debug_warped_binary.png' in your execution directory.
    debug_filename = "debug_warped_binary.png"
    if param_overrides:
        rgb_thresh_val = param_overrides.get('CARND_RGB_THRESH')
        hls_thresh_val = param_overrides.get('CARND_HLS_THRESH')
        
        rgb_thresh_str = f"{rgb_thresh_val[0]}_{rgb_thresh_val[1]}" if rgb_thresh_val else "defaultRGB"
        hls_thresh_str = f"{hls_thresh_val[0]}_{hls_thresh_val[1]}" if hls_thresh_val else "defaultHLS"
        
        # Sanitize strings for filename - keep it simple
        rgb_thresh_str = rgb_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        hls_thresh_str = hls_thresh_str.replace("(", "").replace(")", "").replace(", ", "_")
        
        debug_filename = f"debug_warped_binary_RGB_{rgb_thresh_str}_HLS_{hls_thresh_str}.png"
    
    cv2.imwrite(debug_filename, warped_binary_thresholded * 255) # Save the thresholded version
    # --- END DEBUG ---

    # 4. Find lane line pixels and fit polynomials
    window_width = 50 
    window_height = warped_binary_thresholded.shape[0] // 9 # Use thresholded image shape
    margin = 100 

    try:
        leftx, lefty, rightx, righty = find_lane_line_pixels(warped_binary_thresholded, window_width, window_height, margin) # Use thresholded image
        # --- BEGIN ADDED DEBUG LOGGING ---
        """
        print(f"[DEBUG carnd_pipeline_algorithm.detect_lanes_carnd] find_lane_line_pixels output:")
        print(f"  leftx shape: {leftx.shape}, lefty shape: {lefty.shape}")
        print(f"  rightx shape: {rightx.shape}, righty shape: {righty.shape}")
        if leftx.size > 0: print(f"  leftx[:5]: {leftx[:5]}")
        if lefty.size > 0: print(f"  lefty[:5]: {lefty[:5]}")
        if rightx.size > 0: print(f"  rightx[:5]: {rightx[:5]}")
        if righty.size > 0: print(f"  righty[:5]: {righty[:5]}")
        """
        # --- END ADDED DEBUG LOGGING ---
    except Exception as e:
        left_line_global = Line() # Reset on error
        right_line_global = Line()
        return [] 

    if leftx.size == 0 or lefty.size == 0 or rightx.size == 0 or righty.size == 0:
        left_line_global.detected = False # Mark as not detected for next frame
        right_line_global.detected = False
        return [] 

    ploty = np.linspace(0, warped_binary_thresholded.shape[0] - 1, warped_binary_thresholded.shape[0]) # Use thresholded image shape

    try:
        left_fit_coeffs = np.polyfit(lefty, leftx, 2)
        right_fit_coeffs = np.polyfit(righty, rightx, 2)
        # --- BEGIN ADDED DEBUG LOGGING ---
        """
        print(f"[DEBUG carnd_pipeline_algorithm.detect_lanes_carnd] np.polyfit output:")
        print(f"  left_fit_coeffs: {left_fit_coeffs}")
        print(f"  right_fit_coeffs: {right_fit_coeffs}")
        """
        # --- END ADDED DEBUG LOGGING ---
    except (np.linalg.LinAlgError, TypeError) as e:
        left_line_global.detected = False
        right_line_global.detected = False
        return []

    # Generate x values from the fitted polynomials
    left_fitx = left_fit_coeffs[0] * ploty**2 + left_fit_coeffs[1] * ploty + left_fit_coeffs[2]
    right_fitx = right_fit_coeffs[0] * ploty**2 + right_fit_coeffs[1] * ploty + right_fit_coeffs[2]

    update_line(left_line_global, left_fitx, left_fit_coeffs)
    update_line(right_line_global, right_fitx, right_fit_coeffs)

    # 5. Unwarp detected lane lines (using smoothed bestx) for output
    if left_line_global.bestx is None or right_line_global.bestx is None:
        return []
        
    # Create points for the left and right lanes in the warped image space using smoothed lines
    # These points are relative to the (potentially scaled) warped image dimensions
    pts_left_warped = np.array([np.transpose(np.vstack([left_line_global.bestx, ploty]))], dtype=np.float32)
    pts_right_warped = np.array([np.transpose(np.vstack([right_line_global.bestx, ploty]))], dtype=np.float32)

    # Unwarp these points back to the perspective of the input 'image' (which is potentially scaled)
    unwarped_pts_left = cv2.perspectiveTransform(pts_left_warped, matP_inv)
    unwarped_pts_right = cv2.perspectiveTransform(pts_right_warped, matP_inv)

    detected_lanes_at_processing_scale = []
    if unwarped_pts_left is not None and len(unwarped_pts_left[0]) > 1:
        detected_lanes_at_processing_scale.append(unwarped_pts_left[0].astype(np.int32).tolist())
    if unwarped_pts_right is not None and len(unwarped_pts_right[0]) > 1:
        detected_lanes_at_processing_scale.append(unwarped_pts_right[0].astype(np.int32).tolist())
        
    return detected_lanes_at_processing_scale # Return lanes at the scale of the input 'image'

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running carnd_pipeline_algorithm.py as standalone test...")

    # Attempt to import ROI and Perspective parameters from the project's config file
    try:
        # Adjust path to import from lane_comparison_config.py
        # This script is in src/lane_comparison/algorithms/
        # Config is in src/lane_comparison/
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from lane_comparison_config import (
            ROI_Y_RATIO, ROI_X_START_RATIO, ROI_X_END_RATIO, ROI_Y_END_RATIO,
            ROI_BOTTOM_WIDTH_FACTOR_OFFSET, ASW_SRC_RATIOS, ASW_DST_RATIOS
        )
        print("Successfully imported ROI and Perspective parameters from lane_comparison_config.py")

        # Define example_roi_ratios using imported config values
        # Order for fillPoly: TL, TR, BR, BL
        # Config provides:
        # Top-left Y: ROI_Y_RATIO
        # Top-left X: ROI_X_START_RATIO
        # Top-right Y: ROI_Y_RATIO
        # Top-right X: ROI_X_END_RATIO
        # Bottom-left Y: ROI_Y_END_RATIO
        # Bottom-left X: max(0.0, ROI_X_START_RATIO - ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
        # Bottom-right Y: ROI_Y_END_RATIO
        # Bottom-right X: min(1.0, ROI_X_END_RATIO + ROI_BOTTOM_WIDTH_FACTOR_OFFSET)

        roi_tl_x = ROI_X_START_RATIO
        roi_tl_y = ROI_Y_RATIO
        roi_tr_x = ROI_X_END_RATIO
        roi_tr_y = ROI_Y_RATIO
        roi_bl_x = max(0.0, ROI_X_START_RATIO - ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
        roi_bl_y = ROI_Y_END_RATIO
        roi_br_x = min(1.0, ROI_X_END_RATIO + ROI_BOTTOM_WIDTH_FACTOR_OFFSET)
        roi_br_y = ROI_Y_END_RATIO

        example_roi_ratios = [
            [roi_tl_x, roi_tl_y],  # Top-left
            [roi_tr_x, roi_tr_y],  # Top-right
            [roi_br_x, roi_br_y],  # Bottom-right
            [roi_bl_x, roi_bl_y]   # Bottom-left
        ]

        # Use ASW perspective ratios from config for consistency
        persp_src_ratios_tl_tr_bl_br = ASW_SRC_RATIOS
        persp_dst_ratios_tl_tr_bl_br = ASW_DST_RATIOS

    except ImportError as e:
        print(f"Could not import from lane_comparison_config.py: {e}")
        print("Using default hardcoded ROI and Perspective parameters for standalone test.")
        # Fallback to original hardcoded values if import fails
        example_roi_ratios = [
            [0.40, 0.60],  # Top-left
            [0.60, 0.60],  # Top-right
            [0.95, 0.95],  # Bottom-right
            [0.05, 0.95]   # Bottom-left
        ]
        persp_src_ratios_tl_tr_bl_br = [
            [0.42, 0.62], [0.58, 0.62],
            [0.10, 0.90], [0.90, 0.90]
        ]
        persp_dst_ratios_tl_tr_bl_br = [
            [0.25, 0.0], [0.75, 0.0],
            [0.25, 1.0], [0.75, 1.0]
        ]

    # Create a dummy image (e.g., 1280x720)
    test_img_h, test_img_w = 720, 1280
    test_img = np.zeros((test_img_h, test_img_w, 3), dtype=np.uint8)
    # Draw some diagonal lines to simulate a road for testing
    cv2.line(test_img, (int(test_img_w*0.2), test_img_h), (int(test_img_w*0.45), int(test_img_h*0.6)), (200,200,200), 20) 
    cv2.line(test_img, (int(test_img_w*0.8), test_img_h), (int(test_img_w*0.55), int(test_img_h*0.6)), (200,200,200), 20)

    print("Attempting to detect lanes on the test image...")
    # Reset global line trackers for a fresh test run
    left_line_global = Line()
    right_line_global = Line()
    
    # For standalone test, convert ratios to absolute for this test call (for original image size)
    example_roi_abs_original_scale = np.array([[int(v[0] * test_img_w), int(v[1] * test_img_h)] for v in example_roi_ratios], dtype=np.int32)

    # --- Standalone Test: Example of Downscaling ---
    standalone_scale_factor = 0.5 # Example: process at half size. Set to 1.0 to test without scaling.
    
    scaled_test_img_w = int(test_img_w * standalone_scale_factor)
    scaled_test_img_h = int(test_img_h * standalone_scale_factor)
    
    if standalone_scale_factor < 1.0:
        scaled_test_img = cv2.resize(test_img, (scaled_test_img_w, scaled_test_img_h), interpolation=cv2.INTER_AREA)
        # ROI vertices must also be scaled for the scaled image
        scaled_example_roi_abs = (example_roi_abs_original_scale.astype(np.float32) * standalone_scale_factor).astype(np.int32)
        print(f"Standalone Test: Processing image at {scaled_test_img_w}x{scaled_test_img_h} (original: {test_img_w}x{test_img_h})")
        image_to_process = scaled_test_img
        roi_for_processing = scaled_example_roi_abs
    else:
        print(f"Standalone Test: Processing image at original size {test_img_w}x{test_img_h}")
        image_to_process = test_img.copy() # Process a copy of the original
        roi_for_processing = example_roi_abs_original_scale.copy()


    detected_lanes_at_processing_scale = detect_lanes_carnd(image_to_process, 
                                                            roi_for_processing, 
                                                            persp_src_ratios_tl_tr_bl_br, 
                                                            persp_dst_ratios_tl_tr_bl_br)
    
    # Scale detected lanes back to original image dimensions for visualization/comparison if scaling was applied
    detected_lanes_output_original_scale = []
    if standalone_scale_factor < 1.0 and detected_lanes_at_processing_scale:
        for lane_scaled in detected_lanes_at_processing_scale:
            if lane_scaled: # Check if lane is not empty
                lane_np_scaled = np.array(lane_scaled, dtype=np.float32)
                lane_original_scale = (lane_np_scaled / standalone_scale_factor).astype(np.int32)
                detected_lanes_output_original_scale.append(lane_original_scale.tolist())
            else:
                detected_lanes_output_original_scale.append([]) # Keep empty lane as is
    else:
        detected_lanes_output_original_scale = detected_lanes_at_processing_scale
    
    print(f"Number of detected lanes (visualized at original scale): {len(detected_lanes_output_original_scale)}")
    if detected_lanes_output_original_scale:
        for i, lane in enumerate(detected_lanes_output_original_scale):
            print(f"  Lane {i+1} has {len(lane)} points.")

    # Visualize (optional) - on the original full-size image
    output_vis_image = test_img.copy()
    # Draw ROI (original scale)
    cv2.polylines(output_vis_image, [example_roi_abs_original_scale], isClosed=True, color=(0,0,255), thickness=2) # Red ROI

    for lane_pts_list in detected_lanes_output_original_scale: # Use lanes scaled back to original
        if lane_pts_list and len(lane_pts_list) > 1:
            pts = np.array(lane_pts_list, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(output_vis_image, [pts], isClosed=False, color=(0, 255, 0), thickness=3) # Green lanes
    
    try:
        cv2.imwrite("detected_lanes_carnd_standalone_test.png", output_vis_image)
        print("Saved standalone test output to detected_lanes_carnd_standalone_test.png")
        
        # If you have a display environment:
        # cv2.imshow("Detected Lanes CarND - Standalone Test", output_vis_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error saving or showing image: {e}")
        print("If you are in an environment without a GUI (like a headless server or some Docker containers), "
              "cv2.imshow will fail. The image is saved as 'detected_lanes_carnd_standalone_test.png'.")
