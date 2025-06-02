import cv2
import numpy as np
import math

def fit_polynomial_to_lane_segments(image_shape, lines, roi_vertices_for_y_eval, 
                                  slope_left_min, slope_left_max, 
                                  slope_right_min, slope_right_max,
                                  min_segments_for_fit, min_points_for_polyfit,
                                  num_points_to_generate=20):
    """
    Fits a 2nd degree polynomial to given line segments and generates points along the curve.
    Separates lines into potential left and right lanes based on slope.
    Uses configurable slope ranges and minimum segment/point counts.
    Returns a list of lists of points, where each inner list is a detected lane curve.
    """
    left_lane_points_x = []
    left_lane_points_y = []
    left_lane_segments_count = 0
    right_lane_points_x = []
    right_lane_points_y = []
    right_lane_segments_count = 0

    if lines is None or len(lines) == 0: # Check if lines is empty
        return []

    # ROI y-range for polynomial evaluation
    roi_all_y_coords = roi_vertices_for_y_eval[0][:, 1]
    y_start_eval = np.min(roi_all_y_coords)
    y_end_eval = np.max(roi_all_y_coords)

    for line_segment in lines: # lines is now a list of individual segments [[x1,y1,x2,y2]]
        x1, y1, x2, y2 = line_segment[0] # Each segment is wrapped in a list

        if (x2 - x1) == 0:
            slope = np.inf if (y2 - y1) > 0 else -np.inf # Vertical line
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        # Classify based on slope
        if slope_left_min <= slope <= slope_left_max:
            left_lane_points_x.extend([x1, x2])
            left_lane_points_y.extend([y1, y2])
            left_lane_segments_count += 1
        elif slope_right_min <= slope <= slope_right_max:
            right_lane_points_x.extend([x1, x2])
            right_lane_points_y.extend([y1, y2])
            right_lane_segments_count += 1

    fitted_lanes = []
    if y_start_eval >= y_end_eval: # Should not happen with valid ROI
        plot_y = np.array([int(y_start_eval)]) 
    else:
        plot_y = np.linspace(int(y_start_eval), int(y_end_eval), num_points_to_generate)

    # Fit left lane
    if left_lane_segments_count >= min_segments_for_fit and len(left_lane_points_x) >= min_points_for_polyfit:
        try:
            left_fit_params = np.polyfit(left_lane_points_y, left_lane_points_x, 2) 
            left_fit_x = np.polyval(left_fit_params, plot_y)
            left_lane_curve = list(zip(np.int_(np.round(left_fit_x)), np.int_(np.round(plot_y))))
            if left_lane_curve:
                fitted_lanes.append(left_lane_curve)
        except (np.RankWarning, TypeError, ValueError, np.linalg.LinAlgError) as e:
            # print(f"Hough: Warning/Error fitting left lane polynomial: {e}")
            pass

    # Fit right lane
    if right_lane_segments_count >= min_segments_for_fit and len(right_lane_points_x) >= min_points_for_polyfit:
        try:
            right_fit_params = np.polyfit(right_lane_points_y, right_lane_points_x, 2)
            right_fit_x = np.polyval(right_fit_params, plot_y)
            right_lane_curve = list(zip(np.int_(np.round(right_fit_x)), np.int_(np.round(plot_y))))
            if right_lane_curve:
                fitted_lanes.append(right_lane_curve)
        except (np.RankWarning, TypeError, ValueError, np.linalg.LinAlgError) as e:
            # print(f"Hough: Warning/Error fitting right lane polynomial: {e}")
            pass

    return fitted_lanes

def detect_lanes_hough(image, roi_vertices, config):
    """
    Detects lanes using Hough Transform with enhanced filtering and polynomial fitting.
    Args:
        image: Input BGR image (NumPy array).
        roi_vertices: A NumPy array of shape (1, N, 2) defining the region of interest.
        config: The configuration module (lane_comparison_config).
    Returns:
        A list of detected lane curves (each a list of points).
    """
    if image is None:
        return []

    # 1. Create a mask for the ROI
    mask = np.zeros_like(image[:, :, 0]) # Grayscale mask
    cv2.fillPoly(mask, [roi_vertices], 255)
    
    # Get ROI bounding box for more efficient processing
    x_offset, y_offset, w, h = cv2.boundingRect(roi_vertices)
    
    # Crop the image and mask to the ROI bounding box
    # Ensure slicing does not create an empty array if ROI is at the image edge
    roi_image_slice = image[y_offset:min(y_offset+h, image.shape[0]), x_offset:min(x_offset+w, image.shape[1])]
    roi_mask_slice = mask[y_offset:min(y_offset+h, image.shape[0]), x_offset:min(x_offset+w, image.shape[1])]


    if roi_image_slice.size == 0 or roi_mask_slice.size == 0:
        # print("Hough: ROI slice is empty.")
        return []

    # 2. Preprocessing within ROI
    gray_roi = cv2.cvtColor(roi_image_slice, cv2.COLOR_BGR2GRAY)
    
    blur_kernel = getattr(config, 'HOUGH_GAUSSIAN_BLUR_KERNEL', (5,5))
    blur_sigma_x = getattr(config, 'HOUGH_GAUSSIAN_BLUR_SIGMA_X', 0)
    # Ensure kernel dimensions are odd
    blur_kernel = (blur_kernel[0] if blur_kernel[0] % 2 != 0 else blur_kernel[0] + 1, 
                   blur_kernel[1] if blur_kernel[1] % 2 != 0 else blur_kernel[1] + 1)
    blur_roi = cv2.GaussianBlur(gray_roi, blur_kernel, blur_sigma_x)
    
    # 3. Canny Edge Detection within ROI
    canny_low = getattr(config, 'HOUGH_CANNY_LOW_THRESHOLD', 50)
    canny_high = getattr(config, 'HOUGH_CANNY_HIGH_THRESHOLD', 150)
    edges_roi = cv2.Canny(blur_roi, canny_low, canny_high)
    
    # Apply the ROI mask to the edges
    # Ensure roi_mask_slice has the same dimensions as edges_roi
    if edges_roi.shape != roi_mask_slice.shape:
        # print(f"Hough: Shape mismatch between edges_roi {edges_roi.shape} and roi_mask_slice {roi_mask_slice.shape}. Resizing mask.")
        # This can happen if the ROI polygon is very thin or oddly shaped, leading to a bounding box
        # that Canny might process slightly differently at the edges.
        # We resize the mask to match the Canny output for bitwise_and.
        roi_mask_slice_resized = cv2.resize(roi_mask_slice, (edges_roi.shape[1], edges_roi.shape[0]))
        edges_roi_masked = cv2.bitwise_and(edges_roi, edges_roi, mask=roi_mask_slice_resized)
    else:
        edges_roi_masked = cv2.bitwise_and(edges_roi, edges_roi, mask=roi_mask_slice)


    # 4. Hough Transform
    raw_lines_roi = cv2.HoughLinesP(
        edges_roi_masked,
        config.HOUGH_RHO,
        np.pi / config.HOUGH_THETA_DIVISOR,
        config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH, # Initial filter by HoughP
        maxLineGap=config.HOUGH_MAX_LINE_GAP
    )
    
    if raw_lines_roi is None:
        return []

    # 5. Pre-filter Hough lines (still in ROI coordinates)
    pre_filtered_lines_roi = []
    min_len_sq = config.HOUGH_MIN_LINE_LENGTH ** 2 # Compare squared lengths to avoid sqrt
    max_horizontal_slope_dev = getattr(config, 'HOUGH_MAX_HORIZONTAL_SLOPE_DEVIATION', 0.15)

    for line_segment in raw_lines_roi:
        x1, y1, x2, y2 = line_segment[0]
        
        # Length filter (already partially done by minLineLength in HoughLinesP, but can be more strict here)
        # dx = x2 - x1
        # dy = y2 - y1
        # if (dx*dx + dy*dy) < min_len_sq: # Stricter length check if needed
        #     continue

        # Slope filter for horizontal lines
        if (x2 - x1) == 0: # Vertical line, not horizontal
            slope = np.inf 
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        if abs(slope) < max_horizontal_slope_dev:
            continue # Skip lines that are too horizontal
            
        pre_filtered_lines_roi.append([[x1, y1, x2, y2]]) # Keep the original list-of-list structure

    if not pre_filtered_lines_roi:
        return []
        
    # 6. Convert coordinates of pre-filtered lines to global image coordinates
    globally_adjusted_lines = []
    for line_segment in pre_filtered_lines_roi:
        x1, y1, x2, y2 = line_segment[0]
        globally_adjusted_lines.append([[x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]])

    # 7. Fit polynomials to the globally adjusted, pre-filtered lines
    # Pass the necessary slope and count parameters from config
    fitted_lane_curves = fit_polynomial_to_lane_segments(
        image_shape=image.shape,
        lines=globally_adjusted_lines,
        roi_vertices_for_y_eval=roi_vertices, # Use original ROI for y-evaluation range
        slope_left_min=config.HOUGH_SLOPE_LEFT_MIN,
        slope_left_max=config.HOUGH_SLOPE_LEFT_MAX,
        slope_right_min=config.HOUGH_SLOPE_RIGHT_MIN,
        slope_right_max=config.HOUGH_SLOPE_RIGHT_MAX,
        min_segments_for_fit=config.HOUGH_MIN_SEGMENTS_FOR_LANE_FIT,
        min_points_for_polyfit=config.HOUGH_MIN_POINTS_FOR_POLYFIT
    )
    
    return fitted_lane_curves