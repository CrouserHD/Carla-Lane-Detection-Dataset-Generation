import cv2
import numpy as np
import math

class AdvancedLaneDetectorInternal:
    def __init__(self, config_module):
        self.config = config_module
        self.nwindows = getattr(self.config, 'ASW_NWINDOWS', 9)
        self.margin = getattr(self.config, 'ASW_MARGIN', 80)
        self.minpix = getattr(self.config, 'ASW_MINPIX', 40)
        self.poly_degree = getattr(self.config, 'ASW_POLY_DEGREE', 2)
        self.min_lane_dist_warped = getattr(self.config, 'ASW_MIN_LANE_DIST_WARPED', 50)

        self.s_thresh_min = getattr(self.config, 'ASW_S_THRESH_MIN', 170)
        self.s_thresh_max = getattr(self.config, 'ASW_S_THRESH_MAX', 255)
        self.sobel_kernel_size = getattr(self.config, 'ASW_SOBEL_KERNEL_SIZE', 5)
        self.sobel_thresh_min = getattr(self.config, 'ASW_SOBEL_THRESH_MIN', 30)
        self.sobel_thresh_max = getattr(self.config, 'ASW_SOBEL_THRESH_MAX', 150)
        
        self.src_ratios = getattr(self.config, 'ASW_SRC_RATIOS', 
                                  [[0.42, 0.65], [0.58, 0.65], [0.15, 0.95], [0.85, 0.95]])
        self.dst_ratios = getattr(self.config, 'ASW_DST_RATIOS',
                                  [[0.25, 0.0], [0.75, 0.0], [0.25, 1.0], [0.75, 1.0]])

        self.left_fit = None
        self.right_fit = None
        
        self.M = None
        self.Minv = None
        self.img_h = None
        self.img_w = None

    def initialize_perspective_transform(self, height, width):
        src_pts = np.float32([[sx * width, sy * height] for sx, sy in self.src_ratios])
        dst_pts = np.float32([[dx * width, dy * height] for dx, dy in self.dst_ratios])

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.img_h = height
        self.img_w = width

    def preprocess_image(self, bgr_image):
        hls = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2] # S channel
        l_channel = hls[:, :, 1] # L channel

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh_min) & (s_channel <= self.s_thresh_max)] = 1

        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_size)
        abs_sobel_x = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255 * abs_sobel_x / (np.max(abs_sobel_x) if np.max(abs_sobel_x) > 0 else 1))

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= self.sobel_thresh_min) & (scaled_sobel <= self.sobel_thresh_max)] = 1

        combined_binary = np.zeros_like(s_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
        return combined_binary

    def warp_to_birdeye(self, binary_image):
        h, w = binary_image.shape[:2]
        if self.M is None or self.img_h != h or self.img_w != w:
            self.initialize_perspective_transform(h, w)
        if self.M is None: # Should not happen if init is correct
            return None
        return cv2.warpPerspective(binary_image, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def find_lane_pixels(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = binary_warped.shape[0] // self.nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError: # Can happen if inds lists are empty
            pass

        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped):
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        left_fitx = None
        right_fitx = None

        min_points_for_fit = max(10, self.poly_degree + 1) # Ensure enough points for polyfit

        if len(lefty) > min_points_for_fit and len(leftx) > min_points_for_fit:
            self.left_fit = np.polyfit(lefty, leftx, self.poly_degree)
        if self.left_fit is not None:
            left_fitx = np.polyval(self.left_fit, ploty)

        if len(righty) > min_points_for_fit and len(rightx) > min_points_for_fit:
            self.right_fit = np.polyfit(righty, rightx, self.poly_degree)
        if self.right_fit is not None:
            right_fitx = np.polyval(self.right_fit, ploty)

        # Enforce minimum distance if both lanes are detected
        if left_fitx is not None and right_fitx is not None:
            distances = right_fitx - left_fitx
            invalid_dist_indices = distances < self.min_lane_dist_warped
            
            # Option 1: Invalidate both if any part is too close (simplistic)
            # if np.any(invalid_dist_indices):
            #    left_fitx, right_fitx = None, None # Or revert to previous full fits if available

            # Option 2: Try to adjust or use previous if current is bad
            # For now, we keep them and let visualization show it.
            # A more robust system might try to refit or selectively invalidate.
            pass


        return left_fitx, right_fitx, ploty

    def get_lane_lines_from_image(self, bgr_image):
        if bgr_image is None:
            return []
        
        h, w = bgr_image.shape[:2]
        if self.Minv is None or self.img_h != h or self.img_w != w: # Ensure Minv is initialized for this image size
            self.initialize_perspective_transform(h,w)

        if self.Minv is None: # If initialization failed for some reason
            return []

        binary_image = self.preprocess_image(bgr_image)
        binary_warped = self.warp_to_birdeye(binary_image)
        
        if binary_warped is None:
            return []

        left_fitx, right_fitx, ploty = self.fit_polynomial(binary_warped)

        detected_lanes_orig_coords = []

        if left_fitx is not None:
            left_lane_warped = np.array([left_fitx, ploty]).T.astype(np.float32)
            if len(left_lane_warped) > 0:
                left_lane_orig = cv2.perspectiveTransform(np.array([left_lane_warped]), self.Minv)[0]
                detected_lanes_orig_coords.append([[int(p[0]), int(p[1])] for p in left_lane_orig])
        
        if right_fitx is not None:
            right_lane_warped = np.array([right_fitx, ploty]).T.astype(np.float32)
            if len(right_lane_warped) > 0:
                right_lane_orig = cv2.perspectiveTransform(np.array([right_lane_warped]), self.Minv)[0]
                detected_lanes_orig_coords.append([[int(p[0]), int(p[1])] for p in right_lane_orig])
                
        return detected_lanes_orig_coords

_adv_detector_instance = None

def detect_lanes_advanced_sliding_window(image, roi_vertices, algo_specific_params, common_config):
    """
    Detects lanes using an advanced sliding window algorithm.
    Args:
        image: Input BGR image (NumPy array).
        roi_vertices: A NumPy array defining the region of interest (currently not directly used by this algo).
        algo_specific_params: Dictionary of parameters specific to this algorithm (not used).
        common_config: The configuration module (lane_comparison_config).
    Returns:
        A list of detected lanes. Each lane is a list of [x, y] points in original image coordinates.
        e.g., [ [[x1l,y1l], [x2l,y2l], ...], [[x1r,y1r], [x2r,y2r], ...] ]
    """
    global _adv_detector_instance
    if _adv_detector_instance is None:
        _adv_detector_instance = AdvancedLaneDetectorInternal(common_config)
    
    # Update config reference if it changed (e.g. if common_config is a reloaded module)
    # This is more robust if the common_config object identity can change.
    if _adv_detector_instance.config is not common_config:
         _adv_detector_instance.config = common_config
         # Potentially re-initialize parts of the detector that depend on config values
         # if they are not dynamically read (like nwindows, margin etc. are on init)
         # For now, assuming config values used at __init__ are sufficient or
         # critical ones like thresholds are reread or passed appropriately.
         # The current AdvancedLaneDetectorInternal reads most params at __init__.
         # A full re-init might be safer if config changes are expected during a session:
         # _adv_detector_instance = AdvancedLaneDetectorInternal(common_config)


    return _adv_detector_instance.get_lane_lines_from_image(image)

