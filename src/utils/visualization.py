from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def draw_lanes_on_image(image, lanes, color=(0, 255, 0), thickness=2):
    """Draws lane lines on an image."""
    for lane in lanes:
        for i in range(len(lane) - 1):
            p1 = tuple(map(int, lane[i]))
            p2 = tuple(map(int, lane[i+1]))
            cv2.line(image, p1, p2, color, thickness)
    return image

def create_comparison_image(base_image, ground_truth_lanes, algorithm_results, h_samples):
    """
    Creates a composite image showing the original image with lanes from
    ground truth and multiple algorithms overlaid.
    """
    # Create a PIL image from the numpy array
    vis_image = Image.fromarray(base_image)
    draw = ImageDraw.Draw(vis_image)

    # Draw Ground Truth
    # ... (implementation for drawing ground truth lanes if available)

    # Draw Algorithm Lanes
    for name, result in algorithm_results.items():
        lanes = result.get('lanes', [])
        color = result.get('color', (255, 255, 255))
        
        if not lanes:
            continue

        for lane in lanes:
            # Ensure the lane is a list/tuple and has at least 2 points to draw a line
            if not isinstance(lane, (list, tuple)) or len(lane) < 2:
                continue

            # Ensure all points in the lane are valid coordinate pairs (list/tuple of 2 numbers)
            # This prevents the "incorrect coordinate type" error from PIL
            clean_lane = []
            for point in lane:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    try:
                        # Convert to tuple of ints, which is what PIL expects
                        clean_lane.append(tuple(map(int, point)))
                    except (ValueError, TypeError):
                        # Point contains non-numeric data, skip it
                        continue 
            
            # Only draw if we have at least 2 valid points after cleaning
            if len(clean_lane) > 1:
                draw.line(clean_lane, fill=color, width=3)

    # Add a legend
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    y_offset = 10
    # ... (legend drawing logic)

    return vis_image
