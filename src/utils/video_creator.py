#!/usr/bin/env python3
"""
Video Creator for Lane Detection Results
Creates videos from comparison images for dashboard preview
"""

import os
import cv2
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class VideoCreator:
    """Create videos from comparison images"""
    
    def __init__(self, output_dir: str = "results/videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_video_from_images(self, image_paths: List[str], 
                                output_name: str = "comparison_video.mp4",
                                fps: int = 10) -> Optional[str]:
        """
        Create a video from a list of image paths
        
        Args:
            image_paths: List of paths to comparison images
            output_name: Name of the output video file
            fps: Frames per second for the output video
            
        Returns:
            Path to the created video file or None if failed
        """
        try:
            if not image_paths:
                logger.warning("No images provided for video creation")
                return None
            
            # Get first image to determine video dimensions
            first_img = cv2.imread(image_paths[0])
            if first_img is None:
                logger.error(f"Could not read first image: {image_paths[0]}")
                return None
            
            height, width, _ = first_img.shape
            
            # Create video writer
            output_path = os.path.join(self.output_dir, output_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                logger.error(f"Could not open video writer for {output_path}")
                return None
            
            # Add each image to the video
            for img_path in image_paths:
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize if necessary
                        if img.shape[:2] != (height, width):
                            img = cv2.resize(img, (width, height))
                        video_writer.write(img)
                    else:
                        logger.warning(f"Could not read image: {img_path}")
                else:
                    logger.warning(f"Image not found: {img_path}")
            
            video_writer.release()
            logger.info(f"Created video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return None
    
    def create_live_video(self, comparison_dir: str, 
                         output_name: str = "live_comparison.mp4",
                         max_images: int = 100,
                         fps: int = 10) -> Optional[str]:
        """
        Create a video from the most recent comparison images
        
        Args:
            comparison_dir: Directory containing comparison images
            output_name: Name of the output video file
            max_images: Maximum number of images to include
            fps: Frames per second for the output video
            
        Returns:
            Path to the created video file or None if failed
        """
        try:
            # Get all comparison images
            image_files = []
            for filename in os.listdir(comparison_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(comparison_dir, filename))
            
            # Sort by modification time (newest first)
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Take the most recent images
            recent_images = image_files[:max_images]
            recent_images.reverse()  # Reverse to show oldest to newest
            
            return self.create_video_from_images(recent_images, output_name, fps)
            
        except Exception as e:
            logger.error(f"Error creating live video: {e}")
            return None
    
    def update_live_video(self, new_image_path: str, 
                         video_name: str = "live_comparison.mp4") -> Optional[str]:
        """
        Update the live video with a new image
        This creates a new video with the latest images
        
        Args:
            new_image_path: Path to the new comparison image
            video_name: Name of the video to update
            
        Returns:
            Path to the updated video file or None if failed
        """
        try:
            # Get the directory of the new image
            comparison_dir = os.path.dirname(new_image_path)
            
            # Create updated video
            return self.create_live_video(comparison_dir, video_name)
            
        except Exception as e:
            logger.error(f"Error updating live video: {e}")
            return None

def create_comparison_video(comparison_dir: str, 
                          output_path: str = None,
                          fps: int = 10) -> Optional[str]:
    """
    Convenience function to create a video from comparison images
    
    Args:
        comparison_dir: Directory containing comparison images
        output_path: Path for the output video (optional)
        fps: Frames per second for the output video
        
    Returns:
        Path to the created video file or None if failed
    """
    if output_path is None:
        output_path = os.path.join(comparison_dir, "comparison_video.mp4")
    
    video_creator = VideoCreator(os.path.dirname(output_path))
    return video_creator.create_live_video(
        comparison_dir, 
        os.path.basename(output_path), 
        fps=fps
    )

if __name__ == "__main__":
    # Example usage
    comparison_dir = "data/comparison_results_modular"
    video_path = create_comparison_video(comparison_dir)
    if video_path:
        print(f"Video created: {video_path}")
    else:
        print("Failed to create video")
