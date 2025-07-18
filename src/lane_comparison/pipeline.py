import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import importlib
import time

# Import the full config module and the ROI utility function
import src.config as config
from src.config import get_config, get_available_algorithms
from src.utils.image_processing import resize_image
from src.utils.visualization import draw_lanes_on_image, create_comparison_image
from src.evaluation.metrics import calculate_lane_metrics as evaluate_lanes
from src.evaluation.metrics import define_roi_vertices_from_config


class LaneComparisonPipeline:
    """
    Orchestrates the process of running lane detection algorithms on a video,
    generating a comparison video, and calculating metrics.
    """
    def __init__(self, video_path, selected_algo_names, progress_callback=None, log_callback=None, image_callback=None, frame_callback=None, metrics_callback=None):
        self.video_path = video_path
        self.selected_algo_names = selected_algo_names
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.image_callback = image_callback
        # callback to deliver live processed frames and metrics
        self.frame_callback = frame_callback
        # deprecated per-frame metrics callback (unused)
        self.metrics_callback = metrics_callback
        self.stop_requested = False
        # load ground truth data
        import json
        gt_path = get_config("PATHS", "GROUND_TRUTH_JSON")
        try:
            with open(gt_path, 'r') as f:
                self.gt_data = [json.loads(line) for line in f]
        except Exception:
            self.gt_data = []

        self.algorithms = self._load_algorithms()
        self.paths = {
            "output_vis_dir": get_config("PATHS", "OUTPUT_VIS_DIR"),
            "output_video": get_config("PATHS", "OUTPUT_VIDEO_PATH"),
            "gt_json": get_config("PATHS", "GROUND_TRUTH_JSON")
        }
        self.settings = {
            "resize_factor": get_config("SETTINGS", "RESIZE_PROCESSING_FACTOR"),
            "video_fps": get_config("SETTINGS", "VIDEO_FPS")
        }

        os.makedirs(self.paths["output_vis_dir"], exist_ok=True)

    def _log(self, message, level="INFO"):
        """Logs a message to the console or a callback, with optional level."""
        log_msg = f"[{level}] {message}"
        if self.log_callback:
            # Add a newline for the text box, but don't print it to the console
            self.log_callback(log_msg + '\n')
        else:
            print(log_msg)

    def _load_algorithms(self):
        """Loads the functions for the selected algorithms."""
        loaded_algos = {}
        all_algos = get_available_algorithms()
        for algo_config in all_algos:
            if algo_config.name in self.selected_algo_names:
                try:
                    module = importlib.import_module(f"src.algorithms.{algo_config.module_name}")
                    func = getattr(module, algo_config.function_name)
                    loaded_algos[algo_config.name] = {
                        "func": func,
                        "config": algo_config
                    }
                    self._log(f"Successfully loaded algorithm: {algo_config.display_name}")
                except Exception as e:
                    self._log(f"Error loading algorithm '{algo_config.name}': {e}", level="ERROR")
        return loaded_algos

    def run(self):
        """
        Executes the entire comparison pipeline.
        """
        self._log("Starting video processing pipeline...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._log(f"Error: Could not open video file {self.video_path}", level="ERROR")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_writer = None
        
        # --- Performance and Logging Control ---
        frame_processing_times = []
        log_interval_seconds = 2 # Log progress every 2 seconds
        last_log_time = time.time()
        
        # --- Frame Limiting ---
        num_frames_to_process = get_config("SETTINGS", "PROCESS_NUM_IMAGES")
        if num_frames_to_process is None or num_frames_to_process <= 0:
            num_frames_to_process = total_frames
        else:
            num_frames_to_process = min(num_frames_to_process, total_frames)
            self._log(f"Processing a subset of {num_frames_to_process} frames as per settings.")


        for frame_idx in range(num_frames_to_process):
            if self.stop_requested:
                self._log("Pipeline execution stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()

            # Process frame
            # process frame and handle metrics inside
            processed_frame = self._process_frame(frame, frame_idx)
            
            end_time = time.time()
            frame_processing_times.append(end_time - start_time)


            # Initialize video writer with frame dimensions
            if video_writer is None:
                h, w, _ = processed_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(self.paths["output_video"], fourcc, self.settings["video_fps"], (w, h))

            video_writer.write(processed_frame)

            # Update progress
            if self.progress_callback:
                self.progress_callback((frame_idx + 1) / num_frames_to_process)
            
            # Log progress periodically instead of every frame
            current_time = time.time()
            if current_time - last_log_time >= log_interval_seconds:
                avg_fps = (frame_idx + 1) / sum(frame_processing_times) if frame_processing_times else 0
                self._log(f"Processed frame {frame_idx + 1}/{num_frames_to_process} | Avg FPS: {avg_fps:.2f}")
                last_log_time = current_time


        cap.release()
        if video_writer:
            video_writer.release()
            
        # Final performance summary
        if frame_processing_times:
            avg_total_fps = len(frame_processing_times) / sum(frame_processing_times)
            self._log(f"Pipeline finished. Total frames processed: {len(frame_processing_times)}.")
            self._log(f"Average processing speed: {avg_total_fps:.2f} FPS.")
        else:
            self._log("Pipeline finished. No frames were processed.")

    def _process_frame(self, frame, frame_idx):
        """Processes a single frame through the selected algorithms."""
        original_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define ROI vertices for the current frame using the utility function
        roi_vertices = define_roi_vertices_from_config(frame.shape, config)
        
        algo_results = {}
        for name, algo in self.algorithms.items():
            # No per-frame logging here anymore to reduce noise
            
            try:
                # Pass the required arguments: image, roi_vertices, and the config module
                detected_lanes = algo["func"](frame.copy(), roi_vertices, config)
            except TypeError as e:
                self._log(f"ERROR calling algorithm '{name}' on frame {frame_idx}: {e}", level="ERROR")
                self._log(f"Please check the function signature for '{algo['config'].function_name}'.", level="ERROR")
                self._log("Expected: func(image, roi_vertices, config)", level="ERROR")
                detected_lanes = [] # Default to empty list on error

            algo_results[name] = {
                "lanes": detected_lanes,
                "color": algo["config"].color,
                "display_name": algo["config"].display_name
            }

        # Create the visualization image
        comparison_img = create_comparison_image(
            original_image_rgb,
            ground_truth_lanes=[], # No GT for now
            algorithm_results=algo_results,
            h_samples=[] # No h_samples for now
        )
        
        # Save individual frame for debugging/analysis
        output_img_path = os.path.join(self.paths["output_vis_dir"], f"comp_{frame_idx:04d}.jpg")
        comparison_img.save(output_img_path)

        # Pass the path to the new image back to the GUI
        if self.image_callback:
            self.image_callback(output_img_path)

        # Convert PIL image back to OpenCV format for video writer
        return cv2.cvtColor(np.array(comparison_img), cv2.COLOR_RGB2BGR)


    def request_stop(self):
        """Signals the pipeline to stop processing."""
        self.stop_requested = True
