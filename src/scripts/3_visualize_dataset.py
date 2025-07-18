import cv2
import json
import numpy as np
import os
import sys  # Added sys

# --- Add LaneNet project to sys.path ---
LANENET_PROJECT_ROOT = "c:\\Users\\John\\Desktop\\Masterarbeit_Carla\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\LaneNet_Detection\\lanenet-lane-detection"
if LANENET_PROJECT_ROOT not in sys.path:
    sys.path.append(LANENET_PROJECT_ROOT)

# Try importing LaneNet specific modules
try:
    import tensorflow as tf
    # tf.compat.v1.disable_eager_execution() # Uncomment if issues with TF2.x behavior arise, common for TF1.x models
    from lanenet_model import lanenet
    from lanenet_model import lanenet_postprocess
    from local_utils.config_utils import parse_config_utils  # Corrected CFG import
    CFG = parse_config_utils.lanenet_cfg  # Corrected attribute to lanenet_cfg
except ImportError as e:
    print(f"Error importing LaneNet modules: {e}")
    print(f"Please ensure '{LANENET_PROJECT_ROOT}' is in sys.path and all dependencies are installed.")

# --- Konfiguration für visualize_lanes_on_video ---
WORKSPACE_ROOT_VL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Geht davon aus, dass das Skript in src/ liegt

VIDEO_PATH_VL = os.path.join(WORKSPACE_ROOT_VL, "data", "Town03_Opt.mp4")
LANE_DATA_PATH_VL = os.path.join(WORKSPACE_ROOT_VL, "data", "dataset", "Town03_Opt", "train_gt_tmp.json")
OUTPUT_VIDEO_PATH_VL = os.path.join(WORKSPACE_ROOT_VL, "data", "Town03_Opt_with_lanes.mp4")
DISPLAY_LANE_INDICES_VL = None  # !!!BUGGY!!! z.B. [0] für die erste Spur, [0, 2] für die erste und dritte, None für alle Spuren
MAX_FRAMES_TO_PROCESS = 100  # Maximal zu verarbeitende Frames (0 für alle Frames)

# --- LaneNet Model Configuration ---
LANENET_MODEL_WEIGHTS_PATH = os.path.join(LANENET_PROJECT_ROOT, "weights", "tusimple_lanenet", "tusimple_lanenet.ckpt")


class LaneNetDetector:
    def __init__(self, model_weights_path):
        self.model_weights_path = model_weights_path
        self.sess = None
        self.input_tensor = None
        self.binary_seg_tensor = None
        self.instance_seg_tensor = None

        self._initialize_model()

    def _initialize_model(self):
        try:
            # Configure GPU usage (optional, but good practice)
            # Use tf.compat.v1 for TF1.x APIs if using TF2.x
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
            sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
            sess_config.gpu_options.allocator_type = 'BFC'
            
            self.sess = tf.compat.v1.Session(config=sess_config)

            # Define input placeholder using dimensions from CFG.AUG.TRAIN_CROP_SIZE
            # CFG.AUG.TRAIN_CROP_SIZE is [width, height]
            # Placeholder shape is [batch_size, height, width, channels]
            self.input_tensor = tf.compat.v1.placeholder(dtype=tf.float32,
                                                         shape=[1, CFG.AUG.TRAIN_CROP_SIZE[1], CFG.AUG.TRAIN_CROP_SIZE[0], 3],
                                                         name='input_tensor')
            
            # Build the LaneNet model - pass CFG instead of net_flag
            net = lanenet.LaneNet(phase='test', cfg=CFG) 
            self.binary_seg_tensor, self.instance_seg_tensor = net.inference(self.input_tensor, name='LaneNet')

            # Setup Saver
            saver = tf.compat.v1.train.Saver()

            # Restore weights
            self.sess.run(tf.compat.v1.global_variables_initializer())  # Initialize all variables
            saver.restore(sess=self.sess, save_path=self.model_weights_path)
            print(f"LaneNet model weights loaded from {self.model_weights_path}")

        except Exception as e:
            print(f"Error initializing LaneNetDetector: {e}")
            if self.sess:
                self.sess.close()
            raise

    def detect_lanes(self, image_frame):
        # Image Preprocessing (resize, normalize)
        image_vis = image_frame.copy()  # Keep a copy for visualization
        # Resize to dimensions specified in CFG.AUG.TRAIN_CROP_SIZE: (width, height)
        image_resized = cv2.resize(image_frame, 
                                   (CFG.AUG.TRAIN_CROP_SIZE[0], CFG.AUG.TRAIN_CROP_SIZE[1]), 
                                   interpolation=cv2.INTER_LINEAR)
        image_input = image_resized / 127.5 - 1.0  # Normalize

        # Run Inference
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_tensor, self.instance_seg_tensor],
            feed_dict={self.input_tensor: [image_input]}
        )

        postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_resized
        )

        mask_image = postprocess_result['mask_image']
        detected_lanes_points = []

        # Scale mask_image back to original frame size
        original_height, original_width = image_frame.shape[:2]
        mask_image_scaled = cv2.resize(mask_image, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Scaling factors using CFG.AUG.TRAIN_CROP_SIZE
        # CFG.AUG.TRAIN_CROP_SIZE[0] is width, CFG.AUG.TRAIN_CROP_SIZE[1] is height
        scale_x = original_width / CFG.AUG.TRAIN_CROP_SIZE[0]
        scale_y = original_height / CFG.AUG.TRAIN_CROP_SIZE[1]

        for lane_line in postprocess_result.get('lane_lines', []):
            scaled_lane_points = []
            for point in lane_line:
                x, y = point
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                scaled_lane_points.append([scaled_x, scaled_y])
            if scaled_lane_points:
                detected_lanes_points.append(np.array(scaled_lane_points, dtype=np.int32))

        return detected_lanes_points, mask_image_scaled

    def close(self):
        if self.sess:
            self.sess.close()
            print("LaneNetDetector session closed.")


def visualize_lanes_on_video():
    if not os.path.exists(VIDEO_PATH_VL):
        print(f"Fehler: Videodatei nicht gefunden unter {VIDEO_PATH_VL}")
        return
    if not os.path.exists(LANE_DATA_PATH_VL):
        print(f"Fehler: Spurdatendatei nicht gefunden unter {LANE_DATA_PATH_VL}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH_VL)
    if not cap.isOpened():
        print(f"Fehler: Video konnte nicht geöffnet werden: {VIDEO_PATH_VL}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Eigenschaften: {frame_width}x{frame_height}, FPS: {fps}")
    if fps == 0:
        print("Warnung: FPS konnte nicht aus dem Video gelesen werden. Verwende Standardwert 20 FPS.")
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH_VL, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Fehler: VideoWriter konnte nicht für {OUTPUT_VIDEO_PATH_VL} initialisiert werden.")
        cap.release()
        return

    try:
        try:
            lanenet_detector = LaneNetDetector(model_weights_path=LANENET_MODEL_WEIGHTS_PATH)
        except Exception as e:
            print(f"Failed to initialize LaneNetDetector: {e}. Exiting.")
            if 'lanenet_detector' in locals() and lanenet_detector.sess:
                lanenet_detector.close()
            return

        with open(LANE_DATA_PATH_VL, 'r') as f_lanes:
            frame_idx = 0
            print("Verarbeite Video...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    line_data_str = f_lanes.readline()
                    if not line_data_str:
                        print(f"Warnung: Keine Spurdaten mehr ab Frame {frame_idx}. Stoppe.")
                        out.write(frame)
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            if lanenet_detector:
                                detected_lanes_ln, _ = lanenet_detector.detect_lanes(frame)
                                for lane_pts_ln in detected_lanes_ln:
                                    if len(lane_pts_ln) >= 2:
                                        cv2.polylines(frame, [lane_pts_ln], isClosed=False, color=(0, 0, 255), thickness=2)
                            out.write(frame)
                        break

                    lane_info = json.loads(line_data_str)
                except json.JSONDecodeError:
                    print(f"Warnung: JSON-Dekodierungsfehler für Frame {frame_idx}. Überspringe Frame-Annotation.")
                    out.write(frame)
                    frame_idx += 1
                    continue
                except Exception as e:
                    print(f"Fehler beim Lesen der Spurdaten für Frame {frame_idx}: {e}")
                    out.write(frame)
                    frame_idx += 1
                    continue

                h_samples = lane_info.get('h_samples')
                lanes_coords_all = lane_info.get('lanes')

                if h_samples is None or lanes_coords_all is None:
                    print(f"Warnung: Fehlende 'h_samples' oder 'lanes' in Daten für Frame {frame_idx}. Überspringe Annotation.")
                    out.write(frame)
                    frame_idx += 1
                    continue

                lanes_to_draw_coords = []
                if DISPLAY_LANE_INDICES_VL is None or not DISPLAY_LANE_INDICES_VL:
                    lanes_to_draw_coords = lanes_coords_all
                else:
                    for index in DISPLAY_LANE_INDICES_VL:
                        if 0 <= index < len(lanes_coords_all):
                            lanes_to_draw_coords.append(lanes_coords_all[index])
                        else:
                            print(f"Warnung: Spurindex {index} ist außerhalb des Bereichs für Frame {frame_idx}. Max Index ist {len(lanes_coords_all)-1}. Überspringe diesen Index.")

                for lane_x_coords in lanes_to_draw_coords:
                    points = []
                    if len(lane_x_coords) != len(h_samples):
                        print(f"Warnung: Inkonsistenz zwischen x_coords und h_samples für eine Spur in Frame {frame_idx}. Überspringe diese Spur.")
                        continue
                    for i in range(len(h_samples)):
                        x = lane_x_coords[i]
                        y = h_samples[i]
                        if x != -2:
                            points.append((int(x), int(y)))

                    if len(points) >= 2:
                        cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

                if lanenet_detector:
                    try:
                        detected_lanes_ln, _ = lanenet_detector.detect_lanes(frame.copy())
                        for lane_pts_ln in detected_lanes_ln:
                            if len(lane_pts_ln) >= 2:
                                cv2.polylines(frame, [lane_pts_ln], isClosed=False, color=(0, 0, 255), thickness=2)
                    except Exception as e:
                        print(f"Error during LaneNet detection on frame {frame_idx}: {e}")

                out.write(frame)

                if frame_idx % 100 == 0:
                    print(f"Frame {frame_idx} verarbeitet.")
                frame_idx += 1

                if frame_idx >= MAX_FRAMES_TO_PROCESS and MAX_FRAMES_TO_PROCESS > 0:
                    break

            print(f"Verarbeitung abgeschlossen. {frame_idx} Frames verarbeitet.")

    except FileNotFoundError:
        print(f"Fehler: Spurdatendatei nicht gefunden: {LANE_DATA_PATH_VL}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    finally:
        if 'lanenet_detector' in locals() and lanenet_detector:
            lanenet_detector.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if os.path.exists(OUTPUT_VIDEO_PATH_VL) and os.path.getsize(OUTPUT_VIDEO_PATH_VL) > 0:
        print(f"Video mit eingezeichneten Spuren gespeichert unter: {OUTPUT_VIDEO_PATH_VL}")
    else:
        print(f"Fehler: Ausgabevideo {OUTPUT_VIDEO_PATH_VL} wurde nicht oder leer erstellt.")


if __name__ == '__main__':
    visualize_lanes_on_video()
