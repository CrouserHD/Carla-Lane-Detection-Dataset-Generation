import os
import sys
import cv2
import numpy as np
import time

LANE_NET_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'LaneNet_Detection', 'lanenet-lane-detection'))
if LANE_NET_PROJECT_ROOT not in sys.path:
    sys.path.append(LANE_NET_PROJECT_ROOT)

try:
    # These are lightweight utils, safe for top-level import if they don't import TF themselves.
    from local_utils.config_utils import parse_config_utils 
    from local_utils.log_util import init_logger
except ImportError as e:
    print(f"CRITICAL Error importing LaneNet local_utils: {e}. Ensure LANE_NET_PROJECT_ROOT ('{LANE_NET_PROJECT_ROOT}') is correct.")
    raise

_CFG_LN_SINGLETON = None
_lanenet_algorithm_instance = None
_current_instance_params = {"weights_path": None, "use_gpu": None, "pid": None}
_process_specific_logger_cache = {}

_tf_module = None
_lanenet_module = None
_lanenet_postprocess_module = None

def _get_or_load_lanenet_config():
    global _CFG_LN_SINGLETON
    if _CFG_LN_SINGLETON is None:
        pid = os.getpid()
        print(f"PID {pid}: Loading LaneNet configuration (parse_config_utils.lanenet_cfg) for the first time.")
        
        # Initial load from LaneNet's utility
        cfg_candidate = parse_config_utils.lanenet_cfg # This is an addict.Dict (Config object)

        # Attempt to get LOG_DIR. It should be a string like './log' if defined as such in YAML.
        log_dir_from_config = getattr(cfg_candidate, 'LOG_DIR', None)

        final_log_dir_path_str = None

        if isinstance(log_dir_from_config, str):
            # It's already a string, use it.
            final_log_dir_path_str = log_dir_from_config
        else:
            # LOG_DIR is not a string (it's None, or another Config object, or something else)
            # Assign a default string path.
            if log_dir_from_config is not None:
                print(f"PID {pid}: LOG_DIR from config is not a string (type: {type(log_dir_from_config)}, value: {log_dir_from_config}). Using default path.")
            else:
                print(f"PID {pid}: LOG_DIR not found in config. Using default path.")
            final_log_dir_path_str = os.path.join('log', 'lane_comparison_workers_default_cfg_fallback') # Relative to project root by default
        
        # Ensure the path is absolute, relative to LANE_NET_PROJECT_ROOT if not already absolute.
        if not os.path.isabs(final_log_dir_path_str):
            # This handles cases like './log' or 'some_relative_dir/log'
            final_log_dir_path_str = os.path.join(LANE_NET_PROJECT_ROOT, final_log_dir_path_str)
        
        # Normalize the path (e.g., collapses ../, //, etc.)
        final_log_dir_path_str = os.path.normpath(final_log_dir_path_str)

        # CRITICAL STEP: Update the LOG_DIR attribute on the cfg_candidate object
        # to be this processed string path.
        cfg_candidate.LOG_DIR = final_log_dir_path_str
        
        # Now assign the (potentially modified) cfg_candidate to the global singleton
        _CFG_LN_SINGLETON = cfg_candidate
        
        # Create the directory using the guaranteed string path from the singleton
        try:
            # _CFG_LN_SINGLETON.LOG_DIR is now guaranteed to be a string
            os.makedirs(_CFG_LN_SINGLETON.LOG_DIR, exist_ok=True)
        except Exception as e:
            print(f"PID {pid}: Error creating LOG_DIR '{_CFG_LN_SINGLETON.LOG_DIR}': {e}")
            # Handle error if directory creation is critical. For now, just print.

    return _CFG_LN_SINGLETON

def _ensure_tf_imported(logger):
    global _tf_module, _lanenet_module, _lanenet_postprocess_module
    if _tf_module is None:
        #logger.debug("Attempting to reduce TensorFlow log verbosity for this process...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0:all, 1:filter INFO, 2:filter INFO+WARNING, 3:filter INFO+WARNING+ERROR
        try:
            import logging # Ensure logging is imported
            logging.getLogger('tensorflow').setLevel(logging.ERROR)
            # For h5py, if its DEBUG logs are also too verbose
            # logging.getLogger('h5py').setLevel(logging.INFO) # Or WARNING/ERROR
        except Exception as e_log_config:
            logger.warning(f"Could not configure TensorFlow/h5py logging levels: {e_log_config}")

        #logger.debug("Importing TensorFlow and LaneNet modules for the first time in this process...")
        try:
            import tensorflow as tf
            from lanenet_model import lanenet
            from lanenet_model import lanenet_postprocess
            _tf_module = tf
            _lanenet_module = lanenet
            _lanenet_postprocess_module = lanenet_postprocess
            #logger.debug("TensorFlow and LaneNet modules imported successfully.")
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow or LaneNet modules: {e}", exc_info=True)
            raise
    return _tf_module, _lanenet_module, _lanenet_postprocess_module

def _setup_process_logger(pid):
    if pid not in _process_specific_logger_cache:
        # Load config to get LOG_DIR. This will initialize it if it's the first call in this process.
        current_cfg_ln = _get_or_load_lanenet_config()
        
        log_file_name = f"lanenet_worker_{pid}.log"
        worker_log_dir_detailed = os.path.join(current_cfg_ln.LOG_DIR, 'detailed_worker_logs')
        os.makedirs(worker_log_dir_detailed, exist_ok=True)
        log_path = os.path.join(worker_log_dir_detailed, log_file_name)
        
        logger = init_logger(log_file_path=log_path)
        #logger.info(f"Logger initialized for LaneNet Algorithm, PID {pid}, Log file: {log_path}. Using CFG_LN.LOG_DIR: {current_cfg_ln.LOG_DIR}")
        _process_specific_logger_cache[pid] = logger
    return _process_specific_logger_cache[pid]

class LaneNetAlgorithm:
    def __init__(self, weights_path, use_gpu=True, logger=None):
        self.pid = os.getpid()
        self.logger = logger if logger else _setup_process_logger(self.pid)
        #self.logger.debug(f"Initializing LaneNetAlgorithm (PID: {self.pid}) with weights: '{weights_path}', use_gpu: {use_gpu}")

        self.current_cfg_ln = _get_or_load_lanenet_config() # Get/Load config for this instance
        tf, lanenet_model, lanenet_postprocess_model = _ensure_tf_imported(self.logger)
        
        if not os.path.exists(weights_path + ".index"): 
             self.logger.error(f"LaneNet weights not found at specified path prefix: {weights_path}. Ensure .index and .data files exist.")
             raise FileNotFoundError(f"LaneNet weights not found at {weights_path}")

        self.tf_graph = tf.Graph()
        self.tf_sess = None

        with self.tf_graph.as_default():
            # Ensure tf.compat.v1.placeholder is used if tf version is 2.x
            if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
                tf_placeholder = tf.compat.v1.placeholder
                tf_config_proto = tf.compat.v1.ConfigProto
                tf_session = tf.compat.v1.Session
                tf_train_saver = tf.compat.v1.train.Saver
            else:
                tf_placeholder = tf.placeholder
                tf_config_proto = tf.ConfigProto
                tf_session = tf.Session
                tf_train_saver = tf.train.Saver

            self.input_tensor = tf_placeholder(dtype=tf.float32, shape=[1, self.current_cfg_ln.AUG.TRAIN_CROP_SIZE[1], self.current_cfg_ln.AUG.TRAIN_CROP_SIZE[0], 3], name='input_tensor')
            
            # Determine the net_flag from the loaded configuration
            # This is crucial for matching the checkpoint's graph structure.
            net_flag_from_cfg = self.current_cfg_ln.MODEL.FRONT_END
            #self.logger.debug(f"Using net_flag '{net_flag_from_cfg}' from config for LaneNet model initialization.")

            self.net = lanenet_model.LaneNet(phase='test', cfg=self.current_cfg_ln) # Original LaneNet takes cfg which contains MODEL.FRONT_END
            self.binary_seg_ret, self.instance_seg_ret = self.net.inference(
                self.input_tensor, name='LaneNet', reuse=tf.compat.v1.AUTO_REUSE
            ) # Name scope must match checkpoint
            
            ipm_remap_file = os.path.join(LANE_NET_PROJECT_ROOT, 'data', 'tusimple_ipm_remap.yml')
            if not os.path.exists(ipm_remap_file):
                self.logger.error(f"IPM remap file not found at: {ipm_remap_file}")
                raise FileNotFoundError(f"IPM remap file not found at: {ipm_remap_file}")

            self.postprocessor = lanenet_postprocess_model.LaneNetPostProcessor(cfg=self.current_cfg_ln, ipm_remap_file_path=ipm_remap_file)
            
            saver = tf_train_saver()
            
            sess_config = tf_config_proto()

            # Safely get GPU configuration parameters
            allow_growth_cfg = self.current_cfg_ln.GPU.TF_ALLOW_GROWTH
            if isinstance(allow_growth_cfg, bool):
                allow_growth = allow_growth_cfg
            else:
                self.logger.warning(f"TF_ALLOW_GROWTH is not a boolean (got: {allow_growth_cfg}, type: {type(allow_growth_cfg)}). Using default True.")
                allow_growth = True

            gpu_mem_fraction_cfg = self.current_cfg_ln.GPU.GPU_MEMORY_FRACTION # Corrected key
            if isinstance(gpu_mem_fraction_cfg, (float, int)):
                gpu_mem_fraction = float(gpu_mem_fraction_cfg)
            else:
                # Attempt to convert if it's a string representing a number
                try:
                    gpu_mem_fraction = float(gpu_mem_fraction_cfg)
                    #self.logger.info(f"Successfully converted GPU_MEMORY_FRACTION string '{gpu_mem_fraction_cfg}' to float.") # Updated log message
                except (ValueError, TypeError):
                    self.logger.warning(f"GPU_MEMORY_FRACTION from config is not a number (got: {gpu_mem_fraction_cfg}, type: {type(gpu_mem_fraction_cfg)}). Using default 0.4.") # Updated log message
                    gpu_mem_fraction = 0.4 # Default value

            if use_gpu:
                #self.logger.debug(f"Configuring TensorFlow for GPU usage (PID: {self.pid}). Allow growth: {allow_growth}, Memory fraction: {gpu_mem_fraction}")
                sess_config.gpu_options.allow_growth = allow_growth
                sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
            else:
                #self.logger.debug(f"Configuring TensorFlow for CPU-only usage (PID: {self.pid}).")
                sess_config.device_count['GPU'] = 0
            
            self.tf_sess = tf_session(config=sess_config, graph=self.tf_graph)
            #self.logger.debug(f"Restoring model from {weights_path} (PID: {self.pid})")
            """
            # Diagnostic: List variables in the checkpoint
            try:
                #self.logger.debug(f"--- Variables in Checkpoint ({weights_path}) ---")
                ckpt_vars = tf.train.list_variables(weights_path)
                for var_name, var_shape in ckpt_vars:
                    if 'bisenetv2_backend/instance_seg/pix_bn' in var_name:
                        self.logger.debug(f"Checkpoint Var: {var_name} (Shape: {var_shape})")
                #self.logger.debug(f"Total variables in checkpoint: {len(ckpt_vars)}")
                #self.logger.debug("--------------------------------------------------")
            except Exception as e_ckpt:
                self.logger.warning(f"Error listing checkpoint variables: {e_ckpt}")

            # Diagnostic: List variables in the current graph
            try:
                #self.logger.debug("--- Variables in Current Graph ---")
                graph_vars = [v.name for v in tf.global_variables()] # For TF1.x
                for var_name in graph_vars:
                    if 'bisenetv2_backend/instance_seg/pix_bn' in var_name:
                        self.logger.debug(f"Graph Var: {var_name}")
                #self.logger.debug(f"Total variables in graph: {len(graph_vars)}")
                #self.logger.debug("------------------------------------------------")
            except Exception as e_graph:
                self.logger.warning(f"Error listing graph variables: {e_graph}")
            """
            saver.restore(sess=self.tf_sess, save_path=weights_path)
            #self.logger.info(f"LaneNet model restored successfully (PID: {self.pid}).")

    def detect(self, image_np):
        if not self.tf_sess:
            self.logger.error(f"TensorFlow session is not initialized (PID: {self.pid}). Cannot perform detection.")
            return []

        #self.logger.debug(f"(PID: {self.pid}) Starting detection. Input image shape: {image_np.shape}")
        
        # Log available GPU devices from TensorFlow's perspective
        try:
            if _tf_module: # Check if TensorFlow module is imported
                gpu_devices = _tf_module.config.experimental.list_physical_devices('GPU')
                """
                if gpu_devices:
                    self.logger.debug(f"(PID: {self.pid}) TensorFlow visible Physical GPUs: {gpu_devices}")
                else:
                    self.logger.warning(f"(PID: {self.pid}) TensorFlow sees no Physical GPUs.")
                """
        except Exception as e:
            self.logger.warning(f"(PID: {self.pid}) Error checking for TF GPU devices: {e}")
        image_vis_copy = image_np.copy()
        
        # Resize and normalize image for the network
        target_size = (self.current_cfg_ln.AUG.TRAIN_CROP_SIZE[0], self.current_cfg_ln.AUG.TRAIN_CROP_SIZE[1])
        image_resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
        image_feed = image_resized / 127.5 - 1.0

        try:
            binary_s, instance_s = self.tf_sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [image_feed]}
            )
        except Exception as e:
            self.logger.error(f"(PID: {self.pid}) Error during model inference: {e}", exc_info=True)
            return []
        
        # Postprocess to get lane lines
        try:
            postprocess_result = self.postprocessor.postprocess(
                binary_seg_result=binary_s[0],
                instance_seg_result=instance_s[0],
                source_image=image_vis_copy
            )
        except Exception as e:
            self.logger.error(f"(PID: {self.pid}) Error during postprocessing: {e}", exc_info=True)
            return []
        
        detected_lanes_points = []
        lane_points_from_result = postprocess_result.get('lane_points') if isinstance(postprocess_result, dict) else None

        if lane_points_from_result:
            for lane_line_pts in lane_points_from_result:
                if lane_line_pts is not None and len(lane_line_pts) > 0:
                    # Convert to list of [int, int] points
                    valid_points = []
                    for pt in lane_line_pts:
                        if pt is not None and len(pt) >= 2:
                            try:
                                valid_points.append([int(pt[0]), int(pt[1])])
                            except (ValueError, TypeError):
                                self.logger.warning(f"(PID: {self.pid}) Invalid point format encountered: {pt}. Skipping.")
                                continue
                    
                    if valid_points:
                        # Append as list of lists, not numpy array, for consistency
                        detected_lanes_points.append(valid_points)

        return detected_lanes_points

    def close_session(self):
        if self.tf_sess:
            #self.logger.info(f"Closing TensorFlow session for LaneNetAlgorithm (PID: {self.pid}).")
            self.tf_sess.close()
            self.tf_sess = None

def _get_lanenet_algorithm_instance(process_pid, weights_path_override=None, use_gpu_override=None):
    global _lanenet_algorithm_instance, _current_instance_params
    current_logger = _setup_process_logger(process_pid) # Ensures logger and CFG_LN are ready for this PID

    default_weights_path = os.path.join(LANE_NET_PROJECT_ROOT, 'weights', 'tusimple_lanenet', 'tusimple_lanenet.ckpt')
    final_weights_path = weights_path_override if weights_path_override else default_weights_path
    final_use_gpu = use_gpu_override if use_gpu_override is not None else True

    if (_lanenet_algorithm_instance is None or
            _current_instance_params.get("pid") != process_pid or
            _current_instance_params.get("weights_path") != final_weights_path or
            _current_instance_params.get("use_gpu") != final_use_gpu):

        if _lanenet_algorithm_instance:
            """
            if _current_instance_params.get("pid") == process_pid:
                 current_logger.debug(f"LaneNet parameters changed for PID {process_pid}. Re-initializing LaneNetAlgorithm.")
            else:
                 current_logger.debug(f"LaneNet instance switching from PID {_current_instance_params.get('pid')} to {process_pid}. Re-initializing.")
            """
            _lanenet_algorithm_instance.close_session()

        #current_logger.info(f"Attempting to initialize LaneNetAlgorithm (PID: {process_pid}) with weights: '{final_weights_path}', use_gpu: {final_use_gpu}")
        try:
            _lanenet_algorithm_instance = LaneNetAlgorithm(weights_path=final_weights_path, use_gpu=final_use_gpu, logger=current_logger)
            _current_instance_params["weights_path"] = final_weights_path
            _current_instance_params["use_gpu"] = final_use_gpu
            _current_instance_params["pid"] = process_pid
            #current_logger.debug(f"LaneNetAlgorithm (PID: {process_pid}) initialized successfully.")
        except Exception as e:
            current_logger.error(f"CRITICAL Error initializing LaneNetAlgorithm (PID: {process_pid}): {e}", exc_info=True)
            _lanenet_algorithm_instance = None 
            raise RuntimeError(f"Failed to initialize LaneNetAlgorithm in process {process_pid}: {e}") from e
    
    if _lanenet_algorithm_instance is None:
        msg = f"LaneNetAlgorithm instance is None for PID {process_pid} after initialization attempt(s). Check logs for errors."
        current_logger.error(msg)
        raise RuntimeError(msg)

    return _lanenet_algorithm_instance

def detect_lanes_lanenet(image_np, roi_vertices, cfg_override=None):
    process_pid = os.getpid()
    current_logger = _setup_process_logger(process_pid) # Ensures logger and CFG_LN are ready

    ln_weights_path = None
    ln_use_gpu = None 

    if cfg_override:
        ln_weights_path = getattr(cfg_override, 'LANE_NET_WEIGHTS_PATH', None)
        if hasattr(cfg_override, 'LANE_NET_USE_GPU'): 
            ln_use_gpu = bool(getattr(cfg_override, 'LANE_NET_USE_GPU'))
    
    try:
        model = _get_lanenet_algorithm_instance(
            process_pid,
            weights_path_override=ln_weights_path,
            use_gpu_override=ln_use_gpu
        )
        
        start_time = time.time()
        detected_lanes = model.detect(image_np)
        end_time = time.time()
        #current_logger.info(f"LaneNet detection (PID {process_pid}) on image {image_np.shape} took {end_time - start_time:.4f}s. Found {len(detected_lanes)} lanes.")
        
        return detected_lanes
    except RuntimeError as e: 
        current_logger.error(f"RuntimeError in detect_lanes_lanenet (PID: {process_pid}): {e}", exc_info=False) 
        return [] 
    except Exception as e:
        current_logger.error(f"Unexpected error in detect_lanes_lanenet (PID: {process_pid}): {e}", exc_info=True)
        return []

