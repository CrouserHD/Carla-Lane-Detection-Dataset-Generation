import customtkinter
import os
import sys
import threading
import queue
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import get_config, get_available_algorithms, update_setting
from src.lane_comparison.pipeline import LaneComparisonPipeline


class ScrollableCheckboxFrame(customtkinter.CTkScrollableFrame):
    """A scrollable frame for checkboxes."""
    def __init__(self, master, title, values, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.title = title
        self.values = values
        self.checkboxes = {}

        self.title_label = customtkinter.CTkLabel(self, text=self.title, font=customtkinter.CTkFont(size=14, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        for i, (name, display_name, is_active) in enumerate(self.values):
            checkbox = customtkinter.CTkCheckBox(self, text=display_name)
            checkbox.grid(row=i + 1, column=0, padx=10, pady=2, sticky="w")
            if is_active:
                checkbox.select()
            self.checkboxes[name] = checkbox

    def get_selected(self):
        """Returns a list of names of the selected checkboxes."""
        return [name for name, cb in self.checkboxes.items() if cb.get() == 1]


class ScrollableImageFrame(customtkinter.CTkScrollableFrame):
    """A scrollable frame to display image thumbnails."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.image_widgets = []
        self.image_paths = []
        self.next_row = 0
        self.next_col = 0

    def add_image(self, image_path):
        """Adds a new image thumbnail to the grid."""
        if image_path in self.image_paths:
            return # Avoid duplicates
            
        try:
            img = Image.open(image_path)
            img.thumbnail((150, 150)) # Create a thumbnail
            ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
            
            label = customtkinter.CTkLabel(self, image=ctk_img, text="")
            label.grid(row=self.next_row, column=self.next_col, padx=5, pady=5)
            
            self.image_widgets.append(label)
            self.image_paths.append(image_path)
            
            self.next_col += 1
            if self.next_col >= 4:
                self.next_col = 0
                self.next_row += 1
        except Exception as e:
            print(f"Error adding image to gallery: {e}")

    def clear_images(self):
        """Removes all images from the frame."""
        for widget in self.image_widgets:
            widget.destroy()
        self.image_widgets = []
        self.image_paths = []
        self.next_row = 0
        self.next_col = 0


class VideoPlayer(customtkinter.CTkFrame):
    """A custom video player widget."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # configure internal grid: video display and controls
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        # display area for video frame
        self.player_label = customtkinter.CTkLabel(self, text="No video loaded.")
        self.player_label.grid(row=0, column=0, sticky="nsew")
        # control buttons and seek slider
        ctrl_frame = customtkinter.CTkFrame(self)
        ctrl_frame.grid(row=1, column=0, sticky="ew", pady=(5,0))
        self.play_btn = customtkinter.CTkButton(ctrl_frame, text="Play", command=self.play)
        self.play_btn.pack(side="left", padx=5)
        self.pause_btn = customtkinter.CTkButton(ctrl_frame, text="Pause", command=self.pause)
        self.pause_btn.pack(side="left", padx=5)
        # external open button
        from src.config import get_config
        import os
        self.open_btn = customtkinter.CTkButton(ctrl_frame, text="Open Video", command=lambda: os.startfile(get_config("PATHS", "OUTPUT_VIDEO_PATH")))
        self.open_btn.pack(side="left", padx=5)
        # will configure slider range when loading video
        self.seek_slider = customtkinter.CTkSlider(ctrl_frame, from_=0, to=0, command=self.on_seek)
        self.seek_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.cap = None
        self.playing = False
        self.after_id = None
        self.total_frames = 0
        self.fps = 30  # fallback fps if undetermined
        self.live_update_mode = False

    def set_ui_for_processing(self, is_processing):
        """Disable/Enable controls based on processing state."""
        self.live_update_mode = is_processing
        if is_processing:
            self.player_label.configure(text="Processing live...")
            self.play_btn.configure(state="disabled")
            self.pause_btn.configure(state="disabled")
            self.seek_slider.configure(state="disabled")
        else:
            self.player_label.configure(text="No video loaded.")
            self.play_btn.configure(state="normal")
            self.pause_btn.configure(state="normal")
            self.seek_slider.configure(state="normal")


    def update_live_frame(self, frame_cv):
        """Displays a single frame from the live processing pipeline."""
        if not self.live_update_mode:
            return

        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        widget_w = self.player_label.winfo_width()
        widget_h = self.player_label.winfo_height()
        if widget_w > 1 and widget_h > 1:
            img.thumbnail((widget_w, widget_h), Image.Resampling.LANCZOS)

        ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self.player_label.configure(image=ctk_img, text="")


    def load(self, video_path):
        if self.live_update_mode:
            return # Don't load a file during live processing

        if not os.path.exists(video_path):
            self.player_label.configure(text=f"Error: Video not found\n{video_path}")
            return
        try:
            self.cap = cv2.VideoCapture(video_path)
            # get total frames and fps
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.fps
            # configure slider
            self.seek_slider.configure(to=max(0, self.total_frames-1))
            self.seek_slider.set(0)
            self.playing = False
            # display first frame
            self.on_seek(0)
        except Exception as e:
            self.player_label.configure(text=f"Error loading video:\n{e}")

    def show_frame(self):
        """Displays next frame if playing; stops and resets at end of video."""
        if not (self.cap and self.playing):
            return
        ret, frame = self.cap.read()
        if not ret:
            # End of video: stop playback and reset to first frame
            self.playing = False
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        # Update slider to current frame
        current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.seek_slider.set(current_idx)
        # Convert and resize for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        widget_w = self.player_label.winfo_width()
        widget_h = self.player_label.winfo_height()
        if widget_w > 1 and widget_h > 1:
            img.thumbnail((widget_w, widget_h), Image.Resampling.LANCZOS)
        ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self.player_label.configure(image=ctk_img, text="")
        # Schedule next frame based on stored fps
        delay = int(1000 / self.fps)
        self.after_id = self.after(delay, self.show_frame)


    def play(self):
        if self.cap:
            self.playing = True
            self.show_frame()

    def pause(self):
        self.playing = False
        if self.after_id:
            self.after_cancel(self.after_id)
    def on_seek(self, val):
        """Seek to a specific frame index in the video."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(val))
            # show the frame at this position
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                widget_w = self.player_label.winfo_width()
                widget_h = self.player_label.winfo_height()
                if widget_w>1 and widget_h>1:
                    img.thumbnail((widget_w, widget_h), Image.Resampling.LANCZOS)
                ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                self.player_label.configure(image=ctk_img, text="")


class QueueIO:
    def __init__(self, q):
        self.q = q
    def write(self, s):
        self.q.put(s)
    def flush(self):
        pass

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Lane Detection Comparison Tool")
        self.geometry("1200x800")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.pipeline_thread = None
        self.pipeline_instance = None
        self.result_image_queue = queue.Queue()
        self.live_frame_queue = queue.Queue()
        # store per-frame F1 metrics
        self.metrics_data = []

        # --- Tab View ---
        self.tab_view = customtkinter.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.tab_view.add("Algorithm Comparison")
        self.tab_view.add("Results & Metrics")
        self.tab_view.add("Settings")

        # --- Configure Tabs ---
        self.setup_comparison_tab()
        self.setup_results_tab()
        self.setup_settings_tab()

        # --- Redirect stdout for logging ---
        self.log_queue = queue.Queue()
        sys.stdout = QueueIO(self.log_queue)
        self.after(100, self.process_log_queue)
        self.after(50, self.process_live_frame_queue) # Poll for live frames

    def setup_comparison_tab(self):
        """Configures the main tab for running comparisons."""
        tab = self.tab_view.tab("Algorithm Comparison")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2) # Give more weight to the video player column
        tab.grid_rowconfigure(1, weight=1)

        # --- Left Control Frame ---
        control_frame = customtkinter.CTkFrame(tab)
        control_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        control_frame.grid_rowconfigure(3, weight=1) # Make algo frame scrollable
        control_frame.grid_rowconfigure(4, weight=0) # Legend frame has fixed size

        # Input Source
        input_label = customtkinter.CTkLabel(control_frame, text="Input Source", font=customtkinter.CTkFont(size=14, weight="bold"))
        input_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.source_path_entry = customtkinter.CTkEntry(control_frame, width=250)
        self.source_path_entry.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.source_path_entry.insert(0, get_config("PATHS", "VIDEO_PATH"))

        browse_button = customtkinter.CTkButton(control_frame, text="Browse...", command=self.browse_source)
        browse_button.grid(row=1, column=1, padx=10, pady=5)

        # Algorithm Selection
        available_algorithms = get_available_algorithms()
        algo_values = [(algo.name, algo.display_name, algo.active_by_default) for algo in available_algorithms]
        
        self.algo_frame = ScrollableCheckboxFrame(control_frame, title="Select Algorithms", values=algo_values)
        self.algo_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(10,5), sticky="nsew")


        # --- Algorithm Legend ---
        self.legend_frame = customtkinter.CTkFrame(control_frame)
        self.legend_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.update_legend()


        # --- Right Player and Log Frame ---
        right_frame = customtkinter.CTkFrame(tab)
        right_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=3) # Player gets more space
        right_frame.grid_rowconfigure(1, weight=1) # Log gets less
        right_frame.grid_columnconfigure(0, weight=1)

        # Video Player
        self.video_player = VideoPlayer(right_frame)
        self.video_player.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Log Console
        self.log_textbox = customtkinter.CTkTextbox(right_frame, wrap="word")
        self.log_textbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.log_textbox.configure(state="disabled")

        # --- Bottom Control Frame ---
        bottom_frame = customtkinter.CTkFrame(tab)
        bottom_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        bottom_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.run_button = customtkinter.CTkButton(bottom_frame, text="Run Comparison", command=self.start_comparison_thread)
        self.run_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = customtkinter.CTkButton(bottom_frame, text="Stop", command=self.stop_comparison, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)

        self.progress_bar = customtkinter.CTkProgressBar(bottom_frame)
        self.progress_bar.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        self.progress_bar.set(0)


    def setup_results_tab(self):
        """Configures the tab for displaying detailed results."""
        tab = self.tab_view.tab("Results & Metrics")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        # metrics plot canvas
        self.metrics_fig = Figure(figsize=(5,2), tight_layout=True)
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=tab)
        self.metrics_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=(10,0), sticky="nsew")
        
        # Metrics summary text box
        self.metrics_table = customtkinter.CTkTextbox(tab, height=150, wrap="none")
        self.metrics_table.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.metrics_table.configure(state="disabled")

        # results image gallery
        self.results_gallery = ScrollableImageFrame(tab)
        self.results_gallery.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        tab.grid_rowconfigure(2, weight=1) # give gallery space

    def setup_settings_tab(self):
        """Configures the tab for application settings."""
        tab = self.tab_view.tab("Settings")
        tab.grid_columnconfigure(0, weight=1)

        # --- Number of Frames ---
        frames_label = customtkinter.CTkLabel(tab, text="Number of Frames to Process (0 for all)")
        frames_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.frames_entry = customtkinter.CTkEntry(tab)
        self.frames_entry.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.frames_entry.insert(0, str(get_config("SETTINGS", "PROCESS_NUM_IMAGES")))

        # --- Metrics Threshold ---
        threshold_label = customtkinter.CTkLabel(tab, text="Lane Metrics Pixel Threshold")
        threshold_label.grid(row=2, column=0, padx=20, pady=(20, 5), sticky="w")

        self.threshold_slider = customtkinter.CTkSlider(tab, from_=1, to=50, number_of_steps=49)
        self.threshold_slider.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        self.threshold_slider.set(get_config("SETTINGS", "LANE_METRICS_THRESHOLD_PX"))
        
        self.threshold_value_label = customtkinter.CTkLabel(tab, text=f"{self.threshold_slider.get():.0f} px")
        self.threshold_value_label.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.threshold_slider.configure(command=lambda v: self.threshold_value_label.configure(text=f"{v:.0f} px"))

        # --- Apply Button ---
        apply_button = customtkinter.CTkButton(tab, text="Apply Settings", command=self.apply_settings)
        apply_button.grid(row=4, column=0, columnspan=2, padx=20, pady=20)
        # additional settings
        skip_cb = customtkinter.CTkCheckBox(tab, text="Skip images without GT")
        skip_cb.grid(row=5, column=0, padx=20, pady=(10,5), sticky="w")
        if get_config("SETTINGS", "SKIP_IMAGES_WITHOUT_GT"):
            skip_cb.select()
        self.skip_cb = skip_cb
        fps_label = customtkinter.CTkLabel(tab, text="Video FPS")
        fps_label.grid(row=6, column=0, padx=20, pady=(10,5), sticky="w")
        self.fps_entry = customtkinter.CTkEntry(tab)
        self.fps_entry.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        self.fps_entry.insert(0, str(get_config("SETTINGS", "VIDEO_FPS")))
        resize_label = customtkinter.CTkLabel(tab, text="Resize Processing Factor")
        resize_label.grid(row=8, column=0, padx=20, pady=(10,5), sticky="w")
        self.resize_entry = customtkinter.CTkEntry(tab)
        self.resize_entry.grid(row=9, column=0, padx=20, pady=5, sticky="ew")
        self.resize_entry.insert(0, str(get_config("SETTINGS", "RESIZE_PROCESSING_FACTOR")))
        # --- ROI Configuration ---
        roi_xs_label = customtkinter.CTkLabel(tab, text="ROI X Start Ratio (0.0-1.0)")
        roi_xs_label.grid(row=10, column=0, padx=20, pady=(10,5), sticky="w")
        self.roi_xs_entry = customtkinter.CTkEntry(tab)
        self.roi_xs_entry.grid(row=11, column=0, padx=20, pady=5, sticky="ew")
        self.roi_xs_entry.insert(0, str(get_config("SETTINGS", "ROI_X_START_RATIO")))
        roi_y_label = customtkinter.CTkLabel(tab, text="ROI Y Top Ratio (0.0-1.0)")
        roi_y_label.grid(row=12, column=0, padx=20, pady=(10,5), sticky="w")
        self.roi_y_entry = customtkinter.CTkEntry(tab)
        self.roi_y_entry.grid(row=13, column=0, padx=20, pady=5, sticky="ew")
        self.roi_y_entry.insert(0, str(get_config("SETTINGS", "ROI_Y_RATIO")))
        roi_xe_label = customtkinter.CTkLabel(tab, text="ROI X End Ratio (0.0-1.0)")
        roi_xe_label.grid(row=14, column=0, padx=20, pady=(10,5), sticky="w")
        self.roi_xe_entry = customtkinter.CTkEntry(tab)
        self.roi_xe_entry.grid(row=15, column=0, padx=20, pady=5, sticky="ew")
        self.roi_xe_entry.insert(0, str(get_config("SETTINGS", "ROI_X_END_RATIO")))
        roi_be_label = customtkinter.CTkLabel(tab, text="ROI Y Bottom Ratio (0.0-1.0)")
        roi_be_label.grid(row=16, column=0, padx=20, pady=(10,5), sticky="w")
        self.roi_be_entry = customtkinter.CTkEntry(tab)
        self.roi_be_entry.grid(row=17, column=0, padx=20, pady=5, sticky="ew")
        self.roi_be_entry.insert(0, str(get_config("SETTINGS", "ROI_Y_END_RATIO")))

        # Add Draw ROI button to allow interactive ROI selection on first frame
        # Launch ROI drawing in a separate thread to avoid blocking Tkinter mainloop
        draw_roi_btn = customtkinter.CTkButton(tab, text="Draw ROI on First Frame", command=lambda: threading.Thread(target=self.draw_roi, daemon=True).start())
        draw_roi_btn.grid(row=18, column=0, columnspan=2, padx=20, pady=(10,10))

    def draw_roi(self):
        """Allows user to draw ROI polygon by clicking four points on the first frame."""
        video_path = self.source_path_entry.get()
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_idx = 0
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame for ROI selection.")
            cap.release()
            return
        if not ret:
            print("Error: Could not read first frame for ROI selection.")
            return
        h_frame, w_frame = frame.shape[:2]
        window = "Select ROI Points"
        # initial copy and instruction overlay
        img_copy = frame.copy()
        instruction = "Click 4 points: top left, top right, bottom right, bottom left. Arrow keys to change frame. Press 'q' to cancel. Press 'r' to restart selection."
        cv2.putText(img_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        points = []
        # mouse callback
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                if len(points) > 1:
                    cv2.line(img_copy, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow(window, img_copy)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, click_event)
        # event loop
        while True:
            cv2.imshow(window, img_copy)
            # use waitKeyEx to capture arrow keys
            key = cv2.waitKeyEx(0)
            # define key sets for navigation
            right_keys = {ord('d'), 83, 2555904, 65363}
            left_keys  = {ord('a'), 81, 2424832, 65361}
            if key in right_keys and current_idx < total_frames - 1:
                current_idx += 1
            elif key in left_keys and current_idx > 0:
                current_idx -= 1
            elif key == ord('q'):
                break
            elif key == ord('r'):
                # restart selection on current frame
                points.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                img_copy = frame.copy()
                cv2.putText(img_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue
            # reload frame if navigated
            if key in right_keys.union(left_keys):
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                img_copy = frame.copy()
                points.clear()
                cv2.putText(img_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue
            # finish on 4 points
            if len(points) == 4:
                break
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        cap.release()
        if len(points) == 4:
            # normalize points
            norm_pts = [(x / w_frame, y / h_frame) for x, y in points]
            print("Selected ROI points (normalized):", norm_pts)
            # update rectangular ROI entries from polygon bounds
            xs_val = min(pt[0] for pt in norm_pts)
            ys_val = min(pt[1] for pt in norm_pts)
            xe_val = max(pt[0] for pt in norm_pts)
            ye_val = max(pt[1] for pt in norm_pts)
            self.roi_xs_entry.delete(0, "end")
            self.roi_xs_entry.insert(0, f"{xs_val:.4f}")
            self.roi_y_entry.delete(0, "end")
            self.roi_y_entry.insert(0, f"{ys_val:.4f}")
            self.roi_xe_entry.delete(0, "end")
            self.roi_xe_entry.insert(0, f"{xe_val:.4f}")
            self.roi_be_entry.delete(0, "end")
            self.roi_be_entry.insert(0, f"{ye_val:.4f}")
            # update config settings immediately
            update_setting("ROI_X_START_RATIO", xs_val)
            update_setting("ROI_Y_RATIO", ys_val)
            update_setting("ROI_X_END_RATIO", xe_val)
            update_setting("ROI_Y_END_RATIO", ye_val)
        else:
            print("ROI point selection canceled or incomplete.")

    def apply_settings(self):
        """Applies the settings from the Settings tab."""
        try:
            num_frames = int(self.frames_entry.get())
            update_setting("PROCESS_NUM_IMAGES", num_frames)
        except ValueError:
            print("Error: Invalid number for frames. Please enter an integer.")
        
        threshold = int(self.threshold_slider.get())
        update_setting("LANE_METRICS_THRESHOLD_PX", threshold)
        # skip without GT
        update_setting("SKIP_IMAGES_WITHOUT_GT", bool(self.skip_cb.get()))
        # video fps
        try:
            fps = float(self.fps_entry.get())
            update_setting("VIDEO_FPS", fps)
        except ValueError:
            print("Error: invalid FPS value")
        # resize factor
        try:
            rf = float(self.resize_entry.get())
            update_setting("RESIZE_PROCESSING_FACTOR", rf)
        except ValueError:
            print("Error: invalid resize factor")
        # ROI settings
        try:
            xs = float(self.roi_xs_entry.get())
            update_setting("ROI_X_START_RATIO", xs)
        except ValueError:
            print("Error: invalid ROI X start ratio")
        try:
            ys = float(self.roi_y_entry.get())
            update_setting("ROI_Y_RATIO", ys)
        except ValueError:
            print("Error: invalid ROI Y top ratio")
        try:
            xe = float(self.roi_xe_entry.get())
            update_setting("ROI_X_END_RATIO", xe)
        except ValueError:
            print("Error: invalid ROI X end ratio")
        try:
            ye = float(self.roi_be_entry.get())
            update_setting("ROI_Y_END_RATIO", ye)
        except ValueError:
            print("Error: invalid ROI Y bottom ratio")
        print("Settings applied successfully.")

    def update_legend(self):
        """Updates the algorithm color legend."""
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
            
        legend_label = customtkinter.CTkLabel(self.legend_frame, text="Algorithm Legend:", font=customtkinter.CTkFont(weight="bold"))
        legend_label.pack(anchor="w", padx=5, pady=(5,2))

        selected_algos = self.algo_frame.get_selected()
        for name in selected_algos:
            algo_config = next((a for a in get_available_algorithms() if a.name == name), None)
            if algo_config:
                # Create a small frame for each legend item
                item_frame = customtkinter.CTkFrame(self.legend_frame, fg_color="transparent")
                item_frame.pack(anchor="w", fill="x", padx=5)

                color_box = customtkinter.CTkFrame(item_frame, width=15, height=15, fg_color=self.rgb_to_hex(algo_config.color), border_width=0)
                color_box.pack(side="left", padx=(0, 5), pady=2)
                
                label = customtkinter.CTkLabel(item_frame, text=algo_config.display_name)
                label.pack(side="left", padx=0, pady=2)

    def rgb_to_hex(self, rgb):
        """Converts an (R, G, B) tuple to #RRGGBB hex format."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def browse_source(self):
        """Opens a dialog to select a video file or image directory."""
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if path:
            self.source_path_entry.delete(0, "end")
            self.source_path_entry.insert(0, path)

    def process_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_textbox.configure(state="normal")
                self.log_textbox.insert("end", line)
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)

    def process_live_frame_queue(self):
        """Processes the queue of live frames from the pipeline."""
        try:
            while not self.live_frame_queue.empty():
                frame, metrics = self.live_frame_queue.get_nowait()
                self.video_player.update_live_frame(frame)
                self.metrics_data.append(metrics)
        except queue.Empty:
            pass
        self.after(50, self.process_live_frame_queue)


    def process_image_queue(self):
        """Processes the queue of generated result images."""
        try:
            while True:
                image_path = self.result_image_queue.get_nowait()
                self.results_gallery.add_image(image_path)
        except queue.Empty:
            pass
        self.after(200, self.process_image_queue)

    def update_progress(self, value):
        self.progress_bar.set(value)

    def start_comparison_thread(self):
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress_bar.set(0)
        # reset metrics and results
        self.metrics_data = []
        self.results_gallery.clear_images()
        self.metrics_table.configure(state="normal")
        self.metrics_table.delete("1.0", "end")
        self.metrics_table.configure(state="disabled")
        self.metrics_ax.clear()
        self.metrics_canvas.draw()
        
        self.update_legend() # Update legend for the new run
        self.video_player.set_ui_for_processing(True)
        
        selected_algos = self.algo_frame.get_selected()
        video_path = self.source_path_entry.get()

        if not selected_algos:
            print("Error: No algorithms selected.")
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return

        self.pipeline_instance = LaneComparisonPipeline(
            video_path=video_path,
            selected_algo_names=selected_algos,
            progress_callback=self.update_progress,
            log_callback=lambda msg: self.log_queue.put(msg + '\n'),
            image_callback=lambda path: self.result_image_queue.put(path),
            frame_callback=lambda frame, metrics: self.live_frame_queue.put((frame, metrics))
        )
        
        self.pipeline_thread = threading.Thread(target=self.run_comparison, daemon=True)
        self.pipeline_thread.start()


    def stop_comparison(self):
        if self.pipeline_instance:
            print("Requesting to stop pipeline...")
            self.pipeline_instance.request_stop()
        self.stop_button.configure(state="disabled")


    def run_comparison(self):
        """The main logic for running the lane comparison pipeline."""
        if self.pipeline_instance:
            self.pipeline_instance.run()
            # Schedule the completion tasks to run on the main thread
            self.after(0, self.on_comparison_complete)

    def on_comparison_complete(self):
        """Tasks to run on the main thread after the pipeline finishes."""
        print("Pipeline finished. Finalizing results...")
        self.video_player.set_ui_for_processing(False)
        
        output_video_path = get_config("PATHS", "OUTPUT_VIDEO_PATH")
        if os.path.exists(output_video_path):
            print(f"Loading result video: {output_video_path}")
            self.video_player.load(output_video_path)
            # Do not auto-play; user can control playback
        else:
            print(f"Error: Output video not found at {output_video_path}")

        # Plot final metrics and display summary
        self.plot_metrics()
        self.display_metrics_summary()

        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        print("Comparison complete.")

    def plot_metrics(self):
        """Plots the collected F1 scores for each algorithm."""
        self.metrics_ax.clear()
        
        if not self.metrics_data:
            self.metrics_ax.set_title("No Metrics Data Available")
            self.metrics_canvas.draw()
            return

        selected_algos = self.algo_frame.get_selected()
        algos_config = {a.name: a for a in get_available_algorithms()}

        # Group metrics by algorithm
        metrics_by_algo = {name: [] for name in selected_algos}
        frame_indices = []

        for frame_metric in self.metrics_data:
            frame_indices.append(frame_metric['frame_idx'])
            for algo_name in selected_algos:
                # The metric for an algo might not exist if it failed or was skipped
                metric = frame_metric.get(algo_name, {}).get('f1', None)
                metrics_by_algo[algo_name].append(metric)
        
        # Use a set to get unique frame indices and then sort them
        unique_frames = sorted(list(set(frame_indices)))

        for algo_name, f1_scores in metrics_by_algo.items():
            config = algos_config.get(algo_name)
            if config:
                # Filter out None values for plotting
                plot_frames = [unique_frames[i] for i, score in enumerate(f1_scores) if score is not None]
                plot_scores = [score for score in f1_scores if score is not None]
                
                if plot_frames:
                    self.metrics_ax.plot(
                        plot_frames,
                        plot_scores,
                        label=config.display_name,
                        color=self.rgb_to_hex(config.color)
                    )

        self.metrics_ax.set_title("F1 Score per Frame")
        self.metrics_ax.set_xlabel("Frame Index")
        self.metrics_ax.set_ylabel("F1 Score")
        self.metrics_ax.set_ylim(0, 1.05)
        self.metrics_ax.legend()
        self.metrics_ax.grid(True, linestyle='--', alpha=0.6)
        self.metrics_canvas.draw()

    def display_metrics_summary(self):
        """Calculates and displays summary statistics in the text box."""
        self.metrics_table.configure(state="normal")
        self.metrics_table.delete("1.0", "end")

        if not self.metrics_data:
            self.metrics_table.insert("end", "No metrics data available.")
            self.metrics_table.configure(state="disabled")
            return

        selected_algos = self.algo_frame.get_selected()
        header = f"{'Algorithm':<30}{'Avg F1':>10}{'Min F1':>10}{'Max F1':>10}\n"
        separator = "-" * 60 + "\n"
        
        summary_text = header + separator
        
        # Group metrics by algorithm
        metrics_by_algo = {name: [] for name in selected_algos}
        for frame_metric in self.metrics_data:
            for algo_name in selected_algos:
                metric = frame_metric.get(algo_name, {}).get('f1', None)
                if metric is not None:
                    metrics_by_algo[algo_name].append(metric)

        for algo_name in selected_algos:
            scores = metrics_by_algo.get(algo_name, [])
            if scores:
                avg_f1 = sum(scores) / len(scores)
                min_f1 = min(scores)
                max_f1 = max(scores)
                display_name = next((a.display_name for a in get_available_algorithms() if a.name == algo_name), algo_name)
                summary_text += f"{display_name:<30}{avg_f1:>10.3f}{min_f1:>10.3f}{max_f1:>10.3f}\n"
            else:
                display_name = next((a.display_name for a in get_available_algorithms() if a.name == algo_name), algo_name)
                summary_text += f"{display_name:<30}{'N/A':>10}{'N/A':>10}{'N/A':>10}\n"

        self.metrics_table.insert("end", summary_text)
        self.metrics_table.configure(state="disabled")


if __name__ == "__main__":
    # Set a specific theme
    customtkinter.set_appearance_mode("Dark")
    customtkinter.set_default_color_theme("blue")
    
    app = App()
    app.mainloop()
