import tkinter as tk
from tkinter import filedialog, ttk, colorchooser, messagebox
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import cvzone
from tracker import Tracker  # Make sure tracker.py is in the same directory


class ObjectDetectionInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Interface")
        self.root.geometry("1000x700")
        
        # Set dark theme colors
        self.bg_dark = "#121212"
        self.card_bg = "#1E1E1E"
        self.accent_color = "#8C52FF"
        self.text_color = "#FFFFFF"
        self.secondary_text = "#BBBBBB"
        
        # Configure the window
        self.root.configure(bg=self.bg_dark)
        
        # Variables
        self.video_path = tk.StringVar()
        self.available_objects = self.load_objects()
        self.default_video = "lib/trimmed.mp4"  # Default video path from mainh.py
        self.object_colors = {}  # Dictionary to store color for each object
        self.resize_timer_id = None  # Initialize the resize timer ID
        
        # Detection variables
        self.detection_running = False
        self.detection_thread = None
        self.detection_paused = False
        self.detection_data = {}
        self.current_frame = None
        self.offset = 8  # From mainh.py
        self.cy1 = 184  # From mainh.py 
        self.cy2 = 209  # From mainh.py
        
        # Model
        self.model = None  # Will load on demand
        self.tracker = Tracker()  # Initialize tracker
        
        # Setup the custom theme
        self.setup_style()
        
        # Create the UI components
        self.create_ui()
        
    def setup_style(self):
        """Setup the modern dark style for the app"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button style
        style.configure('TButton', 
                        font=('Helvetica', 11),
                        background=self.accent_color, 
                        foreground=self.text_color,
                        borderwidth=0,
                        focusthickness=0)
        style.map('TButton', 
                 background=[('active', '#6E40CB')])
        
        # Secondary button style
        style.configure('Secondary.TButton',
                       font=('Helvetica', 11),
                       background='#333333',
                       foreground=self.text_color)
        style.map('Secondary.TButton',
                 background=[('active', '#444444')])
        
        # Color button style
        style.configure('Color.TButton',
                       font=('Helvetica', 10),
                       padding=2)
        
        # Configure checkbutton style
        style.configure('TCheckbutton', 
                        font=('Helvetica', 11), 
                        background=self.card_bg,
                        foreground=self.text_color)
        
        # Configure frame styles
        style.configure('Card.TFrame', 
                        background=self.card_bg)
        
        style.configure('Main.TFrame', 
                        background=self.bg_dark)
        
        # Configure label styles
        style.configure('TLabel', 
                        font=('Helvetica', 11), 
                        background=self.bg_dark,
                        foreground=self.text_color)
        
        style.configure('Header.TLabel', 
                        font=('Helvetica', 18, 'bold'), 
                        background=self.bg_dark,
                        foreground=self.text_color)
        
        style.configure('Card.TLabel', 
                        background=self.card_bg,
                        foreground=self.text_color)
        
        style.configure('Subtitle.TLabel', 
                        font=('Helvetica', 14, 'bold'),
                        background=self.card_bg,
                        foreground=self.text_color)
        
    def load_objects(self):
        """Load objects from the coco.txt file"""
        try:
            with open("lib/coco.txt", "r") as f:
                objects = [line.strip() for line in f.readlines()]
            return objects
        except FileNotFoundError:
            # Return some sample objects if file not found
            return ["person", "car", "bicycle", "dog", "cat", "chair", "bottle"]
    
    def create_ui(self):
        """Create the user interface components"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Object Detection", 
                               style='Header.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Cards container (2 columns layout)
        cards_frame = ttk.Frame(main_frame, style='Main.TFrame')
        cards_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Video selection and preview
        left_card = self.create_rounded_frame(cards_frame, self.card_bg)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video section title
        video_title = ttk.Label(left_card, text="Video Source", 
                               style='Subtitle.TLabel')
        video_title.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Video selection options
        video_frame = ttk.Frame(left_card, style='Card.TFrame')
        video_frame.pack(fill=tk.X, padx=15)
        
        # Custom file selection
        file_btn = ttk.Button(video_frame, text="Select Video File", 
                             command=self.browse_video)
        file_btn.pack(side=tk.LEFT, pady=5)
        
        # Default video option
        default_btn = ttk.Button(video_frame, text="Use Default Video", 
                                style='Secondary.TButton',
                                command=self.use_default_video)
        default_btn.pack(side=tk.LEFT, pady=5, padx=(10, 0))
        
        # Path display 
        path_frame = ttk.Frame(left_card, style='Card.TFrame')
        path_frame.pack(fill=tk.X, padx=15, pady=5)
        
        path_label = ttk.Label(path_frame, text="Selected:", 
                              style='Card.TLabel')
        path_label.pack(side=tk.LEFT)
        
        self.path_value = ttk.Label(path_frame, textvariable=self.video_path, 
                                  style='Card.TLabel', foreground=self.secondary_text,
                                  wraplength=380)
        self.path_value.pack(side=tk.LEFT, padx=(5, 0))
        
        # Video preview
        preview_frame = ttk.Frame(left_card, style='Card.TFrame')
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(10, 15))
        
        self.preview_canvas = tk.Canvas(preview_frame, bg='#0A0A0A', 
                                      highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right column - Detection options
        right_card = self.create_rounded_frame(cards_frame, self.card_bg)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Object selection title
        obj_title = ttk.Label(right_card, text="Detection Objects", 
                             style='Subtitle.TLabel')
        obj_title.pack(pady=(15, 10), padx=15, anchor='w')
        
        # Object selection list
        self.create_object_selection(right_card)
        
        # Stats section
        stats_frame = ttk.Frame(right_card, style='Card.TFrame')
        stats_frame.pack(fill=tk.X, padx=15, pady=(10, 15))
        
        stats_title = ttk.Label(stats_frame, text="Detection Statistics", 
                              style='Card.TLabel', font=('Helvetica', 12, 'bold'))
        stats_title.pack(anchor='w', pady=(5, 10))
        
        # Create a frame for statistics display
        self.stats_display = ttk.Frame(stats_frame, style='Card.TFrame')
        self.stats_display.pack(fill=tk.X, pady=(0, 10))
        
        # We'll populate this with labels dynamically when detection runs
        self.stats_labels = {}
        
        # Control buttons at bottom
        controls_frame = ttk.Frame(main_frame, style='Main.TFrame')
        controls_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Control buttons - start, pause, stop detection
        self.start_btn = ttk.Button(controls_frame, text="Start Detection", 
                                  command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_btn = ttk.Button(controls_frame, text="Pause", 
                                   command=self.toggle_pause_detection,
                                   state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(controls_frame, text="Stop Detection", 
                                  command=self.stop_detection,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
    
    def create_rounded_frame(self, parent, bg_color):
        """Create a frame with rounded corners using a Canvas"""
        frame = ttk.Frame(parent, style='Card.TFrame')
        return frame  # For now, return normal frame as Tkinter doesn't directly support rounded corners
        
    def create_object_selection(self, parent):
        """Create scrollable object selection checkboxes with color pickers"""
        # Container for checkboxes with scrolling
        container = ttk.Frame(parent, style='Card.TFrame')
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # Create canvas for scrolling with dark background
        canvas = tk.Canvas(container, bg=self.card_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header row
        header_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 5), padx=5)
        
        object_header = ttk.Label(header_frame, text="Object", style='Card.TLabel', 
                                 font=('Helvetica', 11, 'bold'))
        object_header.pack(side=tk.LEFT, padx=(25, 0))
        
        color_header = ttk.Label(header_frame, text="Color", style='Card.TLabel',
                                font=('Helvetica', 11, 'bold'))
        color_header.pack(side=tk.RIGHT, padx=(0, 30))
        
        # Create checkbox and color picker for each object
        self.object_vars = {}
        for obj in self.available_objects:
            # Generate a default color (could be randomized)
            default_color = self.generate_color_for_object(obj)
            self.object_colors[obj] = default_color
            
            # Create a frame for each row
            row_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
            row_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Checkbox for object selection
            var = tk.BooleanVar()
            self.object_vars[obj] = var
            cb = ttk.Checkbutton(row_frame, text=obj, variable=var, style='TCheckbutton')
            cb.pack(side=tk.LEFT, padx=5)
            
            # Color indicator and picker button
            color_frame = ttk.Frame(row_frame, style='Card.TFrame')
            color_frame.pack(side=tk.RIGHT, padx=5)
            
            # Create color display
            color_indicator = tk.Canvas(color_frame, width=20, height=20, 
                                      bg=default_color, highlightthickness=1, 
                                      highlightbackground="#555555")
            color_indicator.pack(side=tk.LEFT, padx=(0, 5))
            
            # Store the color indicator reference
            self.object_colors[obj + "_indicator"] = color_indicator
            
            # Color picker button
            color_btn = ttk.Button(color_frame, text="Change", style='Color.TButton',
                                  command=lambda obj=obj: self.pick_color(obj))
            color_btn.pack(side=tk.LEFT)
            
        # Select all / Deselect all buttons
        btn_frame = ttk.Frame(parent, style='Card.TFrame')
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        select_all_btn = ttk.Button(btn_frame, text="Select All", 
                                   style='Secondary.TButton',
                                   command=self.select_all_objects)
        select_all_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        deselect_all_btn = ttk.Button(btn_frame, text="Deselect All", 
                                     style='Secondary.TButton',
                                     command=self.deselect_all_objects)
        deselect_all_btn.pack(side=tk.LEFT)
    
    def generate_color_for_object(self, obj):
        """Generate a unique color for an object based on its name"""
        # Simple hash function to generate color based on object name
        hash_value = sum(ord(c) for c in obj)
        r = (hash_value * 123) % 200 + 55  # Avoid too dark colors
        g = (hash_value * 456) % 200 + 55
        b = (hash_value * 789) % 200 + 55
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def pick_color(self, obj):
        """Open color picker and update the object color"""
        # Open color chooser dialog
        color = colorchooser.askcolor(initialcolor=self.object_colors[obj], 
                                     title=f"Choose color for {obj}")
        
        # Update color if user didn't cancel
        if color[1]:
            self.object_colors[obj] = color[1]
            # Update the color indicator
            self.object_colors[obj + "_indicator"].config(bg=color[1])
    
    def select_all_objects(self):
        """Select all detection objects"""
        for var in self.object_vars.values():
            var.set(True)
    
    def deselect_all_objects(self):
        """Deselect all detection objects"""
        for var in self.object_vars.values():
            var.set(False)
    
    def browse_video(self):
        """Open file dialog to select a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if file_path:
            self.video_path.set(file_path)  # Store the full path
            self.preview_video(file_path)
    
    def use_default_video(self):
        """Use the default video for detection"""
        if os.path.exists(self.default_video):
            self.video_path.set(self.default_video)
            self.preview_video(self.default_video)
        else:
            self.video_path.set("Default video not found")
            messagebox.showwarning("File Not Found", f"Default video not found at: {self.default_video}")
    
    def preview_video(self, video_path):
        """Show preview frame from the video"""
        try:
            if not video_path:
                return
                
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert to RGB and resize for preview
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get canvas dimensions
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Resize frame to fit canvas while preserving aspect ratio
                    h, w = frame.shape[:2]
                    aspect = w / h
                    
                    if canvas_width / canvas_height > aspect:
                        new_h = canvas_height
                        new_w = int(aspect * new_h)
                    else:
                        new_w = canvas_width
                        new_h = int(new_w / aspect)
                    
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Convert to PhotoImage
                self.preview_image = ImageTk.PhotoImage(image=Image.fromarray(frame))
                
                # Clear canvas and show image
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(
                    canvas_width // 2, canvas_height // 2, 
                    image=self.preview_image, anchor=tk.CENTER
                )
                
                # Add a play icon overlay
                self.preview_canvas.create_polygon(
                    canvas_width//2 - 15, canvas_height//2 - 20,
                    canvas_width//2 - 15, canvas_height//2 + 20,
                    canvas_width//2 + 25, canvas_height//2,
                    fill=self.accent_color, outline="#ffffff", width=2
                )
        
        except Exception as e:
            print(f"Error previewing video: {e}")
            messagebox.showerror("Preview Error", f"Could not preview video: {e}")
    
    def get_selected_objects(self):
        """Get the list of selected objects with their colors"""
        selected = {}
        for obj, var in self.object_vars.items():
            if var.get():
                # Convert hex color to BGR tuple (OpenCV format)
                hex_color = self.object_colors[obj].lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                selected[obj] = (b, g, r)  # BGR format for OpenCV
        return selected
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def start_detection(self):
        """Start the object detection process"""
        if self.detection_running:
            return
            
        selected_objects = self.get_selected_objects()
        
        if not selected_objects:
            messagebox.showwarning("Warning", "Please select at least one object to detect")
            return
            
        video_path = self.video_path.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("Warning", "Please select a valid video file")
            return
        
        # Try to load the model
        try:
            if self.model is None:
                # Show loading message
                self.preview_canvas.delete("all")
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                self.preview_canvas.create_text(
                    canvas_width // 2, canvas_height // 2,
                    text="Loading YOLO model...",
                    fill="white", font=("Helvetica", 14)
                )
                self.root.update()
                
                # Load model in the main thread to avoid threading issues
                self.model = YOLO('lib/yolov8s.pt')
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}")
            return
            
        # Start detection in a separate thread
        self.detection_thread = threading.Thread(target=self.run_detection, 
                                                args=(video_path, selected_objects))
        self.detection_thread.daemon = True  # Thread will exit when main app exits
        
        # Update UI
        self.detection_running = True
        self.detection_paused = False
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause")
        self.stop_btn.config(state=tk.NORMAL)
        
        # Initialize counters for all selected classes
        self.object_counts = {obj: 0 for obj in selected_objects}
        
        # Start detection thread
        self.detection_thread.start()
        
        # Create or update statistics labels
        self.update_stats_display()
    
    def update_stats_display(self):
        """Create or update the statistics display"""
        # Clear existing labels
        for widget in self.stats_display.winfo_children():
            widget.destroy()
        
        # Create labels for each selected object
        self.stats_labels = {}
        row = 0
        for obj in self.object_counts.keys():
            # Object name label
            obj_label = ttk.Label(self.stats_display, text=f"{obj}:", 
                                style='Card.TLabel')
            obj_label.grid(row=row, column=0, sticky='w', padx=(5, 10), pady=2)
            
            # Count label
            count_label = ttk.Label(self.stats_display, text="0", 
                                  style='Card.TLabel')
            count_label.grid(row=row, column=1, sticky='w')
            
            # Store reference
            self.stats_labels[obj] = count_label
            row += 1
    
    def update_stats(self):
        """Update the statistics labels with current counts"""
        for obj, count in self.object_counts.items():
            if obj in self.stats_labels:
                self.stats_labels[obj].config(text=str(count))
    
    def toggle_pause_detection(self):
        """Pause or resume the detection process"""
        if not self.detection_running:
            return
            
        self.detection_paused = not self.detection_paused
        
        if self.detection_paused:
            self.pause_btn.config(text="Resume")
        else:
            self.pause_btn.config(text="Pause")
    
    def stop_detection(self):
        """Stop the object detection process"""
        self.detection_running = False
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Wait for thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(0.5)  # Give it 0.5 seconds to finish
    
    def run_detection(self, video_path, selected_objects):
        """Run object detection on the video"""
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            
            # Get class indices - match class names with their indices in the model
            with open("lib/coco.txt", "r") as f:
                class_list = [line.strip() for line in f.readlines()]
            
            # Get frame processing dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Processing logic from mainh.py
            count = 0
            self.tracker = Tracker()  # Reset tracker
            
            while self.detection_running:
                if self.detection_paused:
                    time.sleep(0.1)  # Sleep briefly when paused
                    continue
                    
                ret, frame = cap.read()
                if not ret:
                    # End of video, loop back to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                count += 1
                if count % 3 != 0:  # Process every third frame as in mainh.py
                    continue
                    
                # Resize frame for processing
                process_frame = cv2.resize(frame, (1020, 500))
                
                # Initialize classes dict for this frame
                classes = {}
                for obj_name in selected_objects.keys():
                    classes[obj_name] = {
                        'color': selected_objects[obj_name],
                        'coords': [],
                        'bbox': [],
                        'count': 0
                    }
                
                # Predict objects using YOLO
                results = self.model.predict(process_frame)
                detections = results[0].boxes.data
                px = pd.DataFrame(detections).astype("float")
                
                # Process detections
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])  # Class index
                    
                    # Only process selected objects
                    if d < len(class_list):
                        obj_name = class_list[d]
                        if obj_name in selected_objects:
                            classes[obj_name]['coords'].append([x1, y1, x2, y2])
                
                # Update tracker for each object type
                for key in classes:
                    classes[key]['bbox'] = self.tracker.update(classes[key]['coords'])
                
                # Check each detected object
                for class_name, value in classes.items():
                    for bbox in value['bbox']:
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)
                        # Count objects crossing the line
                        if (cy > self.cy1 - self.offset) and (cy < self.cy1 + self.offset):
                            classes[class_name]['count'] += 1
                            # Update the global count
                            self.object_counts[class_name] += 1
                        
                        # Draw bounding box and label
                        color = classes[class_name]['color']
                        cv2.rectangle(process_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(process_frame, class_name, (bbox[0], bbox[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw detection line
                # cv2.line(process_frame, (0, self.cy1), (1020, self.cy1), (255, 0, 255), 2)
                
                # Convert frame for display
                display_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                
                # Resize for canvas
                if canvas_width > 1 and canvas_height > 1:
                    h, w = display_frame.shape[:2]
                    aspect = w / h
                    
                    if canvas_width / canvas_height > aspect:
                        new_h = canvas_height
                        new_w = int(aspect * new_h)
                    else:
                        new_w = canvas_width
                        new_h = int(new_w / aspect)
                    
                    display_frame = cv2.resize(display_frame, (new_w, new_h))
                
                # Update UI with current frame
                self.update_preview(display_frame)
                
                # Update statistics display
                self.root.after(10, self.update_stats)
                
                # Processing delay to manage frame rate
                time.sleep(0.03)
                
            # Clean up
            cap.release()
            
        except Exception as e:
            print(f"Detection error: {e}")
            messagebox.showerror("Detection Error", f"An error occurred during detection: {e}")
        
        finally:
            # Ensure UI is reset
            self.root.after(0, self.stop_detection)
    
    def update_preview(self, frame):
        """Update the preview canvas with a new frame"""
        try:
            # Create PhotoImage from frame
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            
            # Get canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Update canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            
            # Keep a reference to prevent garbage collection
            self.current_frame = photo
            
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def handle_resize(self, event):
        """Handle window resize events with proper timer handling"""
        # Cancel previous timer if it exists
        if self.resize_timer_id is not None:
            try:
                self.root.after_cancel(self.resize_timer_id)
            except ValueError:
                # If timer ID is not valid, just ignore the error
                pass
        
        # Set a new timer
        if hasattr(self, 'video_path') and self.video_path.get():
            self.resize_timer_id = self.root.after(100, lambda: self.preview_video(self.video_path.get()))
    
    def save_selection(self):
        """Save the current selection (video and objects with colors)"""
        selected_objects = self.get_selected_objects()
        
        if not selected_objects:
            self.show_message("Warning", "Please select at least one object to detect")
            return
            
        if not self.video_path.get():
            self.show_message("Warning", "Please select a video file")
            return
        
        # Display the selection (in a real app, you would save or process this data)
        selection_info = f"Video: {self.video_path.get()}\n\nSelected Objects:"
        for obj, color in selected_objects.items():
            # Convert BGR back to hex for display
            b, g, r = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            selection_info += f"\n• {obj} ({hex_color})"
        
        self.show_message("Selection Saved", selection_info)
    
    def show_message(self, title, message):
        """Show a custom styled message box"""
        # For now, use the standard messagebox
        messagebox.showinfo(title, message)


if __name__ == "__main__":
    # Check for required files
        
    # Start the app
    root = tk.Tk()
    app = ObjectDetectionInterface(root)
    
    # Bind the resize event to our improved resize handler
    root.bind("<Configure>", app.handle_resize)
    
    root.mainloop()