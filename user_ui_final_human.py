# Importing All Needed Modules
import asyncio
import threading
import tkinter as tk
from tkinter import ttk
from bleak import BleakScanner, BleakClient
import os
import time
import numpy as np
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration & UUID's
GAIT_LABELS = {0: "Normal Gait", 1: "Equinus Gait Detected", 2: "Standing Right Detected", 3: "Standing Left Detected", 4: "Normal Standing", 5: "Idle"}
NUM_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
SIGNAL_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

# Global Toggles
GRAPH_ENABLED = True
TARGET_SEQUENCE_LENGTH = 50
NOTIFICATION_COOLDOWN = 2.5

# Notification Window
# Creates alert popup when abnormal gait detected
class NotificationWindow(tk.Toplevel):
    def __init__(self, parent, message, on_close_callback, color="#FF453A"): # Added color param
        super().__init__(parent)
        self.callback = on_close_callback
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=color)
        w, h = 320, 80
        sw = self.winfo_screenwidth()
        self.geometry(f"{w}x{h}+{sw-w-20}+40")
        
        header = "⚠️ ALERT" if color == "#FF453A" else "✅ GOOD FORM"
        tk.Label(self, text=header, font=("Arial", 11, "bold"), fg="white", bg=color).pack(pady=(12, 0))
        tk.Label(self, text=message, font=("Arial", 12), fg="white", bg=color).pack(pady=5)
        self.after(2000, self.cleanup)

    def cleanup(self):
        self.callback()
        self.destroy()

# ML Model Creation
# Defines the neural network used for gait classification
class GaitClassifier(nn.Module):
    # Creates neural network layers
    def __init__(self, num_classes):
        super(GaitClassifier, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 7)
        self.conv2 = nn.Conv1d(16, 32, 7)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32*8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    # Defines forward pass
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Main App Code
class BLEApp:
    # Initializes the entire app
    def __init__(self, root):
        self.root = root
        self.root.title("Gait Detector")
        self.root.geometry("1200x900")
        self.root.configure(bg="#121212")
        self.client = None
        self.devices = {}
        self.ml_data_buffer = []
        self.last_alert_time = 0
        self.data_queues = [deque([0]*200, maxlen=200) for _ in range(4)]
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.clf = GaitClassifier(num_classes=len(GAIT_LABELS))
        self.model_ready = False
        self._load_model()
        self._setup_style()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.build_analysis_page()
        if GRAPH_ENABLED:
            self.build_telemetry_page()
            self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.scan()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # Loads pytorch model file
    def _load_model(self):
        path = os.path.join(os.path.dirname(__file__), "models", "gait_classifier_model.pth")
        if os.path.exists(path):
            try:
                self.clf.load_state_dict(torch.load(path, map_location="cpu"))
                self.clf.eval()
                self.model_ready = True
            except Exception as e:
                print("Model load error:", e)

    # Creates visual styling
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#121212", borderwidth=0)
        style.configure("TNotebook.Tab", background="#2c2c2c", foreground="white", padding=[20, 10])
        style.map("TNotebook.Tab", background=[("selected", "#0A84FF")])

    # Creates analysis page
    def build_analysis_page(self):
        self.page1 = tk.Frame(self.notebook, bg="#1c1c1e")
        self.notebook.add(self.page1, text="  ANALYSIS & SENSORS  ")
        
        ctrl = tk.Frame(self.page1, bg="#2c2c2e", pady=10)
        ctrl.pack(fill="x")
        
        self.selected_device = tk.StringVar(value="Select Device...")
        self.dropdown = tk.OptionMenu(ctrl, self.selected_device, "Scanning...")
        self.dropdown.config(width=30, bg="#444", fg="white", highlightthickness=0)
        self.dropdown.pack(side="left", padx=20)

        self.scan_btn = ttk.Button(ctrl, text="Rescan Devices", command=self.scan)
        self.scan_btn.pack(side="left", padx=5)

        self.status_lbl = tk.Label(ctrl, text="● DISCONNECTED", fg="#FF453A", bg="#2c2c2e", font=("Arial", 10, "bold"))
        self.status_lbl.pack(side="right", padx=20)

        self.pred_label = tk.Label(self.page1, text="", font=("Arial", 50, "bold"), fg="#FF453A", bg="#1c1c1e")
        self.pred_label.pack(pady=10)

        self.canvas = tk.Canvas(self.page1, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=40, pady=20)
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill="#4A90E2", outline="white", width=2)
        self.canvas.bind("<Configure>", self.draw_grid)

        # Updated Bottom Display with Labels
        sf = tk.Frame(self.page1, bg="#1c1c1e", pady=10)
        sf.pack(side="bottom", fill="x")
        
        self.s_labels = []
        colors = ["cyan", "magenta", "yellow", "lime"]
        sensor_names = ["FRONT LEFT", "FRONT RIGHT", "BACK LEFT", "BACK RIGHT"]
        
        for i in range(4):
            # Container to stack the Name and the Number
            container = tk.Frame(sf, bg="#1c1c1e")
            container.pack(side="left", expand=True)
            
            # Descriptive Label (Header)
            tk.Label(container, text=sensor_names[i], font=("Arial", 10, "bold"), 
                     fg=colors[i], bg="#1c1c1e").pack()
            
            # Numerical Value
            l = tk.Label(container, text="0.0", font=("Courier", 22, "bold"), 
                         fg=colors[i], bg="#1c1c1e", width=8)
            l.pack()
            self.s_labels.append(l)
    
    # Draws coordinate grid and elongated anatomical insole polygons
    def draw_grid(self, event=None):
        self.canvas.delete("grid_line")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        
        # 1. Draw Darker Coordinate Axes
        self.canvas.create_line(0, h/2, w, h/2, fill="#333333", dash=(4,4), width=2, tags="grid_line")
        self.canvas.create_line(w/2, 0, w/2, h, fill="#333333", dash=(4,4), width=2, tags="grid_line")

        # 2. Elongated Anatomical Insole Shape
        # Adjusted Y-coordinates to be closer to 0.0 and 1.0 to increase length
        base_shape = [
            (0.50, 0.98), # Bottom Heel (Extended)
            (0.32, 0.88), # Inner Heel
            (0.34, 0.60), # Narrower Arch
            (0.18, 0.35), # Wide Ball (Left side)
            (0.35, 0.02), # Top Toe (Extended)
            (0.65, 0.02), # Top Toe (Extended)
            (0.82, 0.35), # Outer Ball
            (0.70, 0.70), # Outer Arch
            (0.68, 0.92), # Outer Heel
        ]

        # Increased height multiplier to 0.92 for maximum vertical coverage
        foot_h = h * 0.92
        foot_w = w * 0.38
        y_offset = h * 0.04 # Reduced offset to allow for longer shape

        for i in [0, 1]:
            center_x = w/4 if i == 0 else 3*w/4
            side_label = "LEFT" if i == 0 else "RIGHT"
            
            points = []
            for px, py in base_shape:
                # Mirror logic: Flip X for the left foot
                actual_x = px if i == 1 else (1 - px)
                
                x = center_x + (actual_x - 0.5) * foot_w
                y = y_offset + (py * foot_h)
                points.extend([x, y])

            # Draw the anatomical polygon with smoothing
            self.canvas.create_polygon(
                points, 
                outline="#444444", 
                fill="", 
                width=3, 
                smooth=True, 
                splinesteps=36, # Higher steps for smoother curves
                tags="grid_line"
            )

            self.canvas.create_text(
                center_x, h - 20, text=side_label, 
                fill="#555555", font=("Arial", 14, "bold"), tags="grid_line"
            )

        # 3. Ensure the Blue Dot stays on top
        self.canvas.tag_raise(self.dot) 
        
    # Creates telemetry graphs page
    def build_telemetry_page(self):
        self.page2 = tk.Frame(self.notebook, bg="black")
        self.notebook.add(self.page2, text="  LIVE GRAPHS  ")

        self.fig = Figure(figsize=(8, 10), facecolor="black")
        self.axes, self.lines = [], []
        colors = ["cyan", "magenta", "yellow", "lime"]
        sensor_titles = ["FRONT LEFT", "BACK LEFT", "FRONT RIGHT", "BACK RIGHT"]

        for i in range(4):
            ax = self.fig.add_subplot(4, 1, i + 1, facecolor="black")
            
            # Set identifying title for each graph
            ax.set_title(sensor_titles[i], color=colors[i], fontsize=9, loc='left', pad=2)
            
            ax.set_ylim(0, 35)
            ax.tick_params(colors="white", labelsize=8)
            ax.grid(True, color="gray", linestyle='--', alpha=0.3)
            for spine in ax.spines.values(): 
                spine.set_color("white")
            
            line, = ax.plot([], [], color=colors[i], lw=2)
            self.axes.append(ax)
            self.lines.append(line)
            
        self.fig.tight_layout(pad=2.0)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.page2)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

    # Detects tab change
    def on_tab_change(self, event):
        if GRAPH_ENABLED and self.notebook.index("current") == 1:
            self.root.after(100, self.refresh_telemetry)

    # Starts BLE scan
    def scan(self):
        self.selected_device.set("Disconnecting & Scanning...")
        self.status_lbl.config(text="● DISCONNECTING...", fg="#FF9F0A")
        menu = self.dropdown["menu"]
        menu.delete(0, "end")
        asyncio.run_coroutine_threadsafe(self._disconnect_and_scan(), self.loop)

    # Disconnects and scans for devices
    async def _disconnect_and_scan(self):
        if self.client and self.client.is_connected:
            try: await self.client.disconnect()
            except: pass
        self.root.after(0, lambda: self.status_lbl.config(text="● DISCONNECTED", fg="#FF453A"))
        devs = await BleakScanner.discover(timeout=4.0)
        self.devices = {(d.name if d.name else d.address): d.address for d in devs}
        self.root.after(0, self._update_dropdown_ui)

    # Updates device dropdown
    def _update_dropdown_ui(self):
        menu = self.dropdown["menu"]
        menu.delete(0, "end")
        if not self.devices: self.selected_device.set("No Devices Found")
        else:
            self.selected_device.set("Select Device...")
            for name in self.devices.keys():
                menu.add_command(label=name, command=lambda n=name: self.on_device_selected(n))

    # Handles device selection
    def on_device_selected(self, name):
        self.selected_device.set(name)
        addr = self.devices.get(name)
        if addr: asyncio.run_coroutine_threadsafe(self._connect_async(addr), self.loop)

    # Connects to BLE device
    async def _connect_async(self, addr):
        try:
            if self.client: await self.client.disconnect()
            self.client = BleakClient(addr, disconnected_callback=self.on_disconnect)
            await self.client.connect()
            self.root.after(0, lambda: self.status_lbl.config(text="● CONNECTED", fg="#30D158"))
            await self.client.start_notify(NUM_UUID, self.ble_handler)
        except Exception as e:
            print("BLE ERROR:", e)
            self.root.after(0, lambda: self.status_lbl.config(text="● CONNECT FAIL", fg="#FF453A"))

    # Handles BLE disconnect
    def on_disconnect(self, client):
        self.root.after(0, lambda: self.status_lbl.config(text="● DISCONNECTED", fg="#FF453A"))

    # Handles incoming BLE data
    def ble_handler(self, sender, data):
        try:
            raw = data.decode("utf-8").split(",")
            if len(raw) < 5: return
            vals = [float(x) for x in raw[1:5]]
            if GRAPH_ENABLED:
                for i, v in enumerate(vals): self.data_queues[i].append(v)
                if self.notebook.index("current") == 1:
                    self.root.after(0, self.refresh_telemetry)
            self.root.after(0, lambda: self.refresh_analysis(vals))
            self.ml_data_buffer.append(vals)
            if len(self.ml_data_buffer) >= TARGET_SEQUENCE_LENGTH:
                snap = np.array(self.ml_data_buffer[:TARGET_SEQUENCE_LENGTH])
                self.ml_data_buffer = self.ml_data_buffer[15:]
                if self.model_ready:
                    threading.Thread(target=self.run_inference, args=(snap,), daemon=True).start()
        except: pass

    # Updates sensor dot and numbers
    def refresh_analysis(self, vals):
        for i, v in enumerate(vals): self.s_labels[i].config(text=f"{v:.1f}")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        dx = ((vals[2]+vals[3]) - (vals[0]+vals[1]))*25
        dy = (((vals[1]+vals[3]) - (vals[0]+vals[2]))*60)
        cx, cy = (w/2)+dx, (h/2)+dy
        cx, cy = max(20, min(w-20, cx)), max(20, min(h-20, cy))
        self.canvas.coords(self.dot, cx-15, cy-15, cx+15, cy+15)

    # Updates live graphs
    def refresh_telemetry(self):
        try:
            for i in range(4):
                y_data = list(self.data_queues[i])
                self.lines[i].set_data(range(len(y_data)), y_data)
                self.axes[i].set_xlim(0, 200)
            self.canvas_plot.draw_idle()
        except: pass

    # Runs ML prediction
    def run_inference(self, data):
        tensor = torch.FloatTensor(data).T.unsqueeze(0)
        with torch.no_grad():
            out = self.clf(tensor)
            _, pred = torch.max(out, 1)
        res = GAIT_LABELS.get(pred.item(), "Normal Gait")
        self.root.after(0, lambda: self.handle_pred_ui(res))

    # Clears alert text
    def clear_ui_alert(self):
        self.pred_label.config(text="")

    # Displays alert and vibrates shoe
    def handle_pred_ui(self, text):
        now = time.time()
        # Define which labels are considered "Positive/Normal"
        positive_labels = ["Normal Gait", "Normal Standing"]
        # Define which labels are "Abnormal/Alert"
        alert_labels = ["Equinus Gait Detected", "Standing Left Detected", "Standing Right Detected"]

        # Handle Idle state first: update text but skip notifications
        if text == "Idle":
            self.pred_label.config(text="IDLE", fg="#8E8E93") # Grey color for idle
            return

        if text in alert_labels or text in positive_labels:
            if now - self.last_alert_time > NOTIFICATION_COOLDOWN:
                self.last_alert_time = now
                
                if text in positive_labels:
                    # Green UI for Success
                    self.pred_label.config(text=text.upper(), fg="#30D158") 
                    NotificationWindow(self.root, text, self.clear_ui_alert, color="#30D158")
                else:
                    # Red UI for Alerts
                    self.pred_label.config(text=text.upper(), fg="#FF453A")
                    NotificationWindow(self.root, text, self.clear_ui_alert, color="#FF453A")
                    # Vibrate only on alerts
                    if self.client and self.client.is_connected:
                        asyncio.run_coroutine_threadsafe(
                            self.client.write_gatt_char(SIGNAL_UUID, b'v'), self.loop
                        )
                        
    # Closes application
    def on_close(self):
        self.root.destroy()
        os._exit(0)

# Main Loop
if __name__ == "__main__":
    root = tk.Tk()
    app = BLEApp(root)
    root.mainloop()