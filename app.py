"""
Industrial Quality Control - Smart Factory Dashboard (Windows Local App)
Full Featured & Production Ready:
1. Product Profiles (Capsule, Metal Nut, Bottle, General)
2. Crash-Proof Hybrid Vision (YOLO11 ONNX + Color Anomaly Filter + Squeeze/Dent Analysis + Capsule Dimensions)
3. Safe Image Reader (Supports Unicode/Arabic paths on Windows)
4. Automatic Industrial Stream Tracker (Counts, Logs, and Archives during Live Camera & Video Stream)
5. Full Factory Telemetry: Production Counters, Pneumatic Ejector Pulse, Emergency Stop, Audio Alarms, CSV Logs & Reports
"""

import os
import cv2
import sys
import time
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from src.hybrid_inspector import HybridQualityInspector
from src.factory_controller import FactoryController
from src.piece_tracker import IndustrialPieceTracker
from src.plc_modbus import PLCModbusManager, VirtualPLCSimulator
from src.vision_worker import (
    StreamVisionWorker,
    execute_piece_inspection,
    is_live_camera_source,
    prepare_rgb_preview,
)


def safe_imread(file_path: str) -> Optional[np.ndarray]:
    """Reads image safely on Windows even if path contains Arabic or non-ASCII characters"""
    try:
        with open(file_path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


class IndustrialQCDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🏭 Smart Factory QC - نظام مراقبة الجودة والفرز الآلي بالذكاء الاصطناعي")
        self.root.geometry("1380x880")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#11111b")

        # Core Engines
        self.model_path = "best_defect_detector.onnx"
        self.inspector: Optional[HybridQualityInspector] = None
        
        # PLC & Modbus TCP Automation Engine
        self.plc_manager = PLCModbusManager(host="127.0.0.1", port=5020, on_log=self.append_log)
        self.plc_simulator = VirtualPLCSimulator(host="127.0.0.1", port=5020, on_log=self.append_log)
        self.controller = FactoryController(plc_manager=self.plc_manager)
        self.tracker = IndustrialPieceTracker(cooldown_sec=0.45)

        # State Variables
        self._closing = False
        self._inference_lock = threading.Lock()
        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._ui_drain_after_id: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.vision_worker: Optional[StreamVisionWorker] = None
        self._video_poll_after_id: Optional[str] = None
        self._batch_after_id: Optional[str] = None
        self._batch_token = 0
        self._single_busy = False
        self._canvas_size: Tuple[int, int] = (750, 500)
        self.is_video_running = False
        self.is_batch_running = False
        self.current_frame = None
        self.last_detections: List[Dict[str, Any]] = []

        # Product Profile & Filters
        self.product_profile_var = tk.StringVar(value="كبسولات وأدوية (Capsule)")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.check_color_var = tk.BooleanVar(value=False)  # Off by default for capsules
        self.check_shape_var = tk.BooleanVar(value=True)
        self.consecutive_limit_var = tk.StringVar(value="5 عيوب متتالية (افتراضي)")

        # Thread-safe plain-Python setting snapshots read by worker threads
        self._runtime_conf = 0.25
        self._runtime_profile = "capsule"
        self._runtime_color_check = False
        self._runtime_shape_check = True
        for var in (
            self.conf_var,
            self.product_profile_var,
            self.check_color_var,
            self.check_shape_var,
        ):
            var.trace_add("write", self._on_runtime_setting_changed)

        # PLC & Actuator UI Variables
        self.plc_host_var = tk.StringVar(value="127.0.0.1")
        self.plc_port_var = tk.StringVar(value="5020")
        self.plc_sim_active_var = tk.BooleanVar(value=False)
        self.plc_pulse_ms_var = tk.IntVar(value=650)
        self._video_arm_lamp_active = False

        # Video stream telemetry & alarms
        self.fps_tracker = 0.0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.last_beep_time = 0.0

        # Build UI
        self.setup_ui()

        # Auto-load model
        self.auto_load_model()

        # Start periodic UI dispatch & PLC polling on Main Thread
        self._drain_ui_queue()
        self.update_plc_status_ui()

    def setup_ui(self):
        # 1. Top Bar (Header & Status)
        header = tk.Frame(self.root, bg="#181825", height=65)
        header.pack(fill="x", side="top")

        lbl_app_title = tk.Label(
            header,
            text="🏭 نظام الفحص والفرز الذكي لمراقبة الجودة (Industrial AI Quality Control)",
            font=("Segoe UI", 16, "bold"),
            fg="#cdd6f4",
            bg="#181825"
        )
        lbl_app_title.pack(side="left", padx=20, pady=12)

        self.lbl_system_status = tk.Label(
            header,
            text="النظام: جاهز للعمل (نظام هجين نشط)",
            font=("Segoe UI", 11, "bold"),
            fg="#a6e3a1",
            bg="#181825"
        )
        self.lbl_system_status.pack(side="right", padx=20, pady=12)

        # 2. Main Work Area
        main_body = tk.Frame(self.root, bg="#11111b")
        main_body.pack(fill="both", expand=True, padx=12, pady=8)

        # Left Panel: Operators Control (Width: 320)
        left_panel = tk.Frame(main_body, bg="#1e1e2e", width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 8))
        left_panel.pack_propagate(False)

        # Center Viewport: Video / Image Screen
        center_panel = tk.Frame(main_body, bg="#181825")
        center_panel.pack(side="left", fill="both", expand=True, padx=4)

        # Right Panel: Factory Indicators & Counters (Width: 340)
        right_panel = tk.Frame(main_body, bg="#1e1e2e", width=340)
        right_panel.pack(side="right", fill="y", padx=(8, 0))
        right_panel.pack_propagate(False)

        # --- Populate Panels ---
        self._build_left_panel(left_panel)
        self._build_center_panel(center_panel)
        self._build_right_panel(right_panel)

    def _build_left_panel(self, parent: tk.Frame):
        lbl = tk.Label(parent, text="🎮 لوحة تحكم المشغل", font=("Segoe UI", 12, "bold"), fg="#89b4fa", bg="#1e1e2e")
        lbl.pack(pady=(8, 4))

        # Model Status Box
        self.btn_load_onnx = tk.Button(
            parent, text="🔄 تحميل ملف ONNX", font=("Segoe UI", 9, "bold"),
            bg="#313244", fg="#cdd6f4", activebackground="#45475a", relief="flat",
            command=self.browse_model_file
        )
        self.btn_load_onnx.pack(fill="x", padx=12, pady=2)

        self.lbl_model_info = tk.Label(parent, text="النموذج: غير متصل", font=("Segoe UI", 8), fg="#f38ba8", bg="#1e1e2e")
        self.lbl_model_info.pack(pady=(0, 4))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=3)

        # Product Profile Selector (Combobox)
        lbl_profile = tk.Label(parent, text="📦 نوع المنتج المراد فحصه (Profile):", font=("Segoe UI", 9, "bold"), fg="#f9e2af", bg="#1e1e2e")
        lbl_profile.pack(anchor="w", padx=12, pady=(2, 2))

        self.combo_profile = ttk.Combobox(
            parent, textvariable=self.product_profile_var, state="readonly",
            values=[
                "كبسولات وأدوية (Capsule)",
                "صواميل وقطع معدنية (Metal Nut)",
                "زجاجات وعبوات (Bottle)",
                "فحص عام مخصص (General Custom)"
            ],
            font=("Segoe UI", 9)
        )
        self.combo_profile.pack(fill="x", padx=12, pady=2)
        self.combo_profile.bind("<<ComboboxSelected>>", self.on_profile_changed)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Inspection Modes Buttons
        btn_image = tk.Button(
            parent, text="🖼️ فحص صورة خارجية منفردة", font=("Segoe UI", 10, "bold"),
            bg="#89b4fa", fg="#11111b", activebackground="#b4befe", relief="flat", height=2,
            command=self.inspect_single_image
        )
        btn_image.pack(fill="x", padx=12, pady=3)

        self.btn_video = tk.Button(
            parent, text="📹 تشغيل البث المباشر (الكاميرا)", font=("Segoe UI", 10, "bold"),
            bg="#a6e3a1", fg="#11111b", activebackground="#94e2d5", relief="flat", height=2,
            command=self.toggle_live_video
        )
        self.btn_video.pack(fill="x", padx=12, pady=3)

        btn_video_file = tk.Button(
            parent, text="🎬 فحص ملف فيديو مسجل", font=("Segoe UI", 9, "bold"),
            bg="#313244", fg="#cdd6f4", activebackground="#45475a", relief="flat", height=1,
            command=self.open_video_file
        )
        btn_video_file.pack(fill="x", padx=12, pady=2)

        self.btn_snap = tk.Button(
            parent, text="⚡ تسجيل القطعة الحالية (Snap & Log)", font=("Segoe UI", 9, "bold"),
            bg="#fab387", fg="#11111b", activebackground="#f9e2af", relief="flat",
            command=self.snap_and_register_current_frame
        )
        self.btn_snap.pack(fill="x", padx=12, pady=2)

        btn_batch = tk.Button(
            parent, text="📦 فحص مجلد صور متتالي (محاكاة سير)", font=("Segoe UI", 9),
            bg="#313244", fg="#cdd6f4", activebackground="#45475a", relief="flat",
            command=self.inspect_folder_stream
        )
        btn_batch.pack(fill="x", padx=12, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Hybrid Checkboxes
        lbl_filters = tk.Label(parent, text="🛡️ طبقات الفحص الذكي (Hybrid Filters):", font=("Segoe UI", 9, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        lbl_filters.pack(anchor="w", padx=12, pady=(0, 2))

        self.cb_color = tk.Checkbutton(
            parent, text="كشف البقع والصدأ (Color Shift)",
            variable=self.check_color_var, bg="#1e1e2e", fg="#fab387",
            selectcolor="#313244", activebackground="#1e1e2e", activeforeground="#fab387",
            font=("Segoe UI", 8)
        )
        self.cb_color.pack(anchor="w", padx=14)

        self.cb_shape = tk.Checkbutton(
            parent, text="كشف الانبعاج والأبعاد (Squeeze & Size)",
            variable=self.check_shape_var, bg="#1e1e2e", fg="#f5c2e7",
            selectcolor="#313244", activebackground="#1e1e2e", activeforeground="#f5c2e7",
            font=("Segoe UI", 8)
        )
        self.cb_shape.pack(anchor="w", padx=14)

        # Sensitivity Slider
        lbl_sens = tk.Label(parent, text="حساسية الكشف (Threshold):", font=("Segoe UI", 8, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        lbl_sens.pack(anchor="w", padx=12, pady=(3, 0))

        self.scale_sens = tk.Scale(
            parent, from_=0.10, to=0.85, resolution=0.05, orient="horizontal",
            variable=self.conf_var, bg="#1e1e2e", fg="#cdd6f4", highlightthickness=0,
            troughcolor="#313244", activebackground="#89b4fa"
        )
        self.scale_sens.pack(fill="x", padx=12, pady=1)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Conveyor Controls
        lbl_consec = tk.Label(parent, text="إيقاف السير التلقائي لتكرار العيوب:", font=("Segoe UI", 8), fg="#cdd6f4", bg="#1e1e2e")
        lbl_consec.pack(anchor="w", padx=12, pady=(2, 0))

        self.combo_consec = ttk.Combobox(
            parent, textvariable=self.consecutive_limit_var, state="readonly",
            values=[
                "5 عيوب متتالية (افتراضي)",
                "10 عيوب متتالية",
                "3 عيوب متتالية",
                "معطل (فحص مستمر بدون إيقاف)"
            ],
            font=("Segoe UI", 8)
        )
        self.combo_consec.pack(fill="x", padx=12, pady=(1, 3))
        self.combo_consec.bind("<<ComboboxSelected>>", self.on_consecutive_limit_changed)

        self.btn_conveyor = tk.Button(
            parent, text="⚙️ إيقاف / استئناف السير الناقل", font=("Segoe UI", 9, "bold"),
            bg="#f9e2af", fg="#11111b", relief="flat",
            command=self.toggle_conveyor_state
        )
        self.btn_conveyor.pack(fill="x", padx=12, pady=2)

        btn_estop = tk.Button(
            parent, text="🛑 إيقاف الطوارئ (EMERGENCY STOP)", font=("Segoe UI", 10, "bold"),
            bg="#f38ba8", fg="#11111b", activebackground="#eba0ac", relief="flat", height=2,
            command=self.emergency_stop
        )
        btn_estop.pack(fill="x", padx=12, pady=3)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=12, pady=3)

        # Reports & Archives Buttons
        btn_report = tk.Button(
            parent, text="📑 تصدير تقرير الوردية (Summary)", font=("Segoe UI", 9, "bold"),
            bg="#cba6f7", fg="#11111b", relief="flat",
            command=self.export_shift_report
        )
        btn_report.pack(fill="x", padx=12, pady=2)

        btn_archive = tk.Button(
            parent, text="📂 فتح مجلد صور العيوب", font=("Segoe UI", 8),
            bg="#313244", fg="#cdd6f4", relief="flat",
            command=self.open_archive_folder
        )
        btn_archive.pack(fill="x", padx=12, pady=2)

    def _build_center_panel(self, parent: tk.Frame):
        # Decision Banner
        self.decision_banner = tk.Label(
            parent,
            text="في انتظار بدء الفحص (Standby Mode)",
            font=("Segoe UI", 14, "bold"),
            bg="#313244",
            fg="#cdd6f4",
            height=2
        )
        self.decision_banner.pack(fill="x", side="top", padx=8, pady=8)

        # Video / Image Canvas
        self.canvas_label = tk.Label(parent, bg="#11111b")
        self.canvas_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas_label.bind("<Configure>", self._on_canvas_configure)

    def _build_right_panel(self, parent: tk.Frame):
        lbl = tk.Label(parent, text="📊 إحصائيات الخط المباشرة", font=("Segoe UI", 12, "bold"), fg="#89b4fa", bg="#1e1e2e")
        lbl.pack(pady=(8, 4))

        # Lamps Frame
        lamps_frame = tk.Frame(parent, bg="#252538", bd=1, relief="solid")
        lamps_frame.pack(fill="x", padx=10, pady=2)

        self.lamp_pass = tk.Label(lamps_frame, text="🟢 PASS (مطابق)", font=("Segoe UI", 10, "bold"), bg="#1e1e2e", fg="#a6e3a1", height=2)
        self.lamp_pass.pack(fill="x", padx=6, pady=2)

        self.lamp_reject = tk.Label(lamps_frame, text="🔴 REJECT (معيب)", font=("Segoe UI", 10, "bold"), bg="#1e1e2e", fg="#f38ba8", height=2)
        self.lamp_reject.pack(fill="x", padx=6, pady=2)

        self.lamp_ejector = tk.Label(lamps_frame, text="⚡ ذراع الفرز: جاهز", font=("Segoe UI", 10, "bold"), bg="#1e1e2e", fg="#cdd6f4", height=2)
        self.lamp_ejector.pack(fill="x", padx=6, pady=2)

        self.lamp_conveyor = tk.Label(lamps_frame, text="⚙️ السير: يعمل (Running)", font=("Segoe UI", 9, "bold"), bg="#1e1e2e", fg="#a6e3a1", height=1)
        self.lamp_conveyor.pack(fill="x", padx=6, pady=(2, 4))

        # PLC & Modbus Automation Panel
        plc_box = tk.LabelFrame(
            parent, text=" 🔌 ربط الـ PLC والذراع (Modbus TCP) ",
            font=("Segoe UI", 9, "bold"), fg="#89dceb", bg="#1e1e2e", bd=1
        )
        plc_box.pack(fill="x", padx=10, pady=(6, 2))

        # IP & Port inputs
        ip_row = tk.Frame(plc_box, bg="#1e1e2e")
        ip_row.pack(fill="x", padx=6, pady=2)

        tk.Label(ip_row, text="IP:", font=("Segoe UI", 8), fg="#cdd6f4", bg="#1e1e2e").pack(side="left")
        self.entry_plc_ip = tk.Entry(ip_row, textvariable=self.plc_host_var, width=12, font=("Segoe UI", 8), bg="#313244", fg="#cdd6f4", relief="flat")
        self.entry_plc_ip.pack(side="left", padx=2)

        tk.Label(ip_row, text="Port:", font=("Segoe UI", 8), fg="#cdd6f4", bg="#1e1e2e").pack(side="left", padx=(3, 0))
        self.entry_plc_port = tk.Entry(ip_row, textvariable=self.plc_port_var, width=5, font=("Segoe UI", 8), bg="#313244", fg="#cdd6f4", relief="flat")
        self.entry_plc_port.pack(side="left", padx=2)

        # Simulator checkbox
        self.cb_sim = tk.Checkbutton(
            plc_box, text="تشغيل محاكي PLC مدمج (Virtual)",
            variable=self.plc_sim_active_var, bg="#1e1e2e", fg="#a6e3a1",
            selectcolor="#313244", activebackground="#1e1e2e", activeforeground="#a6e3a1",
            font=("Segoe UI", 8), command=self.toggle_plc_simulator
        )
        self.cb_sim.pack(anchor="w", padx=6, pady=1)

        # Action Buttons Row (Connect + Manual Arm Test)
        btns_row = tk.Frame(plc_box, bg="#1e1e2e")
        btns_row.pack(fill="x", padx=6, pady=2)

        self.btn_plc_connect = tk.Button(
            btns_row, text="🔗 اتصال (Connect)", font=("Segoe UI", 8, "bold"),
            bg="#89dceb", fg="#11111b", activebackground="#94e2d5", relief="flat",
            command=self.toggle_plc_connection
        )
        self.btn_plc_connect.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_test_arm = tk.Button(
            btns_row, text="🦾 اختبار الذراع", font=("Segoe UI", 8, "bold"),
            bg="#fab387", fg="#11111b", activebackground="#f9e2af", relief="flat",
            command=self.test_manual_arm_pulse
        )
        btn_test_arm.pack(side="right", fill="x", expand=True, padx=(2, 0))

        # Status badge
        self.lbl_plc_badge = tk.Label(plc_box, text="🔴 غير متصل بـ PLC", font=("Segoe UI", 8, "bold"), fg="#f38ba8", bg="#1e1e2e")
        self.lbl_plc_badge.pack(anchor="w", padx=6, pady=1)

        # Live Signals Frame
        plc_sig_frame = tk.Frame(plc_box, bg="#252538")
        plc_sig_frame.pack(fill="x", padx=6, pady=(1, 4))

        self.lbl_plc_coil1 = tk.Label(plc_sig_frame, text="⚡ Coil 1 (الذراع): [0] في الانتظار", font=("Segoe UI", 8), fg="#cdd6f4", bg="#252538")
        self.lbl_plc_coil1.pack(anchor="w", padx=4, pady=1)

        self.lbl_plc_coil2 = tk.Label(plc_sig_frame, text="⚙️ Coil 2 (السير): [1] يعمل (RUN)", font=("Segoe UI", 8), fg="#a6e3a1", bg="#252538")
        self.lbl_plc_coil2.pack(anchor="w", padx=4, pady=1)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)

        # Counters Grid
        counters_frame = tk.Frame(parent, bg="#1e1e2e")
        counters_frame.pack(fill="x", padx=10)

        self.lbl_count_total = self._create_stat_card(counters_frame, "إجمالي المفحوص", "0", "#89b4fa")
        self.lbl_count_passed = self._create_stat_card(counters_frame, "القطع السليمة", "0", "#a6e3a1")
        self.lbl_count_rejected = self._create_stat_card(counters_frame, "القطع المعيبة", "0", "#f38ba8")
        self.lbl_count_yield = self._create_stat_card(counters_frame, "نسبة الجودة (Yield)", "100.0%", "#f9e2af")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)

        # Real-time Event Log Console
        lbl_log = tk.Label(parent, text="📝 سجل أحداث الخط اللحظي:", font=("Segoe UI", 9, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        lbl_log.pack(anchor="w", padx=10, pady=(0, 2))

        self.log_text = tk.Text(parent, height=10, bg="#11111b", fg="#a6adc8", font=("Consolas", 8), relief="flat")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.append_log("نظام الفحص الهجين جاهز للبدء.")

    def _create_stat_card(self, parent: tk.Frame, title: str, init_val: str, val_color: str) -> tk.Label:
        card = tk.Frame(parent, bg="#252538", bd=1, relief="solid")
        card.pack(fill="x", pady=2)
        lbl_title = tk.Label(card, text=title, font=("Segoe UI", 8), fg="#cdd6f4", bg="#252538")
        lbl_title.pack(side="left", padx=8, pady=3)
        lbl_val = tk.Label(card, text=init_val, font=("Segoe UI", 11, "bold"), fg=val_color, bg="#252538")
        lbl_val.pack(side="right", padx=8, pady=3)
        return lbl_val

    def on_profile_changed(self, event=None):
        selected = self.product_profile_var.get()
        if "Capsule" in selected:
            # For capsules, turn off color check to prevent normal orange shell from being flagged
            self.check_color_var.set(False)
            self.check_shape_var.set(True)
            self.append_log("تم تفعيل ملف الكبسولات: فحص الأبعاد والانبعاج نشط، وفحص الألوان معطل لتجنب الإنذار الكاذب.")
        elif "Metal Nut" in selected:
            self.check_color_var.set(True)
            self.check_shape_var.set(True)
            self.append_log("تم تفعيل ملف الصواميل: فحص بقع الصدأ واللون نشط.")
        else:
            self.append_log(f"تم اختيار ملف الفحص: {selected}")

    def get_current_profile_key(self) -> str:
        sel = self.product_profile_var.get()
        if "Capsule" in sel:
            return "capsule"
        elif "Metal Nut" in sel:
            return "metal_nut"
        elif "Bottle" in sel:
            return "bottle"
        return "general"

    def _on_canvas_configure(self, event) -> None:
        if event.width >= 100 and event.height >= 100:
            self._canvas_size = (int(event.width), int(event.height))

    def get_canvas_size(self) -> Tuple[int, int]:
        """Thread-safe canvas size accessor for background preview scaling."""
        return self._canvas_size

    def _on_runtime_setting_changed(self, *_args) -> None:
        """Keep plain Python setting snapshots synchronized with Tk variables."""
        try:
            self._runtime_conf = float(self.conf_var.get())
            self._runtime_profile = self.get_current_profile_key()
            self._runtime_color_check = bool(self.check_color_var.get())
            self._runtime_shape_check = bool(self.check_shape_var.get())
        except (tk.TclError, ValueError):
            pass

    def get_runtime_settings(self) -> Tuple[float, str, bool, bool]:
        """Thread-safe snapshot of (conf_thresh, profile, enable_color, enable_shape)."""
        return (
            self._runtime_conf,
            self._runtime_profile,
            self._runtime_color_check,
            self._runtime_shape_check,
        )

    def get_inspector(self) -> Optional[HybridQualityInspector]:
        """Accessor passed to background vision workers."""
        return self.inspector

    def post_to_ui(self, fn: Callable[[], None]) -> None:
        """Enqueue a callback for execution strictly on Tkinter's Main Thread."""
        if not self._closing:
            self._ui_queue.put(fn)

    def _drain_ui_queue(self) -> None:
        """Drain background UI callbacks on Tkinter's Main Thread."""
        if self._closing:
            return
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                print(f"UI dispatch error: {e}")
        if not self._closing:
            try:
                self._ui_drain_after_id = self.root.after(20, self._drain_ui_queue)
            except Exception:
                pass

    def append_log(self, msg: str):
        def _do_append():
            try:
                if self._closing:
                    return
                t_str = time.strftime("%H:%M:%S")
                self.log_text.insert("end", f"[{t_str}] {msg}\n")
                self.log_text.see("end")
            except Exception:
                pass

        if threading.current_thread() is threading.main_thread():
            _do_append()
        else:
            self.post_to_ui(_do_append)

    def auto_load_model(self):
        candidates = [
            Path("best_defect_detector.onnx"),
            Path("best.onnx"),
            Path("defect_detection_system/models/best.onnx")
        ]
        for c in candidates:
            if c.exists():
                self.load_model(str(c), async_ui=False)
                return

    def browse_model_file(self):
        path = filedialog.askopenfilename(
            title="اختر ملف أوزان ONNX",
            filetypes=[("ONNX Model", "*.onnx"), ("All Files", "*.*")]
        )
        if path:
            self.load_model(path, async_ui=True)

    def _apply_loaded_model_ui(self, path: str, err: Optional[Exception] = None):
        if self._closing:
            return
        self.btn_load_onnx.config(state="normal", text="🔄 تحميل ملف ONNX")
        if err is None:
            self.model_path = path
            m_name = Path(path).name
            self.lbl_model_info.config(text=f"النموذج: {m_name} (متصل)", fg="#a6e3a1")
            self.append_log(f"تم تحميل نموذج ONNX بنجاح: {m_name}")
        else:
            messagebox.showerror("خطأ في تحميل النموذج", f"تعذر تحميل ملف الـ ONNX:\n{err}")
            self.append_log(f"خطأ في تحميل النموذج: {err}")

    def load_model(self, path: str, async_ui: bool = False):
        conf = self._runtime_conf
        if not async_ui:
            try:
                new_inspector = HybridQualityInspector(path, conf_thresh=conf)
                with self._inference_lock:
                    self.inspector = new_inspector
                self._apply_loaded_model_ui(path)
            except Exception as e:
                self._apply_loaded_model_ui(path, err=e)
            return

        self.btn_load_onnx.config(state="disabled", text="⏳ جاري تحميل ONNX...")

        def _worker():
            try:
                new_inspector = HybridQualityInspector(path, conf_thresh=conf)
                with self._inference_lock:
                    self.inspector = new_inspector
                self.post_to_ui(lambda: self._apply_loaded_model_ui(path))
            except Exception as e:
                self.post_to_ui(lambda err=e: self._apply_loaded_model_ui(path, err=err))

        threading.Thread(target=_worker, daemon=True).start()

    def inspect_single_image(self):
        if self.inspector is None:
            messagebox.showwarning("تنبيه", "يرجى أولاً تحميل ملف النموذج ONNX!")
            return
        if self._single_busy:
            self.append_log("⏳ يوجد فحص صورة قيد التنفيذ حالياً، يرجى الانتظار لحظة...")
            return

        file_path = filedialog.askopenfilename(
            title="اختر صورة منتج لفحصها",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        if self.is_video_running:
            self.stop_live_video()
        self._cancel_folder_batch()

        self._single_busy = True
        self.decision_banner.config(
            text="⏳ جاري قراءة وفحص الصورة بالذكاء الاصطناعي...",
            bg="#313244", fg="#f9e2af"
        )
        self._process_piece_async(
            frame=None,
            file_path=file_path,
            on_done=lambda _payload: setattr(self, "_single_busy", False),
        )

    def open_video_file(self):
        file_path = filedialog.askopenfilename(
            title="اختر ملف فيديو لفحصه",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
        )
        if file_path:
            self.start_live_video(file_path)

    def _is_ejector_pulse_active(self) -> bool:
        """Return True while the pneumatic ejector pulse is physically or logically active."""
        self.controller.update_actuator_state()
        if hasattr(self, 'plc_manager') and self.plc_manager.is_connected():
            st = self.plc_manager.read_plc_status()
            return bool(st.get('eject_coil', False))
        return bool(self.controller.ejector_active)

    def _apply_registered_piece_ui(
        self,
        result: Dict[str, Any],
        banner_mode: str = "single",
        log_msg: Optional[str] = None,
        show_stop_popup: bool = True,
    ) -> None:
        """Apply counters, banner/lamp status, logs, and auto-stop alert on Tkinter Main Thread."""
        if self._closing:
            return

        saved_name = Path(result['saved_img_path']).name if result.get('saved_img_path') else ""
        img_note = f" ({saved_name})" if saved_name else ""

        if banner_mode == "single":
            self._video_arm_lamp_active = False
            if result['status'] == 'REJECT':
                self.decision_banner.config(
                    text=f"❌ REJECT (مرفوض) - {result['piece_id']} | عيوب: {result['num_defects']} (ثقة: {result['max_conf']:.0f}%)",
                    bg="#f38ba8", fg="#11111b"
                )
                self.lamp_reject.config(bg="#f38ba8", fg="#11111b")
                self.lamp_pass.config(bg="#1e1e2e", fg="#a6e3a1")
                self.lamp_ejector.config(text="⚡ ذراع الفرز: دفع العينة! (ACTIVE)", bg="#fab387", fg="#11111b")
                default_log = f"{result['piece_id']}: عيب مكتشف -> إطلاق الذراع الهوائي وحفظ الصورة بالسجل{img_note}."
            else:
                self.decision_banner.config(
                    text=f"✅ PASS (سليم) - {result['piece_id']} مطابق لمعايير الجودة 100%",
                    bg="#a6e3a1", fg="#11111b"
                )
                self.lamp_pass.config(bg="#a6e3a1", fg="#11111b")
                self.lamp_reject.config(bg="#1e1e2e", fg="#f38ba8")
                self.lamp_ejector.config(text="⚡ ذراع الفرز: جاهز (STANDBY)", bg="#1e1e2e", fg="#cdd6f4")
                default_log = f"{result['piece_id']}: قطعة سليمة (PASS) - تم تسجيلها."
        else:
            if result['status'] == 'REJECT':
                self.lamp_ejector.config(text="⚡ ذراع الفرز: دفع العينة! (ACTIVE)", bg="#fab387", fg="#11111b")
                default_log = f"{result['piece_id']}: عيب مكتشف بالفيديو -> إطلاق الذراع وحفظ الصورة بالسجل{img_note}."
            else:
                default_log = f"{result['piece_id']}: قطعة سليمة بالفيديو (PASS) -> تم تسجيلها."

        self.append_log(log_msg if log_msg is not None else default_log)

        # Update Live Counters using atomic result snapshot
        self.lbl_count_total.config(text=str(result['total_inspected']))
        self.lbl_count_passed.config(text=str(result['total_passed']))
        self.lbl_count_rejected.config(text=str(result['total_rejected']))
        self.lbl_count_yield.config(text=f"{result['yield_rate']:.1f}%")

        if not result['conveyor_running']:
            self.lamp_conveyor.config(text="🛑 السير: متوقف (تكرار العيوب!)", bg="#f38ba8", fg="#11111b")
            self.append_log("⚠️ تحذير: تم إيقاف السير تلقائياً لتكرار العيوب!")
            if show_stop_popup:
                messagebox.showwarning("تنبيه خط الإنتاج", "تم إيقاف السير الناقل تلقائياً لتكرار العيوب المتتالية!")

        self._refresh_plc_labels_from_cache()

    def _apply_registered_piece_payload_ui(
        self,
        payload: Dict[str, Any],
        log_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        """Render preview first, then update banner, counters, logs, and any stop modal."""
        if self._closing:
            return
        result = payload['result']
        self._render_rgb_to_canvas(payload['rgb_preview'])
        custom_log = log_builder(result) if log_builder is not None else None
        self._apply_registered_piece_ui(
            result,
            banner_mode="single",
            log_msg=custom_log,
            show_stop_popup=True,
        )

    def _update_live_stream_lamps(self, detections: List[Dict[str, Any]]) -> None:
        """Update continuous Real-Time Lamp & Banner visual feedback during video streaming."""
        num_defects = len(detections)
        if num_defects > 0:
            max_conf = max([d['conf'] for d in detections]) * 100
            types_str = ", ".join(set([d['label'].split()[0] for d in detections]))
            self.decision_banner.config(
                text=f"❌ REJECT (مرفوض) - تم اكتشاف {num_defects} عيب [{types_str}] (ثقة: {max_conf:.0f}%)",
                bg="#f38ba8", fg="#11111b"
            )
            self.lamp_reject.config(bg="#f38ba8", fg="#11111b")
            self.lamp_pass.config(bg="#1e1e2e", fg="#a6e3a1")
        else:
            self.decision_banner.config(
                text="✅ PASS (منتج سليم مطابق لمعايير الجودة 100%)",
                bg="#a6e3a1", fg="#11111b"
            )
            self.lamp_pass.config(bg="#a6e3a1", fg="#11111b")
            self.lamp_reject.config(bg="#1e1e2e", fg="#f38ba8")

        if self._is_ejector_pulse_active():
            self._video_arm_lamp_active = True
            self.lamp_ejector.config(text="⚡ ذراع الفرز: دفع العينة! (ACTIVE)", bg="#fab387", fg="#11111b")
        else:
            self._video_arm_lamp_active = False
            self.lamp_ejector.config(text="⚡ ذراع الفرز: جاهز (STANDBY)", bg="#1e1e2e", fg="#cdd6f4")

    def _process_piece_async(
        self,
        frame: Optional[np.ndarray],
        file_path: Optional[str] = None,
        on_done: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
        log_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        require_running: bool = False,
        expected_epoch: Optional[int] = None,
    ) -> None:
        """Load image (if path) and run detection/archiving off Tkinter's Main Thread."""
        settings = self.get_runtime_settings()

        def _worker():
            payload = None
            err_msg = None
            loaded_img = frame
            try:
                if loaded_img is None and file_path is not None:
                    loaded_img = safe_imread(file_path)
                    if loaded_img is None:
                        err_msg = f"تعذر فتح ملف الصورة:\n{file_path}"
                        return
                if loaded_img is not None:
                    payload = execute_piece_inspection(
                        self.get_inspector,
                        self._inference_lock,
                        self.controller,
                        loaded_img,
                        settings,
                        self.get_canvas_size,
                        require_running=require_running,
                        expected_epoch=expected_epoch,
                    )
            except Exception as e:
                err_msg = f"حدث خطأ أثناء الفحص:\n{e}"
            finally:
                def _finish():
                    if self._closing:
                        return
                    if on_done is not None:
                        on_done(payload)
                    if err_msg is not None:
                        self.decision_banner.config(
                            text="في انتظار بدء الفحص (Standby Mode)",
                            bg="#313244", fg="#cdd6f4"
                        )
                        messagebox.showerror("خطأ", err_msg)
                        return
                    if payload is not None:
                        self.current_frame = payload['raw_frame']
                        self.last_detections = payload['detections']
                        self._apply_registered_piece_payload_ui(payload, log_builder=log_builder)

                self.post_to_ui(_finish)

        threading.Thread(target=_worker, daemon=True).start()

    def process_and_register_piece(self, frame: np.ndarray):
        """Synchronously inspect and register a piece (kept for direct programmatic callers)."""
        payload = execute_piece_inspection(
            self.get_inspector,
            self._inference_lock,
            self.controller,
            frame,
            self.get_runtime_settings(),
            self.get_canvas_size,
        )
        if payload is not None:
            self.last_detections = payload['detections']
            self._apply_registered_piece_payload_ui(payload)

    def _render_rgb_to_canvas(self, rgb_img: np.ndarray) -> None:
        """Upload pre-resized RGB image to Tkinter Canvas quickly on Main Thread."""
        if self._closing:
            return
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_img))
        self.canvas_label.img_tk = img_tk
        self.canvas_label.config(image=img_tk)

    def render_frame_to_canvas(self, bgr_img: np.ndarray):
        if self._closing:
            return
        canvas_w = self.canvas_label.winfo_width()
        canvas_h = self.canvas_label.winfo_height()
        if canvas_w < 100 or canvas_h < 100:
            self.root.update_idletasks()
            canvas_w = self.canvas_label.winfo_width()
            canvas_h = self.canvas_label.winfo_height()
        if canvas_w >= 100 and canvas_h >= 100:
            self._canvas_size = (canvas_w, canvas_h)

        rgb_img = prepare_rgb_preview(bgr_img, (canvas_w, canvas_h))
        self._render_rgb_to_canvas(rgb_img)

    def toggle_live_video(self):
        if self.is_video_running:
            self.stop_live_video()
        else:
            self.start_live_video(1)

    def start_live_video(self, source):
        if self.inspector is None:
            messagebox.showwarning("تنبيه", "يرجى أولاً تحميل ملف النموذج ONNX!")
            return

        if self.is_video_running:
            self.stop_live_video()
        self._cancel_folder_batch()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("خطأ", f"تعذر فتح مصدر الفيديو/الكاميرا:\n{source}")
            return

        self.cap = cap
        self.tracker.reset()
        self.is_video_running = True
        self.btn_video.config(text="⏹️ إيقاف البث", bg="#f38ba8")

        src_name = f"الكاميرا المباشرة ({source})" if is_live_camera_source(source) else Path(str(source)).name
        self.append_log(f"بدأ تشغيل: {src_name} - عداد القطع التلقائي نشط.")

        self.vision_worker = StreamVisionWorker(
            source=source,
            cap=cap,
            get_inspector_fn=self.get_inspector,
            inference_lock=self._inference_lock,
            tracker=self.tracker,
            controller=self.controller,
            get_settings_fn=self.get_runtime_settings,
            get_canvas_size_fn=self.get_canvas_size,
        )
        self.vision_worker.start()
        self._poll_video_ui()

    def _poll_video_ui(self):
        """Lightweight UI poller consuming processed frames & events from StreamVisionWorker."""
        if self._closing or not self.is_video_running or self.vision_worker is None:
            return

        try:
            packet, events, eof = self.vision_worker.pop_latest_state()

            show_stop_popup = False
            for res in events:
                if res.get('status') == 'REJECT':
                    self._video_arm_lamp_active = True
                self._apply_registered_piece_ui(
                    res,
                    banner_mode="counter_only",
                    show_stop_popup=False,
                )
                if not res['conveyor_running']:
                    show_stop_popup = True

            if packet is not None:
                self.current_frame = packet.raw_frame
                self.last_detections = packet.detections
                if not packet.paused:
                    self._update_live_stream_lamps(packet.detections)
                self._render_rgb_to_canvas(packet.rgb_preview)

            self._refresh_plc_labels_from_cache()

            if show_stop_popup:
                messagebox.showwarning("تنبيه خط الإنتاج", "تم إيقاف السير الناقل تلقائياً لتكرار العيوب المتتالية!")

            if eof:
                self.stop_live_video(log_msg="انتهى ملف الفيديو أو انقطع اتصال الكاميرا.")
                return
        except Exception as e:
            print(f"Video UI poll error: {e}")

        if self.is_video_running and not self._closing:
            self._video_poll_after_id = self.root.after(20, self._poll_video_ui)

    def stop_live_video(self, log_msg: str = "تم إيقاف البث."):
        if self._video_poll_after_id is not None:
            try:
                self.root.after_cancel(self._video_poll_after_id)
            except Exception:
                pass
            self._video_poll_after_id = None

        was_running = self.is_video_running or self.vision_worker is not None
        self.is_video_running = False

        if self.vision_worker is not None:
            self.vision_worker.stop(timeout=2.0)
            self.vision_worker = None
        self.cap = None

        if not self._closing:
            self.btn_video.config(text="📹 تشغيل البث المباشر (الكاميرا)", bg="#a6e3a1")
            self.decision_banner.config(text="تم إيقاف البث (Standby Mode)", bg="#313244", fg="#cdd6f4")
            self.lamp_pass.config(bg="#1e1e2e", fg="#a6e3a1")
            self.lamp_reject.config(bg="#1e1e2e", fg="#f38ba8")
            if not self._is_ejector_pulse_active():
                self._video_arm_lamp_active = False
                self.lamp_ejector.config(text="⚡ ذراع الفرز: جاهز (STANDBY)", bg="#1e1e2e", fg="#cdd6f4")
            self._refresh_plc_labels_from_cache()
            if was_running:
                self.append_log(log_msg)

    def snap_and_register_current_frame(self):
        """Immediately captures and registers the current frame asynchronously into counters, CSV, and defect archive!"""
        if self.inspector is None:
            messagebox.showwarning("تنبيه", "يرجى أولاً تحميل ملف النموذج ONNX!")
            return
        if self.current_frame is None:
            messagebox.showinfo("تنبيه", "لا يوجد إطار صورة أو فيديو نشط حالياً لالتقاطه.")
            return
        if self._single_busy:
            return

        self._single_busy = True
        frame_copy = self.current_frame.copy()

        def _done(_payload):
            self._single_busy = False

        self._process_piece_async(
            frame_copy,
            on_done=_done,
            log_builder=lambda res: (
                f"⚡ التقاط يدوي: تم تسجيل وتوثيق القطعة #{res['total_inspected']} ({res['status']})"
                + (f" ({Path(res['saved_img_path']).name})" if res.get('saved_img_path') else "")
                + " بنجاح."
            ),
        )

    def _cancel_folder_batch(self) -> None:
        self._batch_token += 1
        self.is_batch_running = False
        if self._batch_after_id is not None:
            try:
                self.root.after_cancel(self._batch_after_id)
            except Exception:
                pass
            self._batch_after_id = None

    def inspect_folder_stream(self):
        if self.inspector is None:
            messagebox.showwarning("تنبيه", "يرجى أولاً تحميل ملف النموذج ONNX!")
            return

        folder = filedialog.askdirectory(title="اختر مجلد صور لاختبار تسلسل خط الإنتاج")
        if not folder:
            return

        p = Path(folder)
        img_files = list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
        if not img_files:
            messagebox.showinfo("تنبيه", "لا توجد صور في المجلد المختار.")
            return

        if self.is_video_running:
            self.stop_live_video()
        self._cancel_folder_batch()

        self._batch_token += 1
        active_token = self._batch_token
        self.is_batch_running = True
        self.append_log(f"بدء فحص تسلسلي لمجلد يحتوي على {len(img_files)} صورة - تسجيل كل قطعة تلقائياً...")

        processed_count = 0

        def run_batch(idx=0):
            nonlocal processed_count
            if self._closing or active_token != self._batch_token:
                return
            if idx >= len(img_files) or not self.controller.is_conveyor_running():
                self.is_batch_running = False
                self._batch_after_id = None
                if idx >= len(img_files):
                    self.append_log(f"اكتمل فحص المجلد بنجاح وتم تسجيل {processed_count} قطعة!")
                    messagebox.showinfo("اكتمل الفحص", f"تم فحص وتسجيل {processed_count} قطعة في التقرير بنجاح!")
                else:
                    self.append_log(f"توقف فحص المجلد عند القطعة {processed_count} من {len(img_files)} بسبب توقف السير.")
                return

            file_path = str(img_files[idx])
            epoch = self.controller.get_state_epoch()
            settings = self.get_runtime_settings()

            def _worker():
                if self._closing or active_token != self._batch_token:
                    return
                img = safe_imread(file_path)
                payload = None
                if img is not None and not self._closing and active_token == self._batch_token:
                    payload = execute_piece_inspection(
                        self.get_inspector,
                        self._inference_lock,
                        self.controller,
                        img,
                        settings,
                        self.get_canvas_size,
                        require_running=True,
                        expected_epoch=epoch,
                    )

                def _after_item():
                    nonlocal processed_count
                    if self._closing or active_token != self._batch_token:
                        return
                    if payload is not None:
                        self.current_frame = payload['raw_frame']
                        self.last_detections = payload['detections']
                        processed_count += 1
                        self._apply_registered_piece_payload_ui(payload)
                    self._batch_after_id = self.root.after(750, lambda: run_batch(idx + 1))

                if not self._closing and active_token == self._batch_token:
                    self.post_to_ui(_after_item)

            threading.Thread(target=_worker, daemon=True).start()

        run_batch(0)

    def on_consecutive_limit_changed(self, event=None):
        val = self.consecutive_limit_var.get()
        if "3" in val:
            self.controller.set_consecutive_limit(3)
        elif "5" in val:
            self.controller.set_consecutive_limit(5)
        elif "10" in val:
            self.controller.set_consecutive_limit(10)
        elif "معطل" in val:
            self.controller.set_consecutive_limit(0)  # 0 means disabled
        self.append_log(f"تم تعديل حد إيقاف تكرار العيوب إلى: {val}")

    def toggle_conveyor_state(self):
        state = self.controller.toggle_conveyor()
        if state:
            self.lamp_conveyor.config(text="⚙️ السير: يعمل (Running)", bg="#1e1e2e", fg="#a6e3a1")
            self.append_log("تم استئناف تشغيل السير الناقل وتصفير عداد تكرار العيوب.")
        else:
            self.lamp_conveyor.config(text="⚙️ السير: متوقف (Paused)", bg="#1e1e2e", fg="#f9e2af")
            self.append_log("تم إيقاف السير الناقل يدوياً.")

    def emergency_stop(self):
        self.controller.trigger_emergency_stop()
        self.lamp_conveyor.config(text="🛑 السير: طوارئ متوقف (E-STOP)", bg="#f38ba8", fg="#11111b")
        self.append_log("🚨 إيقاف طوارئ فوري (E-STOP ACTIVATED)!")
        messagebox.showerror("إيقاف الطوارئ", "تم تفعيل إيقاف الطوارئ وإيقاف خط السير فوراً!")

    def export_shift_report(self):
        report_path = self.controller.export_shift_summary()
        self.append_log(f"تم تصدير تقرير الوردية إلى: {report_path}")
        messagebox.showinfo("تم تصدير التقرير", f"تم إنشاء التقرير الشامل بنجاح:\n{report_path}")

    def open_archive_folder(self):
        folder = self.controller.archive_dir.resolve()
        if sys.platform == "win32":
            os.startfile(str(folder))
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def toggle_plc_simulator(self):
        if self.plc_sim_active_var.get():
            self.append_log("جاري تشغيل محاكي الـ PLC الافتراضي المدمج...")
            success = self.plc_simulator.start()
            if success:
                self.plc_host_var.set("127.0.0.1")
                self.plc_port_var.set("5020")
                self.root.after(400, self.connect_plc)
        else:
            self.append_log("تم إلغاء تفعيل محاكي الـ PLC.")
            self.plc_manager.disconnect()
            self.btn_plc_connect.config(text="🔗 اتصال (Connect)", bg="#89dceb")
            self.lbl_plc_badge.config(text="🔴 غير متصل بـ PLC", fg="#f38ba8")

    def toggle_plc_connection(self):
        if not self.plc_manager.is_connected():
            self.connect_plc()
        else:
            self.plc_manager.disconnect()
            self.btn_plc_connect.config(text="🔗 اتصال (Connect)", bg="#89dceb")
            self.lbl_plc_badge.config(text="🔴 غير متصل بـ PLC", fg="#f38ba8")
            self.append_log("تم قطع الاتصال بالـ PLC.")

    def connect_plc(self):
        host = self.plc_host_var.get().strip()
        try:
            port = int(self.plc_port_var.get().strip())
        except ValueError:
            port = 5020
            self.plc_port_var.set("5020")

        self.append_log(f"جاري الاتصال بـ PLC ({host}:{port})...")
        self.btn_plc_connect.config(text="⏳ جاري الاتصال...", state="disabled")

        def _do_conn():
            ok = self.plc_manager.connect(host=host, port=port)
            def _update_ui():
                try:
                    self.btn_plc_connect.config(state="normal")
                    if ok:
                        self.btn_plc_connect.config(text="❌ قطع (Disconnect)", bg="#f38ba8")
                        self.lbl_plc_badge.config(text=f"🟢 متصل: {host}:{port}", fg="#a6e3a1")
                    else:
                        self.btn_plc_connect.config(text="🔗 اتصال (Connect)", bg="#89dceb")
                        self.lbl_plc_badge.config(text="🔴 فشل الاتصال بالـ PLC", fg="#f38ba8")
                except Exception:
                    pass
            self.post_to_ui(_update_ui)

        threading.Thread(target=_do_conn, daemon=True).start()

    def test_manual_arm_pulse(self):
        if not self.plc_manager.is_connected():
            messagebox.showwarning("تنبيه الاتصال", "يرجى أولاً الاتصال بمتحكم PLC لاختبار الذراع!")
            return
        self.append_log("🦾 اختبار يدوي: إرسال نبضة تحريك الذراع الهوائي (Pulse Test)...")
        self.plc_manager.trigger_ejector_pulse(self.plc_pulse_ms_var.get())
        self.lamp_ejector.config(text="⚡ ذراع الفرز: دفع يدوي!", bg="#fab387", fg="#11111b")
        self._refresh_plc_labels_from_cache()
        if hasattr(self.controller, 'trigger_alarm'):
            self.controller.trigger_alarm()

    def _refresh_plc_labels_from_cache(self):
        """Synchronize Coil 1 & Coil 2 UI badges immediately from PLCManager RAM cache."""
        if self._closing or not hasattr(self, 'plc_manager'):
            return
        try:
            self.controller.update_actuator_state()
            eject_c = False
            plc_connected = self.plc_manager.is_connected()
            if plc_connected:
                st = self.plc_manager.read_plc_status()
                if st.get('connected'):
                    eject_c = bool(st.get('eject_coil', False))
                    conv_c = bool(st.get('conveyor_coil', False))
                    if eject_c:
                        self.lbl_plc_coil1.config(
                            text="⚡ Coil 1 (الذراع): [1] إطلاق نبضة!",
                            bg="#fab387",
                            fg="#11111b",
                        )
                    else:
                        self.lbl_plc_coil1.config(
                            text="⚡ Coil 1 (الذراع): [0] في الانتظار",
                            bg="#252538",
                            fg="#cdd6f4",
                        )

                    if conv_c:
                        self.lbl_plc_coil2.config(
                            text="⚙️ Coil 2 (السير): [1] يعمل (RUN)",
                            bg="#252538",
                            fg="#a6e3a1",
                        )
                    else:
                        self.lbl_plc_coil2.config(
                            text="🛑 Coil 2 (السير): [0] متوقف (STOP)",
                            bg="#252538",
                            fg="#f38ba8",
                        )
            else:
                self.lbl_plc_coil1.config(
                    text="⚡ Coil 1 (الذراع): [0] في الانتظار",
                    bg="#252538",
                    fg="#cdd6f4",
                )

            if self._video_arm_lamp_active:
                pulse_live = eject_c if plc_connected else bool(self.controller.ejector_active)
                if not pulse_live:
                    self._video_arm_lamp_active = False
                    self.lamp_ejector.config(
                        text="⚡ ذراع الفرز: جاهز (STANDBY)",
                        bg="#1e1e2e",
                        fg="#cdd6f4",
                    )
        except Exception:
            pass

    def update_plc_status_ui(self):
        """Periodic fast status checker (50ms RAM lookup) to update PLC coils in UI"""
        if self._closing:
            return
        self._refresh_plc_labels_from_cache()
        try:
            if not self._closing:
                self.root.after(50, self.update_plc_status_ui)
        except Exception:
            pass

    def on_closing(self):
        self._closing = True
        if self._ui_drain_after_id is not None:
            try:
                self.root.after_cancel(self._ui_drain_after_id)
            except Exception:
                pass
            self._ui_drain_after_id = None
        self._cancel_folder_batch()
        if self.is_video_running or self.vision_worker is not None:
            self.stop_live_video()
        if hasattr(self, 'plc_manager'):
            self.plc_manager.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialQCDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
