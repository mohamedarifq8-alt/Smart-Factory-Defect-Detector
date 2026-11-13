"""
Industrial Quality Control - Factory Automation & Controller Logic
Handles:
- Production Counting (Total, Pass, Reject, Yield Rate %)
- Virtual Trigger / Inspection Line for Conveyor Belts
- Actuator / Sorting Arm Signal (Pneumatic Ejector Pulse)
- Audible Alarms (winsound on Windows)
- Conveyor Emergency Stop (E-Stop) & Consecutive Defect Lock
- Automatic Defect Image Archiving & CSV/Excel Shift Reports (Supports Unicode/Arabic paths)
"""

import os
import cv2
import csv
import time
import datetime
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class FactoryController:
    def __init__(self, archive_dir: str = "defect_records", 
                 consecutive_limit: int = 5,
                 log_file: str = "quality_inspection_log.csv",
                 plc_manager: Optional[Any] = None):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = Path(log_file)
        self.consecutive_limit = consecutive_limit
        self.plc_manager = plc_manager

        # Split fast state lock from disk I/O lock so E-Stop never waits on JPEG/CSV writes
        self._state_lock = threading.RLock()
        self._file_lock = threading.Lock()
        self._state_epoch = 0
        
        # Production Counters
        self.total_inspected = 0
        self.total_passed = 0
        self.total_rejected = 0
        self.consecutive_defects = 0
        
        # System States
        self.conveyor_running = True
        self.ejector_active = False
        self.ejector_trigger_time = 0.0
        self.ejector_duration = 0.65  # seconds pulse (synced with PLC Coil 1 duration)
        self.alarm_active = False
        self.emergency_stopped = False
        
        # Initialize CSV log with header if not exists
        if not self.log_file.exists():
            self._init_csv()

    def _init_csv(self):
        with self._file_lock:
            with open(self.log_file, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Piece_ID", 
                    "Timestamp", 
                    "Status", 
                    "Defects_Count", 
                    "Max_Confidence_%", 
                    "Ejector_Action", 
                    "Conveyor_State", 
                    "Archived_Image_Path"
                ])

    def get_state_epoch(self) -> int:
        with self._state_lock:
            return self._state_epoch

    def is_conveyor_running(self) -> bool:
        with self._state_lock:
            return self.conveyor_running

    def set_consecutive_limit(self, limit: int) -> None:
        with self._state_lock:
            self.consecutive_limit = limit

    @property
    def yield_rate(self) -> float:
        """Calculate Yield Rate percentage"""
        with self._state_lock:
            if self.total_inspected == 0:
                return 100.0
            return (self.total_passed / self.total_inspected) * 100.0

    def register_inspection(
        self,
        frame_annotated: np.ndarray,
        detections: List[Dict[str, Any]],
        require_running: bool = False,
        expected_epoch: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Register a completed product inspection safely across worker threads.
        Determines PASS / REJECT, triggers ejector arm, logs defect image and CSV entry.
        Returns None if skipped due to conveyor stop or state epoch mismatch.
        """
        now_dt = datetime.datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_slug = now_dt.strftime("%Y%m%d_%H%M%S")

        num_defects = len(detections)
        is_defect = num_defects > 0
        saved_img_path = ""
        save_path: Optional[Path] = None

        with self._state_lock:
            if require_running and not self.conveyor_running:
                return None
            if expected_epoch is not None and self._state_epoch != expected_epoch:
                return None

            self.total_inspected += 1
            piece_id = f"PC-{self.total_inspected:05d}"

            if is_defect:
                self.total_rejected += 1
                self.consecutive_defects += 1
                status = "REJECT"
                max_conf = max([d['conf'] for d in detections]) * 100

                # 1. Trigger Sorting Arm (Pneumatic Ejector)
                self.ejector_active = True
                self.ejector_trigger_time = time.time()
                ejector_action = "PNEUMATIC_ARM_TRIGGERED"

                # Send Modbus Actuation Pulse & Telemetry to PLC
                if self.plc_manager and self.plc_manager.is_connected():
                    self.plc_manager.trigger_ejector_pulse(int(self.ejector_duration * 1000))
                    self.plc_manager.update_telemetry("REJECT", self.total_inspected, self.total_rejected)

                filename = f"defect_{piece_id}_{timestamp_slug}.jpg"
                save_path = self.archive_dir / filename
                saved_img_path = str(save_path)

                # 2. Check Consecutive Defect Emergency Limit (if enabled > 0)
                if self.consecutive_limit > 0 and self.consecutive_defects >= self.consecutive_limit:
                    self.conveyor_running = False
                    self.emergency_stopped = True
                    self._state_epoch += 1
                    if self.plc_manager and self.plc_manager.is_connected():
                        self.plc_manager.set_conveyor_state(False)
            else:
                self.total_passed += 1
                self.consecutive_defects = 0
                status = "PASS"
                max_conf = 0.0
                ejector_action = "ALLOWED_ON_MAIN_LINE"

                if self.plc_manager and self.plc_manager.is_connected():
                    self.plc_manager.update_telemetry("PASS", self.total_inspected, self.total_rejected)

            conveyor_state_now = self.conveyor_running
            total_inspected_now = self.total_inspected
            total_passed_now = self.total_passed
            total_rejected_now = self.total_rejected
            yield_now = (total_passed_now / total_inspected_now) * 100.0 if total_inspected_now > 0 else 100.0

        # Non-blocking alarm sound outside state lock
        if is_defect:
            self.trigger_alarm()

        # Encode defect JPEG off the state lock so E-Stop remains immediate
        enc_img = None
        if is_defect and save_path is not None:
            stamped_img = frame_annotated.copy()
            cv2.putText(stamped_img, f"{piece_id} | REJECT | Conf: {max_conf:.1f}% | {timestamp_str}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            success, buf = cv2.imencode('.jpg', stamped_img)
            if success:
                enc_img = buf

        # Serialize file writes cleanly
        with self._file_lock:
            if enc_img is not None and save_path is not None:
                enc_img.tofile(str(save_path))
            with open(self.log_file, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    piece_id,
                    timestamp_str,
                    status,
                    num_defects,
                    f"{max_conf:.1f}",
                    ejector_action,
                    "RUNNING" if conveyor_state_now else "STOPPED_ALARM",
                    saved_img_path
                ])

        return {
            'piece_id': piece_id,
            'timestamp': timestamp_str,
            'status': status,
            'num_defects': num_defects,
            'max_conf': max_conf,
            'ejector_action': ejector_action,
            'conveyor_running': conveyor_state_now,
            'saved_img_path': saved_img_path,
            'yield_rate': yield_now,
            'total_inspected': total_inspected_now,
            'total_passed': total_passed_now,
            'total_rejected': total_rejected_now,
        }

    def update_actuator_state(self):
        """Reset ejector arm pulse after pulse duration expires"""
        with self._state_lock:
            if self.ejector_active:
                if time.time() - self.ejector_trigger_time > self.ejector_duration:
                    self.ejector_active = False

    def trigger_alarm(self):
        """Play industrial alert beep asynchronously (non-blocking)"""
        if HAS_WINSOUND:
            def _beep():
                try:
                    winsound.Beep(1200, 180)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()
        else:
            print("\a", end="")

    def toggle_conveyor(self):
        """Manual Start/Stop Conveyor Belt"""
        with self._state_lock:
            self.conveyor_running = not self.conveyor_running
            self._state_epoch += 1
            if self.conveyor_running:
                self.emergency_stopped = False
                self.consecutive_defects = 0
            running = self.conveyor_running
        if self.plc_manager and self.plc_manager.is_connected():
            self.plc_manager.set_conveyor_state(running)
        return running

    def trigger_emergency_stop(self):
        """Operator Emergency Stop (E-Stop)"""
        with self._state_lock:
            self.conveyor_running = False
            self.emergency_stopped = True
            self._state_epoch += 1
        if self.plc_manager and self.plc_manager.is_connected():
            self.plc_manager.set_conveyor_state(False)
        if HAS_WINSOUND:
            def _beep():
                try:
                    winsound.Beep(800, 400)
                except Exception:
                    pass
            threading.Thread(target=_beep, daemon=True).start()

    def export_shift_summary(self, export_path: Optional[str] = None) -> str:
        """Generate a complete shift summary report"""
        if export_path is None:
            now_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"shift_inspection_summary_{now_slug}.txt"

        with self._state_lock:
            total = self.total_inspected
            passed = self.total_passed
            rejected = self.total_rejected
            yield_pct = (passed / total) * 100.0 if total > 0 else 100.0
            consec = self.consecutive_defects
            running = self.conveyor_running

        lines = [
            "================================================================",
            "        🏭 تقرير ملخص وردية فحص الجودة ومراقبة الإنتاج        ",
            "================================================================",
            f"تاريخ التقرير: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"إجمالي القطع المفحوصة (Total Inspected): {total}",
            f"القطع السليمة المطابقة (PASS):           {passed}",
            f"القطع المعيبة المرفوضة (REJECT):        {rejected}",
            f"نسبة جودة الإنتاج الإجمالية (Yield Rate): {yield_pct:.2f}%",
            f"عدد العيوب المتتالية عند الإيقاف:       {consec}",
            f"حالة خط السير الحالية:                  {'يعمل (Running)' if running else 'متوقف (Stopped)'}",
            f"سجل البيانات الكامل محفوظ في:           {self.log_file.resolve()}",
            f"مجلد صور العينات المعيبة المحفوظة:      {self.archive_dir.resolve()}",
            "================================================================",
        ]
        report_text = "\n".join(lines)
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        return export_path
