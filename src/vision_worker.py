"""
Industrial Quality Control - Asynchronous Vision Capture & ONNX Inference Engine
Architecture:
- Dedicated background Producer-Consumer pipeline decoupling cv2.VideoCapture,
  ONNX Runtime (session.run), OpenCV hybrid filters, and defect archiving from Tkinter's Main Thread.
- Live Cameras (USB/RTSP/HTTP): Dedicated Grabber Thread + 1-slot newest-frame buffer
  to eliminate driver queue lag without blocking UI or accumulating stale frames.
- Recorded Video Files: Frame-by-frame background inference paced by source FPS
  so short test clips play smoothly without skipping pieces or racing at 300+ FPS.
- Zero-Lag Preview Preparation: Resizes display preview + converts BGR->RGB in background
  so Tkinter only uploads a small PhotoImage to the Canvas.
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any, Tuple


@dataclass
class FrameResult:
    """Latest processed frame package ready for Tkinter rendering."""
    seq: int
    raw_frame: np.ndarray
    rgb_preview: np.ndarray
    detections: List[Dict[str, Any]]
    inference_time_ms: float
    fps: float
    paused: bool = False


def is_live_camera_source(source: Any) -> bool:
    """Return True if video source is a live camera index or network stream."""
    if isinstance(source, int):
        return True
    s = str(source).strip()
    if s.isdigit():
        return True
    lower = s.lower()
    return lower.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def prepare_rgb_preview(bgr_img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize BGR image to fit target Canvas dimensions and convert to RGB."""
    canvas_w, canvas_h = target_size
    if canvas_w < 100 or canvas_h < 100:
        canvas_w, canvas_h = 750, 500

    h, w = bgr_img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def execute_piece_inspection(
    get_inspector_fn: Callable[[], Any],
    inference_lock: threading.Lock,
    controller: Any,
    frame: np.ndarray,
    settings: Tuple[float, str, bool, bool],
    get_canvas_size_fn: Callable[[], Tuple[int, int]],
    require_running: bool = False,
    expected_epoch: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run hybrid AI detection + production registration off the Tkinter Main Thread."""
    conf, profile, enable_color, enable_shape = settings

    with inference_lock:
        inspector = get_inspector_fn()
        if inspector is None:
            return None
        detections = inspector.detect_all(
            frame,
            profile=profile,
            enable_color_check=enable_color,
            enable_shape_check=enable_shape,
            conf_thresh=conf,
        )
        annotated_frame = inspector.draw_detections(frame, detections)

    result = controller.register_inspection(
        annotated_frame,
        detections,
        require_running=require_running,
        expected_epoch=expected_epoch,
    )
    if result is None:
        return None

    rgb_preview = prepare_rgb_preview(annotated_frame, get_canvas_size_fn())
    return {
        'raw_frame': frame,
        'annotated_frame': annotated_frame,
        'rgb_preview': rgb_preview,
        'detections': detections,
        'result': result,
    }


class StreamVisionWorker:
    """
    Runs video/camera capture, ONNX inference, piece tracking, and preview scaling
    on background thread(s), exposing latest FrameResult & events cleanly to Tkinter.
    """

    def __init__(
        self,
        source: Any,
        cap: cv2.VideoCapture,
        get_inspector_fn: Callable[[], Any],
        inference_lock: threading.Lock,
        tracker: Any,
        controller: Any,
        get_settings_fn: Callable[[], Tuple[float, str, bool, bool]],
        get_canvas_size_fn: Callable[[], Tuple[int, int]],
    ):
        self.source = source
        self.cap = cap
        self.is_live_camera = is_live_camera_source(source)
        self.get_inspector_fn = get_inspector_fn
        self.inference_lock = inference_lock
        self.tracker = tracker
        self.controller = controller
        self.get_settings_fn = get_settings_fn
        self.get_canvas_size_fn = get_canvas_size_fn

        self._stop_event = threading.Event()
        self._capture_cond = threading.Condition()
        self._latest_captured_frame: Optional[np.ndarray] = None
        self._capture_eof = False

        self._state_lock = threading.Lock()
        self._latest_result: Optional[FrameResult] = None
        self._last_consumed_seq = 0
        self._registered_events: deque = deque()
        self._worker_eof = False
        self._frame_seq = 0

        self._grab_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch capture (if live camera) and inference background threads."""
        if self.is_live_camera:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._grab_thread = threading.Thread(
                target=self._grab_loop,
                name="CameraGrabberThread",
                daemon=True,
            )
            self._grab_thread.start()

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="VisionInferenceWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal worker to stop and wait cleanly for threads & VideoCapture release."""
        self._stop_event.set()
        with self._capture_cond:
            self._capture_cond.notify_all()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def pop_latest_state(self) -> Tuple[Optional[FrameResult], List[Dict[str, Any]], bool]:
        """
        Non-blocking poll called by Tkinter Main Thread.
        Returns (new_frame_result_or_none, registered_piece_events, eof_reached).
        """
        with self._state_lock:
            result = None
            if (
                self._latest_result is not None
                and self._latest_result.seq != self._last_consumed_seq
            ):
                result = self._latest_result
                self._last_consumed_seq = result.seq

            events = list(self._registered_events)
            self._registered_events.clear()
            eof = self._worker_eof
            return result, events, eof

    def _grab_loop(self) -> None:
        """Continuously drains live camera driver buffer and keeps only newest frame."""
        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                with self._capture_cond:
                    if not ret or frame is None:
                        self._capture_eof = True
                        self._capture_cond.notify_all()
                        break
                    self._latest_captured_frame = frame
                    self._capture_cond.notify_all()
        except Exception as e:
            print(f"Camera grabber error: {e}")
        finally:
            with self._capture_cond:
                self._capture_eof = True
                self._capture_cond.notify_all()

    def _read_next_frame(self) -> Tuple[str, Optional[np.ndarray]]:
        """Retrieve next frame from live grabber buffer or directly from video file."""
        if not self.is_live_camera:
            if self._stop_event.is_set():
                return "STOP", None
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return "EOF", None
            return "FRAME", frame

        with self._capture_cond:
            deadline = time.perf_counter() + 0.2
            while (
                not self._stop_event.is_set()
                and self._latest_captured_frame is None
                and not self._capture_eof
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return "TIMEOUT", None
                self._capture_cond.wait(timeout=remaining)

            if self._stop_event.is_set():
                return "STOP", None

            if self._latest_captured_frame is not None:
                frame = self._latest_captured_frame
                self._latest_captured_frame = None
                return "FRAME", frame

            if self._capture_eof:
                return "EOF", None

            return "TIMEOUT", None

    def _get_file_frame_period(self) -> float:
        """Calculate natural frame pacing interval for recorded video files."""
        if self.is_live_camera:
            return 0.0
        try:
            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            src_fps = 0.0
        if not np.isfinite(src_fps) or src_fps < 1.0 or src_fps > 120.0:
            src_fps = 25.0
        return max(0.015, min(0.1, 1.0 / src_fps))

    def _worker_loop(self) -> None:
        """Main background inference & piece-tracking loop."""
        file_frame_period = self._get_file_frame_period()
        fps_tracker = 0.0
        fps_counter = 0
        last_fps_time = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                iter_start = time.perf_counter()
                status, frame = self._read_next_frame()

                if status == "STOP":
                    break
                if status == "TIMEOUT":
                    continue
                if status == "EOF" or frame is None:
                    with self._state_lock:
                        self._worker_eof = True
                    break

                epoch = self.controller.get_state_epoch()
                running = self.controller.is_conveyor_running()

                if running:
                    conf, profile, color_chk, shape_chk = self.get_settings_fn()
                    t0 = time.perf_counter()

                    with self.inference_lock:
                        inspector = self.get_inspector_fn()
                        if inspector is None:
                            self._stop_event.wait(0.05)
                            continue
                        detections = inspector.detect_all(
                            frame,
                            profile=profile,
                            enable_color_check=color_chk,
                            enable_shape_check=shape_chk,
                            conf_thresh=conf,
                        )
                        annotated_frame = inspector.draw_detections(frame, detections)

                    inference_time_ms = (time.perf_counter() - t0) * 1000.0

                    fps_counter += 1
                    now = time.perf_counter()
                    if now - last_fps_time >= 1.0:
                        fps_tracker = fps_counter / (now - last_fps_time)
                        fps_counter = 0
                        last_fps_time = now

                    h, w = annotated_frame.shape[:2]
                    cv2.line(annotated_frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
                    cv2.putText(
                        annotated_frame,
                        "VIRTUAL INSPECTION LINE",
                        (10, h // 2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 0),
                        1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Latency: {inference_time_ms:.1f} ms | FPS: {fps_tracker:.1f}",
                        (max(10, w - 240), 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

                    registered_result = None
                    if (
                        not self._stop_event.is_set()
                        and self.controller.is_conveyor_running()
                        and self.controller.get_state_epoch() == epoch
                    ):
                        should_register, reg_dets, reg_frame = self.tracker.process_frame(frame, detections)
                        if should_register:
                            with self.inference_lock:
                                curr_inspector = self.get_inspector_fn()
                                reg_annotated = (
                                    curr_inspector.draw_detections(reg_frame, reg_dets)
                                    if curr_inspector is not None
                                    else reg_frame
                                )
                            registered_result = self.controller.register_inspection(
                                reg_annotated,
                                reg_dets,
                                require_running=True,
                                expected_epoch=epoch,
                            )

                    self.controller.update_actuator_state()
                    rgb_preview = prepare_rgb_preview(annotated_frame, self.get_canvas_size_fn())
                    paused = False
                else:
                    paused_frame = frame.copy()
                    cv2.putText(
                        paused_frame,
                        "CONVEYOR PAUSED / STOPPED",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                    rgb_preview = prepare_rgb_preview(paused_frame, self.get_canvas_size_fn())
                    detections = []
                    inference_time_ms = 0.0
                    registered_result = None
                    paused = True

                self._frame_seq += 1
                packet = FrameResult(
                    seq=self._frame_seq,
                    raw_frame=frame,
                    rgb_preview=rgb_preview,
                    detections=detections,
                    inference_time_ms=inference_time_ms,
                    fps=fps_tracker,
                    paused=paused,
                )

                with self._state_lock:
                    self._latest_result = packet
                    if registered_result is not None:
                        self._registered_events.append(registered_result)

                if not self.is_live_camera and file_frame_period > 0.0:
                    elapsed = time.perf_counter() - iter_start
                    remaining = file_frame_period - elapsed
                    if remaining > 0.0:
                        self._stop_event.wait(remaining)

        except Exception as e:
            print(f"StreamVisionWorker error: {e}")
            with self._state_lock:
                self._worker_eof = True
        finally:
            if self._grab_thread is not None and self._grab_thread.is_alive():
                self._grab_thread.join(timeout=1.5)
            # Only release VideoCapture when grabber thread is no longer inside cap.read()
            if self._grab_thread is None or not self._grab_thread.is_alive():
                try:
                    self.cap.release()
                except Exception:
                    pass
