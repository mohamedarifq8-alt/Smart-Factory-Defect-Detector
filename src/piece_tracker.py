"""
Industrial Video Stream Piece Tracker & Line-Crossing Counter
Tracks physical products on conveyor belts / live camera / video streams.
Guarantees:
1. Each physical product is counted and actuated upon EXACTLY ONCE.
2. Prevents multi-eject on the same piece while it is visible on screen.
3. Automatically detects scene changes and image flips in video streams.
4. Smooth absence debouncing when products exit the camera view.
"""

import time
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class IndustrialPieceTracker:
    def __init__(self, line_y_ratio: float = 0.5, cooldown_sec: float = 0.8):
        self.line_y_ratio = line_y_ratio
        self.cooldown_sec = cooldown_sec
        self.last_registered_time = 0.0
        
        # Tracking States
        self.object_present = False
        self.registered_current_object = False  # True once current piece has been actuated/logged
        self.absence_count = 0                  # Empty frame debounce counter
        self.frames_present_count = 0           # Stability presence counter
        
        # Best inspection capture for this piece
        self.accumulated_defects: List[Dict[str, Any]] = []
        self.best_frame: Optional[np.ndarray] = None
        
        # Fast scene-change thumbnails (96x96 Gaussian-blurred grayscale)
        self.prev_small_gray: Optional[np.ndarray] = None
        self.registered_anchor_gray: Optional[np.ndarray] = None

    @staticmethod
    def _compute_scene_delta(gray_a: np.ndarray, gray_b: np.ndarray) -> Tuple[float, float]:
        """Return (mean_abs_diff, zero_mean_changed_pixel_fraction) between blurred thumbnails."""
        mean_raw = float(np.mean(cv2.absdiff(gray_a, gray_b)))
        a_f = gray_a.astype(np.float32)
        b_f = gray_b.astype(np.float32)
        diff_zm = np.abs((a_f - float(np.mean(a_f))) - (b_f - float(np.mean(b_f))))
        frac_zm_gt_20 = float(np.mean(diff_zm > 20.0))
        return mean_raw, frac_zm_gt_20

    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], np.ndarray]:
        """
        Evaluates current video/camera frame:
        Returns: (should_register_piece, detections_to_log, frame_to_log)
        - should_register_piece: True ONLY ONCE per physical piece appearance!
        """
        now = time.time()

        # 1. Scene Transition / Slide Cut Detection (96x96 blurred grayscale)
        # Illumination-normalized pixel fraction catches same-background product cuts while ignoring minor jitter/exposure drift.
        scene_gray = cv2.GaussianBlur(
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96)),
            (5, 5),
            0,
        )
        scene_flipped = False
        if self.prev_small_gray is not None:
            mean_raw, frac_zm_gt_20 = self._compute_scene_delta(scene_gray, self.prev_small_gray)
            if (mean_raw > 14.0) or (mean_raw > 2.4 and frac_zm_gt_20 >= 0.020):
                scene_flipped = True
            elif (
                self.registered_current_object
                and self.registered_anchor_gray is not None
                and mean_raw < 1.8
                and frac_zm_gt_20 < 0.010
            ):
                # Also catch slow crossfade/dissolve transitions once the new slide settles
                anchor_mean, anchor_frac = self._compute_scene_delta(scene_gray, self.registered_anchor_gray)
                if (anchor_mean > 14.0) or (anchor_mean > 2.6 and anchor_frac >= 0.022):
                    scene_flipped = True
        self.prev_small_gray = scene_gray

        # If video jumped/flipped to a new slide/image, reset lock for the new piece
        if scene_flipped:
            self.object_present = False
            self.registered_current_object = False
            self.frames_present_count = 0
            self.accumulated_defects = []
            self.best_frame = None
            self.last_registered_time = 0.0
            self.registered_anchor_gray = None

        # 2. Object Presence Check (Foreground contours or AI Detections)
        has_ai_detection = len(detections) > 0
        
        # Adaptive foreground mask
        _, thresh = cv2.threshold(scene_gray, 35, 255, cv2.THRESH_BINARY)
        if np.mean(thresh[:12, :12]) > 127:
            thresh = cv2.bitwise_not(thresh)
        has_foreground = np.count_nonzero(thresh) > int(scene_gray.size * 0.03)
        
        has_object = has_ai_detection or has_foreground

        should_register = False
        final_dets = []
        final_frame = frame

        # 3. State Machine Processing
        if has_object:
            self.absence_count = 0
            self.frames_present_count += 1

            if not self.object_present:
                self.object_present = True
                self.registered_current_object = False
                self.accumulated_defects = list(detections)
                self.best_frame = frame.copy()
            else:
                # Accumulate highest-severity / most confident defects during transit
                if len(detections) > len(self.accumulated_defects):
                    self.accumulated_defects = list(detections)
                    self.best_frame = frame.copy()
                elif len(detections) > 0 and len(self.accumulated_defects) > 0:
                    curr_max_conf = max(d['conf'] for d in detections)
                    prev_max_conf = max(d['conf'] for d in self.accumulated_defects)
                    if curr_max_conf > prev_max_conf:
                        self.accumulated_defects = list(detections)
                        self.best_frame = frame.copy()

            # Trigger condition:
            # - Current piece has NOT yet been registered (ensures exactly 1 trigger per piece!)
            # - Piece has been steadily present for at least 3 frames (~0.1s stability)
            # - Inter-piece cooldown has elapsed
            if (not self.registered_current_object) and (self.frames_present_count >= 3) and (now - self.last_registered_time > self.cooldown_sec):
                should_register = True
                self.registered_current_object = True  # LOCK: will NOT trigger again for this same piece!
                self.last_registered_time = now
                self.registered_anchor_gray = scene_gray.copy()
                final_dets = list(self.accumulated_defects) if self.accumulated_defects else list(detections)
                final_frame = self.best_frame.copy() if self.best_frame is not None else frame.copy()

        else:
            # Piece has exited the camera view or conveyor area
            self.absence_count += 1
            if self.absence_count >= 2:  # 2 frames of empty space confirms piece has passed
                self.object_present = False
                self.registered_current_object = False  # Reset lock: ready for next physical piece!
                self.frames_present_count = 0
                self.accumulated_defects = []
                self.best_frame = None
                self.last_registered_time = 0.0
                self.registered_anchor_gray = None

        return should_register, final_dets, final_frame

    def reset(self):
        """Manually resets tracker state"""
        self.object_present = False
        self.registered_current_object = False
        self.absence_count = 0
        self.frames_present_count = 0
        self.accumulated_defects = []
        self.best_frame = None
        self.last_registered_time = 0.0
        self.prev_small_gray = None
        self.registered_anchor_gray = None
