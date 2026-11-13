"""
Industrial Quality Control - Hybrid Vision Engine (Upgraded)
Features:
1. YOLO11 Deep Learning (Cracks, Scratches, Holes, Broken parts)
2. OpenCV Color Anomaly Filter (Red spots, Rust, Oil, Discoloration) - with product profile isolation
3. OpenCV Geometric Convexity & Squeeze Analysis (Indentation, Dents) - crash-proof on OpenCV 4 & 5
4. Capsule Dimension & Aspect Ratio Verification (Length, Width, Ratio, Area)
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from src.detector_onnx import YOLO11ONNXDetector


class HybridQualityInspector:
    def __init__(self, onnx_model_path: str, conf_thresh: float = 0.25):
        # Core YOLO11 ONNX Detector
        self.yolo_detector = YOLO11ONNXDetector(onnx_model_path, conf_thresh=conf_thresh)
        self.conf_thresh = conf_thresh

    def _prepare_cv_image(self, frame: np.ndarray, max_dim: int = 1280) -> Tuple[np.ndarray, float]:
        """Downscale very large frames (> max_dim) for OpenCV branches so 4K feeds remain fast."""
        h, w = frame.shape[:2]
        longest = max(h, w)
        if longest <= max_dim:
            return frame, 1.0
        scale = max_dim / float(longest)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), (1.0 / scale)

    def _append_non_overlapping(
        self,
        target: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        inv_scale: float,
        orig_w: int,
        orig_h: int,
    ) -> None:
        """Rescale candidate boxes back to original frame coordinates and merge without overlap."""
        existing_boxes = [d['box'] for d in target]
        for det in candidates:
            if inv_scale != 1.0:
                x, y, bw, bh = det['box']
                sx = max(0, min(orig_w - 1, int(round(x * inv_scale))))
                sy = max(0, min(orig_h - 1, int(round(y * inv_scale))))
                sw = max(1, min(orig_w - sx, int(round(bw * inv_scale))))
                sh = max(1, min(orig_h - sy, int(round(bh * inv_scale))))
                det = {**det, 'box': [sx, sy, sw, sh]}

            if not self._is_overlapping(det['box'], existing_boxes):
                target.append(det)
                existing_boxes.append(det['box'])

    def detect_all(self, frame: np.ndarray, 
                   profile: str = "general",
                   enable_color_check: bool = True, 
                   enable_shape_check: bool = True,
                   conf_thresh: float = None) -> List[Dict[str, Any]]:
        """
        Runs complete hybrid inspection pipeline tailored to the product profile:
        - 'capsule': YOLO + Squeeze + Size/Aspect Ratio (Color check disabled to prevent false alarms on natural orange/red capsule shells).
        - 'metal_nut': YOLO + Color Anomaly (Rust/Red spots) + Bent check.
        - 'general': User-configured toggles.
        """
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        all_defects: List[Dict[str, Any]] = []
        orig_h, orig_w = frame.shape[:2]

        # -------------------------------------------------------------
        # Branch 1: YOLO11 Object Detection (Deep Learning)
        # -------------------------------------------------------------
        try:
            yolo_dets = self.yolo_detector.detect(frame, conf_thresh=conf_thresh)
            all_defects.extend(yolo_dets)
        except Exception as e:
            print(f"YOLO detection error: {e}")

        # Prepare shared OpenCV frame once (only downscaled if > 1280px)
        cv_frame = None
        inv_scale = 1.0
        if (enable_color_check and profile != "capsule") or enable_shape_check:
            cv_frame, inv_scale = self._prepare_cv_image(frame)

        # -------------------------------------------------------------
        # Branch 2: Color Anomaly Filter (Red spots, Rust, Stains)
        # -------------------------------------------------------------
        # In capsule mode, color check is skipped so natural capsule shell colors aren't flagged as rust!
        if enable_color_check and profile != "capsule" and cv_frame is not None:
            try:
                color_dets = self.detect_color_anomalies(cv_frame)
                self._append_non_overlapping(all_defects, color_dets, inv_scale, orig_w, orig_h)
            except Exception as e:
                print(f"Color check error: {e}")

        # -------------------------------------------------------------
        # Branch 3: Geometric Squeeze, Dent & Size Verification
        # -------------------------------------------------------------
        if enable_shape_check and cv_frame is not None:
            try:
                shape_dets = self.detect_shape_and_size_anomalies(cv_frame, profile=profile)
                self._append_non_overlapping(all_defects, shape_dets, inv_scale, orig_w, orig_h)
            except Exception as e:
                print(f"Shape/Size check error: {e}")

        return all_defects

    def detect_color_anomalies(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects abnormal red, rust, oil or colored spots on metallic parts
        using HSV color space thresholding and morphological filtering.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        # Red color wraps around 0 and 180 in HSV
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([168, 80, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_defects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Spot must be prominent (greater than 40 pixels, but not the entire background)
            if 40 < area < (h * w * 0.35):
                x, y, bw, bh = cv2.boundingRect(cnt)
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                w1 = min(w - x1, bw + pad * 2)
                h1 = min(h - y1, bh + pad * 2)

                confidence = min(0.96, 0.70 + (area / 1500.0) * 0.25)
                color_defects.append({
                    'box': [int(x1), int(y1), int(w1), int(h1)],
                    'conf': float(confidence),
                    'class_id': 1,
                    'label': 'color_defect (بقعة لون/صدأ)'
                })

        return color_defects

    def detect_shape_and_size_anomalies(self, img: np.ndarray, profile: str = "general") -> List[Dict[str, Any]]:
        """
        Detects indentations, squeezes, dents, and size/aspect ratio deformities.
        Crash-proof implementation supporting OpenCV 4.x and 5.x.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is white
        if np.mean(thresh[:15, :15]) > 127:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        # Find primary product contour
        main_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_cnt)
        if area < 3000:
            return []

        defects_list = []

        # ---------------------------------------------------------
        # Check A: Capsule Dimensions & Aspect Ratio (Size Defect)
        # ---------------------------------------------------------
        rect = cv2.minAreaRect(main_cnt)
        (cx, cy), (rw, rh), angle = rect
        length = max(rw, rh)
        width = min(rw, rh)
        aspect_ratio = length / max(1.0, width)

        if profile == "capsule":
            # Normal capsule aspect ratio is typically between 2.6 and 3.3
            # If capsule is squashed, cut, broken, or size is distorted:
            if aspect_ratio < 2.35 or aspect_ratio > 3.6 or length < 180:
                bx, by, bw, bh = cv2.boundingRect(main_cnt)
                defects_list.append({
                    'box': [int(bx), int(by), int(bw), int(bh)],
                    'conf': 0.88,
                    'class_id': 3,
                    'label': f'size_defect (خلل في الأبعاد {aspect_ratio:.1f})'
                })

        # ---------------------------------------------------------
        # Check B: Convexity Squeeze & Dent Analysis
        # ---------------------------------------------------------
        try:
            cnt_approx = cv2.approxPolyDP(main_cnt, 0.003 * cv2.arcLength(main_cnt, True), True)
            if len(cnt_approx) >= 4:
                hull = cv2.convexHull(cnt_approx, returnPoints=False)
                if hull is not None and len(hull) >= 3:
                    cd = cv2.convexityDefects(cnt_approx, hull)
                    if cd is not None:
                        # Reshape to (-1, 4) to ensure compatibility with OpenCV 4 & 5
                        cd_matrix = cd.reshape(-1, 4)
                        for s_idx, e_idx, f_idx, d in cd_matrix:
                            depth = d / 256.0
                            # Significant indentation / squeeze (between 8 and 90 pixels)
                            if 8.0 < depth < 90.0:
                                far_pt = tuple(cnt_approx[f_idx][0])
                                bx = max(0, far_pt[0] - 25)
                                by = max(0, far_pt[1] - 25)
                                bw = min(w - bx, 50)
                                bh = min(h - by, 50)

                                conf = min(0.95, 0.65 + (depth / 40.0) * 0.25)
                                defects_list.append({
                                    'box': [int(bx), int(by), int(bw), int(bh)],
                                    'conf': float(conf),
                                    'class_id': 2,
                                    'label': 'squeeze_defect (انبعاج/تشوه)'
                                })
        except Exception as e:
            print(f"Safe convexity handling: {e}")

        return defects_list

    def _is_overlapping(self, box1: List[int], boxes: List[List[int]], iou_thresh: float = 0.3) -> bool:
        """Check if box1 significantly overlaps with any box in boxes"""
        x1, y1, w1, h1 = box1
        for b in boxes:
            x2, y2, w2, h2 = b
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            if box1_area > 0 and (inter_area / box1_area) > iou_thresh:
                return True
        return False

    def draw_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections with distinct color badges according to defect type"""
        annotated = img.copy()
        for det in detections:
            x, y, w, h = det['box']
            conf = det['conf']
            label = det['label']
            text = f"{label} {conf * 100:.0f}%"

            # Color coding:
            # Color defect -> Orange [0, 100, 255]
            # Squeeze defect -> Magenta [255, 0, 200]
            # Size defect -> Yellow [0, 220, 255]
            # Deep Learning YOLO -> Red [0, 0, 255]
            if 'color' in label:
                box_color = (0, 100, 255)
            elif 'squeeze' in label:
                box_color = (255, 0, 200)
            elif 'size' in label:
                box_color = (0, 220, 255)
            else:
                box_color = (0, 0, 255)

            # Draw double border for high visibility
            cv2.rectangle(annotated, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 0), 2)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Badge background
            (lbl_w, lbl_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            badge_y = max(y, lbl_h + 8)
            cv2.rectangle(annotated, (x, badge_y - lbl_h - 6), (x + lbl_w + 6, badge_y), box_color, -1)
            cv2.putText(annotated, text, (x + 3, badge_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated
