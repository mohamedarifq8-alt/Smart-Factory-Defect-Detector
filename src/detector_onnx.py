"""
Industrial Defect Detection - ONNX Runtime Engine
Optimized for YOLO11 Object Detection at 60+ FPS on CPU/GPU
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Dict, Any


class YOLO11ONNXDetector:
    def __init__(self, model_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = ['defect']

        # Initialize ONNX Runtime Session with optimized providers
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        
        # Get Model Input details
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [1, 3, 640, 640]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        # Get Model Output details
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize image to 640x640 with padding for high accuracy"""
        orig_h, orig_w = img.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create 640x640 blank canvas (114 is standard YOLO padding color)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR to RGB, Normalize to [0, 1], Transpose to [1, 3, H, W]
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return blob, scale, (pad_x, pad_y)

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int], 
                    scale: float, pad: Tuple[int, int], conf_thresh: float) -> List[Dict[str, Any]]:
        """
        Parse YOLO11 output [1, 5, 8400]
        Channels: [x_c, y_c, w, h, class0_score]
        """
        preds = outputs[0]  # shape: [5, 8400]
        preds = np.transpose(preds, (1, 0))  # shape: [8400, 5]

        boxes = []
        confidences = []
        class_ids = []

        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape

        # Filter by confidence threshold
        scores = preds[:, 4]  # defect score
        mask = scores > conf_thresh
        valid_preds = preds[mask]
        valid_scores = scores[mask]

        for i in range(len(valid_preds)):
            x_c, y_c, w, h, _ = valid_preds[i]
            score = float(valid_scores[i])

            # Convert to canvas box coordinates
            x1 = x_c - w / 2.0
            y1 = y_c - h / 2.0

            # Undo padding and scale back to original image
            x1_orig = (x1 - pad_x) / scale
            y1_orig = (y1 - pad_y) / scale
            w_orig = w / scale
            h_orig = h / scale

            # Clip to image boundaries
            x1_orig = max(0, min(orig_w, x1_orig))
            y1_orig = max(0, min(orig_h, y1_orig))
            w_orig = max(1, min(orig_w - x1_orig, w_orig))
            h_orig = max(1, min(orig_h - y1_orig, h_orig))

            boxes.append([int(x1_orig), int(y1_orig), int(w_orig), int(h_orig)])
            confidences.append(score)
            class_ids.append(0)

        # Non-Maximum Suppression (NMS)
        if len(boxes) == 0:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, self.iou_thresh)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                detections.append({
                    'box': box,  # [x, y, w, h]
                    'conf': confidences[idx],
                    'class_id': class_ids[idx],
                    'label': self.classes[class_ids[idx]]
                })

        return detections

    def detect(self, img: np.ndarray, conf_thresh: float = None) -> List[Dict[str, Any]]:
        """Run detection on a single BGR image"""
        if conf_thresh is None:
            conf_thresh = self.conf_thresh

        blob, scale, pad = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        detections = self.postprocess(outputs, img.shape[:2], scale, pad, conf_thresh)
        return detections

    def draw_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw high-visibility red bounding boxes and badges on defects"""
        annotated = img.copy()
        for det in detections:
            x, y, w, h = det['box']
            conf = det['conf']
            label = f"{det['label']} {conf * 100:.1f}%"

            # Draw glowing double border for industrial visibility
            cv2.rectangle(annotated, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 180), 2)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Draw filled label badge
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            badge_y = max(y, lbl_h + 8)
            cv2.rectangle(annotated, (x, badge_y - lbl_h - 6), (x + lbl_w + 6, badge_y), (0, 0, 230), -1)
            cv2.putText(annotated, label, (x + 3, badge_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return annotated
