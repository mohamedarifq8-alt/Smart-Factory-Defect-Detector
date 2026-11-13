"""
Industrial Quality Control: Real-Time Defect Detection with YOLO11
Kaggle Training & ONNX Export Script
"""

import os
import cv2
import shutil
import yaml
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

print("🚀 Step 1: Installing and setting up environment...")

# Locate MVTec Dataset inside Kaggle /kaggle/input
kaggle_input = Path('/kaggle/input')
mvtec_root = None

for root, dirs, files in os.walk(kaggle_input):
    if 'bottle' in dirs and 'metal_nut' in dirs:
        mvtec_root = Path(root)
        break

if mvtec_root is None:
    possible_paths = list(kaggle_input.glob('**/mvtec*')) + list(kaggle_input.glob('*'))
    for p in possible_paths:
        if (p / 'bottle').exists():
            mvtec_root = p
            break

print(f"✅ MVTec Root found at: {mvtec_root}")
assert mvtec_root is not None, "❌ MVTec dataset not found in /kaggle/input. Please add MVTec AD dataset to inputs!"

# Selected 3 diverse industrial products
TARGET_PRODUCTS = ['bottle', 'metal_nut', 'capsule']
YOLO_DATA_DIR = Path('/kaggle/working/yolo_mvtec_dataset')

if YOLO_DATA_DIR.exists():
    shutil.rmtree(YOLO_DATA_DIR)

for split in ['train', 'val']:
    (YOLO_DATA_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DATA_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def mask_to_yolo_boxes(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:  # filter noise
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x_c = (x + w / 2.0) / img_w
        y_c = (y + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        boxes.append((0, x_c, y_c, norm_w, norm_h))
    return boxes

sample_count = {'train': 0, 'val': 0}

print("🔄 Converting MVTec masks to YOLO Bounding Boxes automatically...")
for product in TARGET_PRODUCTS:
    prod_dir = mvtec_root / product
    print(f" -> Processing {product}...")
    
    # 1. Good normal images
    train_good_dir = prod_dir / 'train' / 'good'
    if train_good_dir.exists():
        for img_p in list(train_good_dir.glob('*.png'))[:70]:
            dst_name = f"{product}_good_{img_p.stem}.png"
            shutil.copy(img_p, YOLO_DATA_DIR / 'images' / 'train' / dst_name)
            (YOLO_DATA_DIR / 'labels' / 'train' / f"{product}_good_{img_p.stem}.txt").touch()
            sample_count['train'] += 1
            
    # 2. Defect images + masks
    test_dir = prod_dir / 'test'
    gt_dir = prod_dir / 'ground_truth'
    if test_dir.exists():
        for defect_type_dir in test_dir.iterdir():
            if not defect_type_dir.is_dir():
                continue
            defect_name = defect_type_dir.name
            defect_images = list(defect_type_dir.glob('*.png'))
            
            if defect_name == 'good':
                for img_p in defect_images[:20]:
                    dst_name = f"{product}_val_good_{img_p.stem}.png"
                    shutil.copy(img_p, YOLO_DATA_DIR / 'images' / 'val' / dst_name)
                    (YOLO_DATA_DIR / 'labels' / 'val' / f"{product}_val_good_{img_p.stem}.txt").touch()
                    sample_count['val'] += 1
            else:
                gt_defect_dir = gt_dir / defect_name
                for idx, img_p in enumerate(defect_images):
                    split = 'train' if (idx % 5 != 0) else 'val'
                    mask_candidates = [
                        gt_defect_dir / f"{img_p.stem}_mask.png",
                        gt_defect_dir / f"{img_p.stem}.png"
                    ]
                    mask_p = next((m for m in mask_candidates if m.exists()), None)
                    
                    img = cv2.imread(str(img_p))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    boxes = mask_to_yolo_boxes(mask_p, w, h) if mask_p else []
                    
                    if len(boxes) > 0:
                        dst_name = f"{product}_{defect_name}_{img_p.stem}.png"
                        shutil.copy(img_p, YOLO_DATA_DIR / 'images' / split / dst_name)
                        lbl_file = YOLO_DATA_DIR / 'labels' / split / f"{product}_{defect_name}_{img_p.stem}.txt"
                        with open(lbl_file, 'w') as f:
                            for box in boxes:
                                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                        sample_count[split] += 1

print(f"✅ Dataset prepared: {sample_count['train']} train, {sample_count['val']} val images.")

# Write dataset YAML
yaml_config = {
    'path': str(YOLO_DATA_DIR),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'defect'}
}
yaml_path = YOLO_DATA_DIR / 'mvtec_qc.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)

# Start YOLO11 Transfer Learning
print("🧠 Training YOLO11 Small with Transfer Learning on GPU...")
model = YOLO('yolo11s.pt')
model.train(
    data=str(yaml_path),
    epochs=35,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    name='mvtec_defect_detector',
    patience=10,
    save=True
)

# Export to ONNX for Controller Deployment
print("📦 Exporting to ONNX format...")
best_pt_path = Path('/kaggle/working/runs/detect/mvtec_defect_detector/weights/best.pt')
best_model = YOLO(str(best_pt_path))
onnx_file = best_model.export(format='onnx', dynamic=False, imgsz=640, simplify=True)

deploy_dir = Path('/kaggle/working/industrial_qc_deployment')
deploy_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(best_pt_path, deploy_dir / 'best_defect_detector.pt')
shutil.copy(onnx_file, deploy_dir / 'best_defect_detector.onnx')

qc_config = {
    'model_name': 'YOLO11-Industrial-Defect-Detector',
    'input_size': [640, 640],
    'confidence_threshold': 0.35,
    'classes': ['defect'],
    'target_products': TARGET_PRODUCTS,
    'decision_logic': {
        'no_defect_detected': 'PASS',
        'defect_detected': 'REJECT'
    }
}
with open(deploy_dir / 'qc_config.json', 'w') as f:
    json.dump(qc_config, f, indent=4)

shutil.make_archive('/kaggle/working/industrial_qc_model_package', 'zip', deploy_dir)
print("\n🎉 SUCCESS! Package ready at: /kaggle/working/industrial_qc_model_package.zip")
