import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch

# ✅ ép toàn bộ pipeline chạy trên GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics import YOLO
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

@dataclass
class TrackingConfig:
    source      : str   = "/home/duongnh/boxmot_rebuild/test_video/Recording 2025-07-11 144148.mp4"
    weights     : str   = "yolo11n.pt"
    detect_class: int   = 0
    tracker_type: str   = "botsort"
    conf_thres  : float = 0.25
    iou_thres   : float = 0.5
    output      : str   = "output_yolo11.avi"
    half        : bool  = False
    per_class   : bool  = False
    reid_weights: str   = "osnet_x0_75_msmt17.pt"

def run_tracking(cfg: TrackingConfig) -> None:
    print("[INFO] Initializing YOLOv8 model...")
    model = YOLO(cfg.weights)  # YOLO tự động chọn GPU:0 → chính là CUDA:1 do env

    print("[INFO] Initializing tracker on CUDA device 1 (mapped as 0 inside)...")
    tracker = create_tracker(
        tracker_type=cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half=cfg.half,
        per_class=cfg.per_class,
        reid_weights=Path(cfg.reid_weights),
        device="0"  # device=0 = cuda:1 (do đã map qua env)
    )

    print("[INFO] Opening video source...")
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    colors: List[tuple] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    print("[INFO] Start processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=cfg.conf_thres,
            iou=cfg.iou_thres,
            classes=[cfg.detect_class],
            verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            detections = []

            for box in results.boxes:
                cls = int(box.cls[0])
                if cls != cfg.detect_class:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf, cls])

            if detections:
                dets_np = np.array(detections, dtype=np.float32)
                outputs = tracker.update(dets_np, frame)

                for det in outputs:
                    x1, y1, x2, y2, tid, score, cls, _ = det
                    x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
                    color = colors[tid % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID {tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ Done! Output saved to: {cfg.output}")

if __name__ == "__main__":
    config = TrackingConfig()
    run_tracking(config)
