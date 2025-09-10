
import logging
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch

# Thêm thư mục gốc yolov7 vào sys.path để "utils.general" hoạt động
YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "yolov7")
sys.path.insert(0, YOLOV7_PATH)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from yolov7.utils.datasets      import letterbox
from yolov7.utils.torch_utils   import select_device
from yolov7.models.experimental import attempt_load
from yolov7.utils.general       import non_max_suppression, scale_coords


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    VIDEO_PATH     : str   = "rtsp://admin:88888888abc@dxtechcameranvr1.cameraddns.net:8554/Streaming/Channels/301"
    WEIGHTS_PATH   : str   = "best.pt"
    DEVICE         : str   = '0'
    CONF_THRESHOLD : float = 0.25
    IOU_THRESHOLD  : float = 0.2
    PERSON_CLASS_ID: int   = 2  # Thường là 0 với COCO
    OUTPUT_VIDEO   : str   = "output_video_2.avi"

config = Config()

device = select_device(config.DEVICE)
model  = attempt_load(config.WEIGHTS_PATH, map_location=device)
model.eval()
logger.info(f"✅ YOLO model loaded successfully on device: {device}")

cap = cv2.VideoCapture(config.VIDEO_PATH)
if not cap.isOpened():
    logger.error("❌ Không mở được video!")
    sys.exit(1)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    original_frame = frame.copy()

    try:
        img = letterbox(frame, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            detections = non_max_suppression(
                pred,
                conf_thres=config.CONF_THRESHOLD,
                iou_thres=config.IOU_THRESHOLD
            )

        for det in detections:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                for i, (*xyxy, conf, cls) in enumerate(det):
                    if int(cls.item()) == config.PERSON_CLASS_ID:
                        x1, y1, x2, y2 = map(int, xyxy)
                        color = colors[i % len(colors)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        logger.info(f"✅ Processed frame {frame_idx}")

    except Exception as e:
        logger.error(f"❌ Error at frame {frame_idx}: {e}")
        continue

cap.release()
out.release()
logger.info(f"🎥 Output video saved: {config.OUTPUT_VIDEO}")
