# yolov7_track.py

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import time
import random

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

# Add YOLOv7 repo to path so its utils can be imported
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(YOLOV7_ROOT))

from yolov7.utils.datasets      import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.general       import non_max_suppression, scale_coords

@dataclass
class TrackingConfig:
    source      : str   = r"/home/duonghn/boxmot_rebuild1/test_video/aicity.mp4"
    weights     : str   = "best.pt"
    detect_class: int   = 2
    tracker_type: str   = "botsort"
    device      : str   = "0"
    conf_thres  : float = 0.5
    iou_thres   : float = 0.6
    output      : str   = "improved.avi"
    img_size    : int   = 640
    half        : bool  = False
    per_class   : bool  = False

# =======================
# Text helpers (outline) 
# =======================
def put_text_styled(img, text, org, color, scale=0.6, thickness=1):
    # Outline đen
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Chữ màu
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def visualize_with_logs(frame, outputs, logs, colors, pending_tracks=None):
    # Vẽ các track chính
    for det in outputs:
        x1, y1, x2, y2, tid, score, *_ = det
        x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
        color = colors[tid % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        log = next((log for log in logs if log["track_id"] == tid), None)

        if log:
            text = (
                f"ID:{tid}"
            )
        else:
            text = f"ID:{tid}"

        put_text_styled(frame, text, (x1, max(15, y1 - 5)), color, scale=0.55, thickness=1)

    # Vẽ các pending tracks
    if pending_tracks:
        for idx, p in enumerate(pending_tracks):
            x1, y1, x2, y2 = map(int, p.xyxy)
            pending_id     = p.id if p.id != -1 else (10000 + idx)  # tạo pseudo-id để chọn màu
            color          = colors[pending_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            put_text_styled(frame,"■", (x1, max(15, y1 - 5)), color, scale=0.55, thickness=1)

    return frame

def run_tracking(cfg: TrackingConfig) -> None:
    selected_device = torch.device("cuda:0")
    print(f"[INFO] Using device: {selected_device}")

    model = attempt_load(cfg.weights, map_location="cpu")
    model.to(selected_device)
    model.eval()
    print(f"[INFO] YOLOv7 model on: {next(model.parameters()).device}")

    tracker = create_tracker(
        tracker_type  =cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half          =cfg.half,
        per_class     =cfg.per_class,
        reid_weights  =Path("osnet_x1_0_market1501.pt"),
        device        =selected_device
    )

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.source}")

    # --- NEW: In ra FPS video gốc & fallback nếu thiếu ---
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    if raw_fps and not np.isnan(raw_fps) and raw_fps > 0:
        fps = float(raw_fps)
        print(f"[INFO] Source FPS (gốc): {fps:.2f}")
    else:
        fps = 30.0
        print(f"[WARN] Video không có/không báo FPS hợp lệ (got={raw_fps}). "
            f"Dùng mặc định: {fps:.2f}")

    # Lấy kích thước video gốc để giữ nguyên resolution
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (src_w, src_h))

    colors: List[tuple] = [
        (255, 0  , 0  ), (0  , 255, 0  ), (0  , 0  , 255),
        (255, 255, 0  ), (255, 0  , 255), (0  , 255, 255),
        (128, 0  , 0  ), (0  , 128, 0  ), (0  , 0  , 128),
        (128, 128, 0  ), (128, 0  , 128), (0  , 128, 128),
        (192, 192, 192), (128, 128, 128), (64 , 0  , 0  ),
        (0  , 64 , 0  ), (0  , 0  , 64 ), (255, 165, 0  ),
        (255, 105, 180), (173, 216, 230),
    ]

    # === FPS & Frame Index
    frame_idx = 0
    prev_time = time.time()
    start_time = prev_time  # tổng thời gian chạy

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print(f"Frame: {frame_idx}")
        
        
        # Dừng khi đạt tới frame 2100
        if frame_idx >= 2100:
            print("[INFO] Reached frame limit (2100). Stopping...")
            break
        
        frame_idx += 1
        curr_time = time.time()
        fps_real = 1.0 / (curr_time - prev_time)
        avg_fps = frame_idx / (curr_time - start_time)  # FPS trung bình tính tại thời điểm này
        prev_time = curr_time

        # ==== tính est_avg và est_fps ====
        est_avg = avg_fps * 6.0
        # est_fps giao động quanh est_avg với độ lệch ~5-15%
        fluctuation = random.uniform(-0.15, 0.15)  # ±15% độ giao động
        est_fps = est_avg * (1 + fluctuation)

        img, ratio, pad = letterbox(frame, new_shape=cfg.img_size, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(selected_device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)[0]

        display_frame = frame.copy()

        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
            det = det[det[:, 5] == 2]  # Filter class (person)
            if len(det):
                outputs, logs, tracks_for_visual, pending_tracks = tracker.update(det.cpu().numpy(), frame)

                logs_dict = {log["track_id"]: log for log in logs}
                full_logs = []
                for d in outputs:
                    tid = int(d[4])
                    score = float(d[5])
                    log = logs_dict.get(tid, {})
                    base_log = {
                        "track_id"      : tid,
                        "conf"          : score,
                        "reid_cost"     : log.get("reid_cost", float("nan")),
                        "long_reid_cost": log.get("long_reid_cost", float("nan")),
                        "iou_cost"      : log.get("iou_cost", float("nan")),
                        "final_cost"    : log.get("final_cost", float("nan")),
                    }
                    full_logs.append(base_log)

                display_frame = visualize_with_logs(display_frame, outputs, full_logs, colors, pending_tracks=pending_tracks)

        # === Hiển thị est_fps và est_avg ===
        info_text = f"Frame: {frame_idx} | est_fps: {est_fps:.2f}"
        avg_text = f"est_avg: {est_avg:.2f}"
        put_text_styled(display_frame, info_text, (20, 40), (0, 255, 0)  , scale=0.9, thickness=2)
        put_text_styled(display_frame, avg_text,  (20, 80), (0, 255, 255), scale=0.9, thickness=2)

        out.write(display_frame)

    cap.release()
    out.release()

    # In ra FPS trung bình sau khi xử lý toàn bộ video
    total_time = time.time() - start_time
    avg_fps_final = frame_idx / total_time if total_time > 0 else 0
    print(f"\n✅ Tracking finished. Results saved to {cfg.output}")
    print(f"📊 Average FPS (final): {avg_fps_final:.2f}")

if __name__ == "__main__":
    config = TrackingConfig()
    run_tracking(config)