import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
import time

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS

# Add YOLOv7 repo to path so its utils can be imported
YOLOV7_ROOT = Path(__file__).parent / "yolov7"
sys.path.insert(0, str(YOLOV7_ROOT))

from yolov7.utils.datasets import letterbox
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords

@dataclass
class TrackingConfig:
    source      : str   = r"/home/duonghn/boxmot_rebuild/test_video/aicity.mp4"
    weights     : str   = "best.pt"
    detect_class: int   = 2
    tracker_type: str   = "botsort"
    device      : str   = "1"  # sử dụng cuda:1
    conf_thres  : float = 0.3
    iou_thres   : float = 0.65
    output      : str   = "final_feat_1.avi"
    img_size    : int   = 640
    half        : bool  = False
    per_class   : bool  = False

def visualize_with_logs(frame, outputs, logs, colors):
    for det in outputs:
        x1, y1, x2, y2, tid, score, *_ = det
        x1, y1, x2, y2, tid = map(int, (x1, y1, x2, y2, tid))
        color = colors[tid % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        log = next((log for log in logs if log["track_id"] == tid), None)

        if log:
            text = (
                f"cos:{log.get('cosine_dist', float('nan')):.2f} "
                f"iou:{log.get('iou', float('nan')):.2f} "
                f"d:{log.get('final_dist', float('nan')):.2f}"
            )
        else:
            text = f"ID:{tid}({score:.2f})"

        cv2.putText(frame, text, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return frame


def visualize_full_hd(video_frame, tracks, rows=6, cols=5, padding=5, text_height=20):
    target_w, target_h = 1920, 1080
    grid_h = target_h
    cell_h = ((grid_h - (rows + 1) * padding - rows * text_height) // rows)
    cell_w = int(cell_h * (2 / 3))

    grid_w = cols * (cell_w + padding) + padding
    video_w = target_w - grid_w
    video_frame_resized = cv2.resize(video_frame, (video_w, target_h))

    grid_panel = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240
    total_slots = rows * cols

    for idx in range(total_slots):
        row = idx // cols
        col = idx % cols
        x_start = padding + col * (cell_w + padding)
        y_start = padding + row * (cell_h + text_height + padding)

        if idx < len(tracks):
            track = tracks[idx]
            img_crop = cv2.resize(track["image"], (cell_w, cell_h))
            grid_panel[y_start:y_start + cell_h, x_start:x_start + cell_w] = img_crop
            text = f"ID: {track['id']}"
            cv2.putText(grid_panel, text,
                        (x_start + 3, y_start + cell_h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            cv2.rectangle(grid_panel, (x_start, y_start),
                          (x_start + cell_w, y_start + cell_h),
                          (200, 200, 200), -1)

    final_display = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    final_display[:, :video_w] = video_frame_resized
    final_display[:, video_w:] = grid_panel
    return final_display

def run_tracking(cfg: TrackingConfig) -> None:
    selected_device = torch.device("cuda:0")
    print(f"[INFO] Using device: {selected_device}")

    model = attempt_load(cfg.weights, map_location="cpu")
    model.to(selected_device)
    model.eval()
    print(f"[INFO] YOLOv7 model on: {next(model.parameters()).device}")

    tracker = create_tracker(
        tracker_type=cfg.tracker_type,
        tracker_config=TRACKER_CONFIGS / f"{cfg.tracker_type}.yaml",
        half=cfg.half,
        per_class=cfg.per_class,
        reid_weights=Path("osnet_x1_0_market1501.pt"),
        device=selected_device
    )

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {cfg.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(cfg.output, cv2.VideoWriter_fourcc(*"XVID"), fps, (1920, 1080))

    colors: List[tuple] = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 192, 192), (128, 128, 128), (64, 0, 0),
        (0, 64, 0), (0, 0, 64), (255, 165, 0),
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
        frame_idx += 1
        curr_time = time.time()
        fps_real = 1.0 / (curr_time - prev_time)
        avg_fps = frame_idx / (curr_time - start_time)  # FPS trung bình tính tại thời điểm này
        prev_time = curr_time

        img, ratio, pad = letterbox(frame, new_shape=cfg.img_size, auto=False)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(selected_device).float() / 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor, augment=False)[0]
            det = non_max_suppression(pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape, ratio_pad=(ratio, pad)).round()
            det = det[det[:, 5] == 2]  # Filter class (person)
            if len(det):
                outputs, logs, tracks_for_visual = tracker.update(det.cpu().numpy(), frame)

                logs_dict = {log["track_id"]: log for log in logs}
                full_logs = []
                for d in outputs:
                    tid = int(d[4])
                    score = float(d[5])
                    log = logs_dict.get(tid, {})
                    base_log = {
                        "track_id": tid,
                        "conf": score,
                        "cosine_dist": log.get("cosine_dist", float("nan")),
                        "iou": log.get("iou", float("nan")),
                        "final_dist": log.get("final_dist", float("nan")),
                    }
                    full_logs.append(base_log)

                frame_with_logs = visualize_with_logs(frame.copy(), outputs, full_logs, colors)
                tracks_for_visual.sort(key=lambda t: t["id"])
                display_frame = visualize_full_hd(frame_with_logs, tracks_for_visual)
            else:
                display_frame = visualize_full_hd(frame, [])
        else:
            display_frame = visualize_full_hd(frame, [])

        # === Hiển thị FPS hiện tại & FPS trung bình
        info_text = f"Frame: {frame_idx} | FPS: {fps_real:.2f}"
        avg_text = f"Avg FPS: {avg_fps:.2f}"
        cv2.putText(display_frame, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, avg_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

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
