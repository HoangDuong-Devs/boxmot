from boxmot.engine.track import main as run_track
from boxmot.utils import WEIGHTS
import sys
from pathlib import Path

class Args:
    yolo_model = Path(WEIGHTS / "yolov8n.pt")
    reid_model = Path(WEIGHTS / "osnet_x0_25_msmt17.pt")
    source = "data/videos/demo.mp4"

    imgsz = [640, 640]
    conf = 0.3
    iou = 0.7
    device = "0"
    classes = [0]

    project = Path("runs")
    name = "python_track_run"
    exist_ok = True
    half = False
    vid_stride = 1
    ci = False

    tracking_method = "deepocsort"
    dets_file_path = None
    embs_file_path = None
    exp_folder_path = None

    verbose = True
    agnostic_nms = False
    gsi = False
    n_trials = 4
    objectives = ["HOTA", "MOTA", "IDF1"]
    val_tools_path = Path("trackeval")
    split_dataset = False

    show = True
    show_labels = True
    show_conf = True
    show_trajectories = False

    save_txt = False
    save_crop = False
    save = True
    line_width = None
    per_class = False

# Thêm benchmark và split cho tương thích CLI
source_path = Path(Args.source)
Args.benchmark, Args.split = source_path.parent.name, source_path.name

# Gọi hàm tracking
run_track(Args)