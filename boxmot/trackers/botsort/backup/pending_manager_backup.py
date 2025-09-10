# pending_manager.py

import numpy as np
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.utils.ops import xyxy2xywh, xywh2xyxy
from boxmot.trackers.botsort.botsort_track import crop_image
from boxmot.utils.matching import (
    iou_distance,
    linear_assignment,
)

from boxmot.trackers.botsort.botsort_utils import compute_frame_metrics

def iou(bbox1, bbox2):
    """Tính IoU giữa 2 bounding boxes (xyxy)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0
    
class PendingTrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()
    
    def __init__(self, det, frame_id, feat=None, min_lost_matches_to_promote=5, promotion_deadline=15):
        """
        det: [x1, y1, x2, y2, conf, cls, det_ind]
        """
        self.xywh    = xyxy2xywh(det[:4])
        self.conf    = det[4]
        self.cls     = det[5]
        self.det_ind = det[6]
        
        self.kalman_filter         = None
        self.mean, self.covariance = None, None
        
        self.state        = TrackState.Pending
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.is_activated = False
        
        self.min_lost_matches_to_promote = min_lost_matches_to_promote
        self.promotion_deadline = promotion_deadline
        self.match_count_lost   = 0
        self.curr_feat = None
        self.smooth_feat = None
        if feat is not None:
            feat = np.asarray(feat, dtype=np.float32, copy=False).ravel()
            nrm  = np.linalg.norm(feat)
            feat = feat / np.clip(nrm, 1e-6, None)
            self.curr_feat = feat
            self.smooth_feat = feat.copy()
                
        self.id = -1  # chưa cấp ID thật
        
        self.first_image  = None
        self.latest_image = None
        
        self.feature_selector = PendingFeatureSelector()
        self.best_feat  = None
        self.best_img   = None
        self.best_score = -1
        
        if feat is not None:
            # Add ngay frame đầu tiên vào selector
            self.feature_selector.add_frame(
                feat,
                area=(det[2] - det[0]) * (det[3] - det[1]),
                occlusion=0.0,
                sharpness=1.0,
                image=None
            )
            self.best_feat, self.best_img, self.best_score = self.feature_selector.get_best()
        
    def activate(self, kalman_filter):
        self.kalman_filter         = kalman_filter
        self.mean, self.covariance = kalman_filter.initiate(self.xywh)
        self.is_activated          = True
        
    def predict(self):
        if self.mean is None:
            return
        
        mean_state = self.mean.copy()
        mean_state[6:8] = 0  # reset velocity
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    def update(self, det, frame_id, feat=None, img=None, other_tracks=None):
        """Update pending track với detection mới."""
        self.frame_id     = frame_id
        self.xywh         = xyxy2xywh(det[:4])
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.xywh)
        self.conf         = det[4]
        
        if feat is not None:
            feat = np.asarray(feat, dtype=np.float32, copy=False).ravel()
            nrm  = np.linalg.norm(feat)
            if nrm > 1e-6:
                feat = feat / nrm
                self.curr_feat = feat
                if self.smooth_feat is None:
                    self.smooth_feat = feat
                else:
                    self.smooth_feat = 0.9 * self.smooth_feat + 0.1 * feat
                    self.smooth_feat /= np.clip(np.linalg.norm(self.smooth_feat), 1e-6, None)
                
            # Compute metrics dùng other_tracks nếu có
            if other_tracks is not None:
                valid_tracks = [t for t in other_tracks if not np.allclose(t.xyxy, self.xyxy)]
                area, occlusion, sharpness = compute_frame_metrics(
                    self, self.xyxy, img, valid_tracks
                )
            else:
                area, occlusion, sharpness = (0.0, 0.0, 1.0)    # fallback
            
            # Crop ảnh cho selector
            img_crop = crop_image(img, det[:4]) if img is not None else None
            
            # Add vào selector
            self.feature_selector.add_frame(feat, area, occlusion, sharpness, img_crop)
            
            # Cập nhật best ngay
            self.best_feat, self.best_img, self.best_score = self.feature_selector.get_best()
        
    def can_promote(self, frame_id):
        return ((frame_id - self.start_frame) > self.promotion_deadline) or (self.match_count_lost >= self.min_lost_matches_to_promote)
    
    def is_expired(self, frame_id):
        return (frame_id - self.start_frame) > self.promotion_deadline
    
    @property
    def xyxy(self):
        ret = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
        return xywh2xyxy(ret)
    
    
class PendingManager:
    def __init__(self, kalman_filter, min_lost_matches_to_promote=3, promotion_deadline=30,
                 iou_thresh=0.7, appearance_thresh=0.1, match_thresh=0.8, use_dynamic_weights=True,
                w_motion_base=0.6, w_motion_start_dw=0.6, dw_start_frames=20, dw_step_frames=20,
                dw_step_delta=0.05, reid_dim=None, dw_app_split=0.5):
        self.pending_tracks = []
        self.kalman_filter  = kalman_filter
        self.min_lost_matches_to_promote = min_lost_matches_to_promote
        self.promotion_deadline = promotion_deadline
        self.iou_thresh         = iou_thresh
        self.appearance_thresh  = appearance_thresh
        self.match_thresh       = match_thresh
        
        # DW params
        self.use_dynamic_weights = use_dynamic_weights
        self.w_motion_base       = w_motion_base
        self.w_motion_start_dw   = w_motion_start_dw
        self.dw_start_frames     = dw_start_frames
        self.dw_step_frames      = dw_step_frames
        self.dw_step_delta       = dw_step_delta
        # Bổ sung thêm chiều đặc trưng của Reid
        self.reid_dim = int(reid_dim) if reid_dim is not None else None
        self.dw_app_split = float(dw_app_split)

    def add_pending(self, detections, frame_id, feats=None, img=None):
        """Thêm pending track mới từ detections."""
        for i, det in enumerate(detections):
            feat  = feats[i] if feats is not None else None
            track = PendingTrack(det, frame_id, feat=feat,
                                 min_lost_matches_to_promote=self.min_lost_matches_to_promote,
                                 promotion_deadline=self.promotion_deadline)
            track.activate(self.kalman_filter)
            if img is not None:
                track.latest_image = crop_image(img, det[:4])
            self.pending_tracks.append(track)

    def _first_valid_dim(self, *lists, default=512):
        for L in lists:
            if not L: 
                continue
            for f in L:
                if f is None:
                    continue
                f = np.asarray(f).ravel()
                if f.size > 0:
                    return int(f.size)
        return int(default)
    
        
    # ------------------------------
    # Chuẩn bị batch feature cho pending tracks
    # ------------------------------
    def _stack_and_l2norm_with_mask(self, feature_list, dim):
        """
        Trả (mat, mask_invalid):
        - Không nhét zero âm thầm.
        - Vector None/sai dim/|v|~0 -> invalid (True).
        """
        k = len(feature_list)
        mat  = np.zeros((k, dim), dtype=np.float32)
        mask = np.zeros((k,), dtype=bool)  # True = invalid

        for i, f in enumerate(feature_list):
            if f is None:
                mask[i] = True
                continue
            v = np.asarray(f, dtype=np.float32).ravel()
            if v.shape[0] != dim:
                mask[i] = True
                continue
            n = np.linalg.norm(v)
            if not np.isfinite(n) or n < 1e-6:
                mask[i] = True
                continue
            mat[i, :] = v / n

        return mat, mask
        
    def update_pending(self, detections, feats, lost_tracks, frame_id,
                    with_reid=True, img=None, other_tracks=None):
        """
        Update pending with LostTrack and detections.
        Returns:
            unmatched_dets, unmatched_feats, merged_to_lost
        """
        unmatched_dets  = list(detections)
        unmatched_feats = list(feats) if feats is not None else [None] * len(detections)
        merged_to_lost  = []

        if not self.pending_tracks:
            return unmatched_dets, unmatched_feats, merged_to_lost

        # Predict tất cả pending
        for t in self.pending_tracks:
            t.predict()

        keep_idx   = set()
        remove_idx = set()

        # Xác định dimension của embedding
        feat_dim = self.reid_dim
        if feat_dim is None:
            feat_dim = self._first_valid_dim(
                [p.curr_feat for p in self.pending_tracks],
                unmatched_feats,  # dùng list đã chuẩn hoá
                default=512
            )

        # ------------------------------
        # 1) Pending vs Detections (điều kiện tiên quyết để giữ)
        # ------------------------------
        if unmatched_dets:
            pending_boxes = np.array([p.xyxy for p in self.pending_tracks], dtype=np.float32)
            det_boxes     = np.array([d[:4] for d in unmatched_dets], dtype=np.float32)

            iou_cost = 1.0 - np.maximum(0.0, self._batch_iou(pending_boxes, det_boxes))
            iou_cost = np.clip(iou_cost.astype(np.float32), 0.0, 1.0)

            if with_reid:
                pending_feats, p_mask = self._stack_and_l2norm_with_mask(
                    [p.curr_feat for p in self.pending_tracks], feat_dim
                )
                
                det_feats, d_mask = self._stack_and_l2norm_with_mask(unmatched_feats, feat_dim)

                cos = np.clip(pending_feats @ det_feats.T, -1.0, 1.0)
                emb_cost = 0.5 * (1.0 - cos)

                invalid_pairs = p_mask[:, None] | d_mask[None, :]
                emb_cost[invalid_pairs] = 1.0

                valid_pairs = ~invalid_pairs
                emb_cost[valid_pairs & (emb_cost > self.appearance_thresh)] = 1.0

                iou_gate = iou_cost > float(self.iou_thresh)
                emb_cost[iou_gate] = 1.0
                cost_matrix = np.fmin(iou_cost, emb_cost).astype(np.float32)
                cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)
            else:
                cost_matrix = iou_cost.astype(np.float32)

            matches, _, u_det_idx = linear_assignment(cost_matrix, thresh=self.match_thresh)

            for ipend, idet in matches:
                det  = unmatched_dets[idet]
                feat = unmatched_feats[idet]

                if other_tracks is not None:
                    other_tracks_filtered = [
                        t for t in other_tracks
                        if not (hasattr(t, "xyxy") and np.allclose(t.xyxy, det[:4], atol=1e-3))
                    ]
                else:
                    other_tracks_filtered = None

                self.pending_tracks[ipend].update(det, frame_id, feat, img=img, other_tracks=other_tracks_filtered)
                keep_idx.add(ipend)

                if img is not None:
                    cropped_img = crop_image(img, det[:4])
                    self.pending_tracks[ipend].latest_image = cropped_img
                    if self.pending_tracks[ipend].first_image is None:
                        self.pending_tracks[ipend].first_image = cropped_img.copy()

            # Giữ lại cặp (det, feat) CHƯA ghép để add_pending với đúng feature
            unmatched_dets  = [unmatched_dets[i]  for i in u_det_idx]
            unmatched_feats = [unmatched_feats[i] for i in u_det_idx]

        # ------------------------------
        # 2) Pending vs LostTracks (chỉ để cộng dồn/merge; không làm 'keep alive')
        # ------------------------------
        # (giữ nguyên toàn bộ logic Người đã viết; không thêm vào keep_idx)
        print(f"[DBG][frame={frame_id}] lost_pool={[t.id for t in lost_tracks]}", flush=True)

        if lost_tracks:
            pending_boxes = np.array([p.xyxy for p in self.pending_tracks], dtype=np.float32)
            lost_boxes    = np.array([t.xyxy for t in lost_tracks], dtype=np.float32)

            print(f"[DBG][frame={frame_id}] PM: shapes pend={pending_boxes.shape}, lost={lost_boxes.shape}", flush=True)

            iou_cost = 1.0 - np.clip(self._batch_iou(pending_boxes, lost_boxes), 0.0, 1.0)

            lost_ids = [t.id for t in lost_tracks]
            for j, lid in enumerate(lost_ids):
                col = iou_cost[:, j]
                print(f"[DBG][frame={frame_id}] PM: IoU summary for lost_id={lid}: min={col.min():.3f} max={col.max():.3f}", flush=True)

            if with_reid:
                # Pending dùng curr/smooth (ưu tiên smooth nếu có), còn Lost:
                #   - SHORT: smooth_feat/curr_feat
                #   - LONG : long_feat_mean (fallback smooth/curr)
                # Chuẩn hoá & mask invalid
                pend_feat_list = [ (p.smooth_feat if p.smooth_feat is not None else p.curr_feat)
                                for p in self.pending_tracks ]
                P, p_mask = self._stack_and_l2norm_with_mask(pend_feat_list, feat_dim)

                lost_short_list = []
                lost_long_list  = []
                for t in lost_tracks:
                    # SHORT side
                    fs = getattr(t, "smooth_feat", None)
                    if fs is None: fs = getattr(t, "curr_feat", None)
                    lost_short_list.append(fs)

                    # LONG side
                    fl = getattr(t, "long_feat_mean", None)
                    if fl is None:
                        fl = getattr(t, "smooth_feat", None)
                        if fl is None: fl = getattr(t, "curr_feat", None)
                    lost_long_list.append(fl)

                Ls, ls_mask = self._stack_and_l2norm_with_mask(lost_short_list, feat_dim)
                Ll, ll_mask = self._stack_and_l2norm_with_mask(lost_long_list , feat_dim)

                # SHORT reid cost
                cos_s = np.clip(P @ Ls.T, -1.0, 1.0)
                cost_s = 0.5 * (1.0 - cos_s)
                inv_s  = p_mask[:, None] | ls_mask[None, :]
                cost_s[inv_s] = 1.0
                valid_s = ~inv_s
                cost_s[valid_s & (cost_s > self.appearance_thresh)] = 1.0

                # LONG reid cost
                cos_l = np.clip(P @ Ll.T, -1.0, 1.0)
                cost_l = 0.5 * (1.0 - cos_l)
                inv_l  = p_mask[:, None] | ll_mask[None, :]
                cost_l[inv_l] = 1.0
                valid_l = ~inv_l
                cost_l[valid_l & (cost_l > self.appearance_thresh)] = 1.0

            else:
                # Không dùng reid
                cost_s = np.ones_like(iou_cost, dtype=np.float32)
                cost_l = np.ones_like(iou_cost, dtype=np.float32)

            # Chống NaN
            cost_s = np.nan_to_num(cost_s, nan=1.0, posinf=1.0, neginf=1.0)
            cost_l = np.nan_to_num(cost_l, nan=1.0, posinf=1.0, neginf=1.0)

            # Dynamic weights: chia appearance thành short/long
            # Base weight cho motion (giống BOTSort)
            use_dw = bool(self.use_dynamic_weights)
            base_motion = self.w_motion_start_dw if use_dw else self.w_motion_base

            M, N = iou_cost.shape
            Wm = np.full((M, N), base_motion, dtype=np.float32)

            if use_dw:
                for j, lost in enumerate(lost_tracks):
                    if lost.state == TrackState.Lost:
                        end_f = getattr(lost, "end_frame", getattr(lost, "frame_id", None))
                        lost_frames = max(0, frame_id - end_f) if end_f is not None else 0
                        if lost_frames >= int(self.dw_start_frames):
                            steps = 1 + (lost_frames - int(self.dw_start_frames)) // max(1, int(self.dw_step_frames))
                            val = max(0.0, base_motion - steps * float(self.dw_step_delta))
                            Wm[:, j] = val

            Wa = 1.0 - Wm
            Wa_long  = Wa

            cost_matrix = (Wm * iou_cost + Wa_long * cost_l).astype(np.float32)
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)

            # === DEBUG LOGS ===
            for j, lid in enumerate(lost_ids):
                col_s = cost_s[:, j]; col_l = cost_l[:, j]
                print(f"[DBG][frame={frame_id}] PM: lost_id={lid} "
                      f"Wm={Wm[0,j]:.2f} WaL={Wa_long[0,j]:.2f} "
                      f"L(min={col_l.min():.3f},max={col_l.max():.3f})",
                      flush=True)

            matches, _, _ = linear_assignment(cost_matrix, thresh=self.match_thresh)
            for ipend, ilost in matches:
                pending = self.pending_tracks[ipend]
                lost    = lost_tracks[ilost]
                pending.match_count_lost += 1
                if pending.match_count_lost >= pending.min_lost_matches_to_promote:
                    merged_to_lost.append((pending, lost))
                    remove_idx.add(ipend)

        # ------------------
        # 3) Cleanup pending
        # ------------------
        for i in remove_idx:
            self.pending_tracks[i].feature_selector.reset()

        # Chỉ giữ pending đã match với detection
        self.pending_tracks = [
            t for i, t in enumerate(self.pending_tracks)
            if (i in keep_idx) and (i not in remove_idx)
        ]

        return unmatched_dets, unmatched_feats, merged_to_lost

    def promote_pending(self, frame_id):
        promotable = [t for t in self.pending_tracks if t.can_promote(frame_id)]
        for track in promotable:
            track.feature_selector.reset()
        self.pending_tracks = [t for t in self.pending_tracks if t not in promotable]
        return promotable

    def cleanup_expired(self, frame_id):
        expired_tracks = [t for t in self.pending_tracks if t.is_expired(frame_id)]
        for track in expired_tracks:
            track.feature_selector.reset()
        self.pending_tracks = [t for t in self.pending_tracks if t not in expired_tracks]

    def _batch_iou(self, boxes1, boxes2):
        boxes1 = boxes1.astype(np.float32, copy=False)
        boxes2 = boxes2.astype(np.float32, copy=False)
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        w  = x2 - x1; np.clip(w, 0, None, out=w)
        h  = y2 - y1; np.clip(h, 0, None, out=h)
        inter = w * h
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.clip(union, 1e-6, None)

class PendingFeatureSelector:
    def __init__(self, score_weights=None, hard_thresholds=None):
        """
        Bộ chọn feature cho PendingTrack.
        - score_weights: trọng số cho tính điểm (area, occlusion, sharpness)
        - hard_thresholds: ngưỡng loại bỏ (vd: max_occlusion)
        """
        self.buffer = []
        self.frame_count = 0
        self.score_weights = score_weights or {'alpha': 0.5, 'beta': 0.5, 'gamma': 0.0}
        self.hard_thresholds = hard_thresholds or {'max_occlusion': 0.6} 
        
    def compute_quality_score(self, area, occlusion, sharpness):
        """Tính điểm chất lượng cho frame."""
        w = self.score_weights
        return w['alpha'] * area - w['beta'] * occlusion + w['gamma'] * sharpness
    
    def add_frame(self, feature, area, occlusion, sharpness, image=None):
        self.frame_count += 1
        if occlusion > self.hard_thresholds['max_occlusion']:
            return
        
        score = self.compute_quality_score(area, occlusion, sharpness)
        self.buffer.append((feature, image, score))
        
    def get_best(self):
        """Lấy feature có điểm cao nhất và ảnh tương ứng."""
        if not self.buffer:
            return None, None, -1
        best_feature, best_image, best_score = max(self.buffer, key=lambda x: x[2])
        return best_feature, best_image, best_score    

    def reset(self):
        """Xóa buffer sau khi promote hoặc pending hết hạn."""
        self.buffer.clear()
        self.frame_count = 0     