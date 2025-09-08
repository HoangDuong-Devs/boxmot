# botsort.py

from pathlib import Path

import numpy as np
import cv2
import torch

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)

from boxmot.utils.matching import (
    fuse_score,
    iou_distance,
    linear_assignment,
)

from boxmot.trackers.botsort.assign_manager  import AssignIDManager
from boxmot.trackers.botsort.pending_manager import PendingManager

class BotSort(BaseTracker):
    """
    BoTSORT Tracker: A tracking algorithm that combines appearance and motion-based tracking.

    Args:
        reid_weights (str): Path to the model weights for ReID.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        half (bool): Use half-precision (fp16) for faster inference.
        per_class (bool, optional): Whether to perform per-class tracking.
        track_high_thresh (float, optional): Detection confidence threshold for first association.
        track_low_thresh (float, optional): Detection confidence threshold for ignoring detections.
        new_track_thresh (float, optional): Threshold for creating a new track.
        track_buffer (int, optional): Frames to keep a track alive after last detection.
        match_thresh (float, optional): Matching cost threshold for data association.
        proximity_thresh (float, optional): IoU distance threshold for first-round association.
        appearance_thresh (float, optional): Appearance embedding distance threshold for ReID.
        cmc_method (str, optional): Method for correcting camera motion, e.g., "sof" (simple optical flow).
        frame_rate (int, optional): Video frame rate, used to scale the track buffer.
        fuse_first_associate (bool, optional): Fuse appearance and motion in the first association step.
        with_reid (bool, optional): Use ReID features for association.
    """

    def __init__(
        self,
        reid_weights        : str   = Path,
        device              : str   = torch.device,
        half                : bool  = False,
        per_class           : bool  = False,
        track_high_thresh   : float = 0.5,
        track_low_thresh    : float = 0.1,
        new_track_thresh    : float = 0.6,
        track_buffer        : int   = 30,
        match_thresh        : float = 0.8,
        proximity_thresh    : float = 0.5,
        appearance_thresh   : float = 0.25,
        cmc_method          : str   = "ecc",
        frame_rate          : int   = 30,
        fuse_first_associate: bool  = False,
        with_reid           : bool  = True,
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks    = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class         = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh  = track_low_thresh
        self.new_track_thresh  = new_track_thresh
        self.match_thresh      = match_thresh

        self.buffer_size   = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh  = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid         = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            ).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate    
        
        self.assign_manager = AssignIDManager(max_age=self.max_time_lost)
        
        self.use_dynamic_weights = True
        self.w_motion_base       = 0.6
        self.w_motion_start_dw   = 0.65 # Khi bật dynamic -> motion cao hơn
        
        self.dw_start_frames = 0      # bắt đầu giảm motion sau X frame Lost
        self.dw_step_frames  = 20
        self.dw_step_delta   = 0.05
        # Vì bản này chưa dùng long-term cost -> sử dụng hết vào short
        self.dw_app_split    = 0.5
        
        self.long_update_cos_thresh = 0.15  # anti-drift threshold theo long
        self.max_obs = 50
        
        self.pending_manager = PendingManager(
            kalman_filter              =self.kalman_filter,
            min_lost_matches_to_promote=5,
            promotion_deadline         =15,
            iou_thresh                 =0.7,
            appearance_thresh          =0.1,
            match_thresh               =0.7,
            use_dynamic_weights        =self.use_dynamic_weights,
            w_motion_base              =self.w_motion_base,
            w_motion_start_dw          =self.w_motion_start_dw,
            dw_start_frames            =self.dw_start_frames,
            dw_step_frames             =self.dw_step_frames,
            dw_step_delta              =self.dw_step_delta,
            reid_dim                   =getattr(self.model, "feature_dim", None) if self.with_reid else None,
            dw_app_split               =self.dw_app_split,
        )
        
        self.use_long_veto = True
        self.occlusion_overlap_thresh = 0.6 # Diện tích bị che -> không cập nhật
    
    def allow_update(self, iou_val, reid_val, long_val, track, other_tracks=None):
        long_th = float(getattr(self, "long_update_cos_thresh", 0.15))
        occ_th  = float(getattr(self, "occlusion_overlap_thresh", 0.6))
        long_ok = (long_val is not None) and (long_val <= long_th)
        occluded = False
        if other_tracks:
            occluded = self._occluded_heavily(track, [t for t in other_tracks if t.id != track.id], thresh=occ_th)
        allow_short = True
        allow_long  = bool(allow_short and long_ok and(not occluded))
        return allow_short, allow_long

    def embedding_distance(self, tracks, detections, metric="cosine"):
        M, N = len(tracks), len(detections)
        cost_matrix = np.zeros((M, N), dtype=np.float32)
        if M == 0 or N == 0:
            return cost_matrix

        det_feats_list = [getattr(d, "curr_feat", None) for d in detections]
        feat_dim = next((len(f) for f in det_feats_list if f is not None), None)
        if feat_dim is None:
            return np.ones((M, N), dtype=np.float32)

        det_feats = [
            (f if f is not None else np.zeros(feat_dim, dtype=np.float32))
            for f in det_feats_list
        ]

        tr_feats = []
        for t in tracks:
            f = getattr(t, "smooth_feat", None)
            if f is None:
                f = getattr(t, "curr_feat", None)
            if f is None or len(f) != feat_dim:
                f = np.zeros(feat_dim, dtype=np.float32)
            tr_feats.append(f)

        tr = np.asarray(tr_feats, dtype=np.float32)
        de = np.asarray(det_feats, dtype=np.float32)

        eps = 1e-6
        tr /= np.clip(np.linalg.norm(tr, axis=1, keepdims=True), eps, None)
        de /= np.clip(np.linalg.norm(de, axis=1, keepdims=True), eps, None)

        # cosine distance (như cdist cosine), ra [0, 2] nếu chưa chia 2
        cost = 1.0 - tr @ de.T
        cost = np.clip(cost, 0.0, 2.0).astype(np.float32)
        return cost
    
    def _build_feat_matrix(self, objs, prefer=("long_feat_mean", "smooth_feat", "curr_feat")):
        feats, valid, ref = [], [], None
        for o in objs:
            v = None
            for name in prefer:
                v = getattr(o, name, None)
                if v is not None:
                    break
            feats.append(v)
            ok = v is not None
            valid.append(ok)
            if ok and ref is None:
                ref = v
        
        valid = np.asarray(valid, dtype=bool)
        if ref is None:
            return None, valid
        
        D = int(ref.shape[0])
        mat = np.zeros((len(feats), D), dtype=np.float32)
        for i, v in enumerate(feats):
            if v is None:
                continue
            x = v.astype(np.float32, copy=False)
            n = np.linalg.norm(x) + 1e-12
            mat[i] = x / n
        return mat, valid
    
    def _compute_long_cost(self, tracks, dets):
        """Cosine-based long-term cost in [0...1]; 1.0 nếu thiếu feature."""
        M, N = len(tracks), len(dets)
        if M == 0 or N == 0:
            return np.ones((M, N), dtype=np.float32)
        
        T, tvalid = self._build_feat_matrix(tracks, ("long_feat_mean", "smooth_feat"))
        D, dvalid = self._build_feat_matrix(dets  , ("curr_feat", "smooth_feat"))
        
        if T is None or D is None:
            return np.ones((M, N), dtype=np.float32)
        
        sim = T @ D.T
        np.clip(sim, -1.0, 1.0, out=sim)
        
        cost = 0.5 * (1.0 - sim)   # [0..1]
        mask = tvalid[:, None] & dvalid[None, :]
        cost = np.where(mask, cost, 1.0).astype(np.float32, copy=False)
        
        return cost
    
    def _occluded_heavily(self, track, other_tracks, thresh=None):
        if thresh is None:
            thresh = getattr(self, "occlusion_overlap_thresh", 0.5)
        ratio = self._max_overlap_ratio(track, other_tracks)
        return ratio >= float(thresh)
    
    def _update_assign_manager_attributes(self):
        tuple_suspect_swapped_id = []
        assign_id_request        = []
        ids                      = []
        coexist_dict             = {}

        # Build active_id_map (nếu cần dùng sau này)
        active_id_map = {t.id: t for t in self.active_tracks}

        # --- Case 1: Track đang suspect (match với lost track)
        for t in self.active_tracks:
            ids.append(t.id)
            coexist_dict[t.id] = getattr(t, "coexist_ids", [])

            if getattr(t, "suspect", False):
                assign_id_request.append(t.id)
                if t.id not in self.assign_manager.local_assignment:
                    self.assign_manager.local_assignment[t.id] = (t.id, None, 0)

                # Check match với lost track để cập nhật expected_id
                best_match_id = None
                best_score    = 999
                for lost in self.lost_stracks:
                    if (
                        t.smooth_feat is not None and
                        lost.smooth_feat is not None and
                        np.linalg.norm(t.smooth_feat) > 1e-6 and
                        np.linalg.norm(lost.smooth_feat) > 1e-6
                    ):
                        f1  = t.smooth_feat / np.linalg.norm(t.smooth_feat)
                        f2  = lost.smooth_feat / np.linalg.norm(lost.smooth_feat)
                        sim = 1 - np.dot(f1, f2)
                        if sim < best_score:
                            best_score    = sim
                            best_match_id = lost.id

                cr_id, old_expected_id, count = self.assign_manager.local_assignment.get(t.id, (t.id, None, 0))
                if best_match_id and best_score < 0.15:
                    if cr_id > best_match_id:
                        count = count + 1 if old_expected_id == best_match_id else 1
                        self.assign_manager.local_assignment[t.id] = (cr_id, best_match_id, count)
                        # print(f"[AssignUpdate] Track {t.id} → expected_id={best_match_id} | sim={best_score:.4f} | count={count}")
                    else:
                        self.assign_manager.local_assignment[t.id] = (cr_id, None, count)

        # # Update attributes vào AssignManager
        # self.assign_manager.update_attributes(
        #     tuple_suspect_swapped_id=tuple_suspect_swapped_id,
        #     assign_id_request       =assign_id_request,
        #     coexistence_ids         =coexist_dict,
        #     ids                     =ids
        # )

        # print(f"[Check] assign_id_request        = {self.assign_manager.assign_id_request}")
        # print(f"[Check] local_assignment         = {self.assign_manager.local_assignment}")
        # print(f"[Check] tuple_suspect_swapped_id = {self.assign_manager.tuple_suspect_swapped_id}")

    def _handle_assign_id_approval(self):
        # Chạy các case
        self.assign_manager.approve_suspect_case_1(self.active_tracks)
        # self.assign_manager.approve_suspect_case_3(self.active_tracks)
        
        # Gán lại ID nếu được approve
        self.assign_manager.assign_id(self.active_tracks)
        self.assign_manager.assign_to_root(self.active_tracks)
        
        # Cập nhật local_assignment và xóa entry không còn hợp lệ
        self.assign_manager.update_trk_attributes(self.active_tracks)
        
        # ✅ Refresh active_tracks để tránh duplicate hoặc sai ID mapping
        self.active_tracks = joint_stracks([], self.active_tracks)

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # --- 1. Preprocess detections ---
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)
        else:
            features_high = embs_first if embs_first is not None else []

        # New: Extract appearance features (second tier) chỉ để hiện thị reid_cost
        features_second = None
        if self.with_reid and len(dets_second) > 0 and embs is None:
            features_second = self.model.get_features(dets_second[:, 0:4], img)
        
        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # --- 2. First association ---
        matches_first, u_track_first, u_detection_first = self._first_association(
            dets, dets_first, active_tracks, unconfirmed, img,
            detections, activated_stracks, refind_stracks, strack_pool,
        )

        # --- 3. Second association ---
        matches_second, u_track_second, u_detection_second = self._second_association(
            dets_second, activated_stracks, lost_stracks,
            refind_stracks, u_track_first, strack_pool, img,
            features_second=features_second,
        )
        
        lost_pool = joint_stracks(self.lost_stracks, lost_stracks)
        
        feats_for_pending  = []
        raw_unmatched_dets = []
        
        # Lấy từ nhóm first (đã có feature)
        for idx in u_detection_first:
            det_box = dets_first[idx]
            raw_unmatched_dets.append(det_box)
            
            if self.with_reid and idx < len(features_high):
                f = features_high[idx]
                # L2-norm an toàn
                f = f.astype(np.float32, copy=False)
                f /= (np.linalg.norm(f) + 1e-6)
                
                feats_for_pending.append(f)
            else:
                feats_for_pending.append(None)
        
        # Lấy từ nhóm second (chưa có feature hoặc feature yếu)
        for idx in u_detection_second:
            raw_unmatched_dets.append(dets_second[idx])
            if self.with_reid and features_second is not None and idx < len(features_second):
                f = features_second[idx]            
                f = f.astype(np.float32, copy=False)
                f /= (np.linalg.norm(f) + 1e-6)
                feats_for_pending.append(f)
            else:
                if self.with_reid and embs is None:
                    # cắt roi và rút đặc trưng “tại chỗ”
                    roi = dets_second[idx, 0:4]
                    f = self.model.get_features(np.asarray([roi]), img)[0]
                    f = f.astype(np.float32, copy=False)
                    f /= (np.linalg.norm(f) + 1e-6)
                    feats_for_pending.append(f)
                else:
                    feats_for_pending.append(None)
        
        # --- 5. Update Pending ---
        other_tracks = self.active_tracks + [
            STrack(det, f) for det, f in zip(raw_unmatched_dets, feats_for_pending) if det is not None
        ]
        
        unmatched_dets, unmatched_feats, merged_to_lost = self.pending_manager.update_pending(
            raw_unmatched_dets,
            feats       =feats_for_pending, 
            lost_tracks =lost_pool,
            frame_id    =self.frame_count, 
            with_reid   =self.with_reid,
            img         =img,
            other_tracks=other_tracks
        )
        
        # --- 6. Merge Pending -> LostTrack ---
        for pending, lost in merged_to_lost:
            lost.re_activate(pending, self.frame_count, new_id=False, img=img)
            # Giữ nguyên ảnh và feature gốc của lost
            self.lost_stracks = [t for t in self.lost_stracks if t.id != lost.id]
            refind_stracks.append(lost)
            
        # --- 7. Thêm pending mới nếu còn unmatched detections ---
        if unmatched_dets:
            self.pending_manager.add_pending(unmatched_dets, self.frame_count, feats=unmatched_feats, img=img)
            
        # --- 8. Promote pending thành track chính thức ---
        for p_track in self.pending_manager.promote_pending(self.frame_count):
            feat_for_strack = (
                p_track.smooth_feat if getattr(p_track, "smooth_feat", None) is not None
                else getattr(p_track, "curr_feat", None)
            )

            if feat_for_strack is not None:
                feat_for_strack = feat_for_strack / np.clip(np.linalg.norm(feat_for_strack), 1e-6, None)
            
            new_track = STrack(np.array([*p_track.xyxy, p_track.conf, p_track.cls, -1]), feat=feat_for_strack)
            new_track.activate(self.kalman_filter, self.frame_count, img=img)
            
            activated_stracks.append(new_track)
        
        self.pending_manager.cleanup_expired(self.frame_count)
        
        # # --- 9. Update suspect logic ---            
        # for track in self.active_tracks:
        #     if getattr(track, "suspect", False):
        #         track.update_suspect(infer_major=False)
                
        # # --- 10. Assign ID logic ---
        # self._update_assign_manager_attributes()
        # self._handle_assign_id_approval()
        
        # --- 11. Xóa những track đã lost quá han ---
        self._update_track_states(lost_stracks, removed_stracks)
        
        # --- 12. Merge & return output ---
        outputs, logs, tracks_for_visual = self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )
        
        pending_tracks = self.pending_manager.pending_tracks
                
        return outputs, logs, tracks_for_visual, pending_tracks

    def _split_detections(self, dets, embs):
        dets  = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        
        dets_second = dets[second_mask]
        first_mask  = confs > self.track_high_thresh
        dets_first  = dets[first_mask]
        embs_first  = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(det, f, max_obs=self.max_obs)
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
        else:
            detections = []
            
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        # reset logs
        for track in strack_pool:
            track.iou_cost       = float("nan")
            track.reid_cost      = float("nan")
            track.final_cost     = float("nan")

        # 1) motion predict + camera compensation
        STrack.multi_predict(strack_pool)
        warp = self.cmc.apply(img, dets[:, :4])  # chỉ lấy xyxy
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # 2) IoU distance (0 tốt, 1 xấu)
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)
        iou_cost = np.clip(ious_dists.astype(np.float32), 0.0, 1.0)

        # 3) ReID cost (short-term), clamp theo appearance_thresh & IoU-mask
        if self.with_reid and len(strack_pool) > 0 and len(detections) > 0:
            reid_cost = self.embedding_distance(strack_pool, detections) / 2.0
            reid_cost[reid_cost > self.appearance_thresh] = 1.0
        else:
            reid_cost = np.ones_like(iou_cost, dtype=np.float32)
            
        # --- Long cost ---    
        if self.with_reid and len(strack_pool) > 0 and len(detections) > 0:
            long_cost = self._compute_long_cost(strack_pool, detections)
            long_cost = np.clip(long_cost.astype(np.float32), 0.0, 1.0)
        else:
            long_cost = np.ones_like(iou_cost, dtype=np.float32)
            
        # HARD GATES
        # Gate theo IoU: nếu IoU > prox_thresh 
        reid_cost[ious_dists_mask] = 1.0
        # long_cost không dùng cho cost, nhưng vẫn gate để tránh nhiễu cực đoan
        long_cost[ious_dists_mask] = 1.0

        # if use_long_veto:
        #     # Nếu long xấu hơn ngưỡng appearance
        #     bad_long = long_cost > self.appearance_thresh
        #     reid_cost[bad_long] = 1.0
        #     long_cost[bad_long] = 1.0
        
        # chống NaN
        iou_cost  = np.nan_to_num(iou_cost , nan=1.0, posinf=1.0, neginf=1.0)
        reid_cost = np.nan_to_num(reid_cost, nan=1.0, posinf=1.0, neginf=1.0)
        long_cost = np.nan_to_num(long_cost, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 6) Final cost: motion + short ReID
        C = np.fmin(iou_cost, reid_cost).astype(np.float32)
        C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)

        # 7) Ghép
        matches, u_track, u_detection = linear_assignment(C, thresh=self.match_thresh)

        # 8) Update & log
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det   = detections[idet]

            track.iou_cost       = float(iou_cost[itracked, idet])
            track.reid_cost      = float(reid_cost[itracked, idet]) if self.with_reid else float("nan")
            track.long_reid_cost = float(long_cost[itracked, idet])
            track.final_cost     = float(C[itracked, idet])
                 
            allow_short, allow_long = self.allow_update(
                iou_cost[itracked, idet],
                reid_cost[itracked, idet] if self.with_reid else None,
                long_cost[itracked, idet] if self.with_reid else None,
                track,
                other_tracks=self.active_tracks
            )
                                
            if track.state == TrackState.Tracked:
                track.update(
                    det, self.frame_count, img,
                    allow_feat=allow_short,     # luôn short
                    allow_long=allow_long       # long có điều kiện
                )
                activated_stracks.append(track)
            else:
                track.re_activate(
                    det, self.frame_count, new_id=False, img=img,
                    allow_feat=allow_short,     # luôn short
                    allow_long=allow_long       # long có điều kiện
                )
                refind_stracks.append(track)

        return matches, u_track, u_detection

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
        img,
        features_second=None,
    ):
        # 1) Tạo detections_second (có curr_feat nếu đã extract ổn định)
        if len(dets_second) > 0:
            has_feats = (
                self.with_reid
                and features_second is not None
                and len(features_second) == len(dets_second)
                and len(features_second) > 0
            )
            if has_feats:
                detections_second = [
                    STrack(det, f, max_obs=self.max_obs)
                    for det, f in zip(dets_second, features_second)
                ]
            else:
                detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]
        else:
            detections_second = []

        # 2) Lấy những track còn đang Tracked trong pool chưa match vòng 1
        r_tracked_stracks = [
            strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked
        ]

        # 3) Ghép IoU-only ở vòng 2
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        # 4) Tính chi phí ReID để LOG (không ảnh hưởng ghép)
        reid_cost_mat = None
        if (
            self.with_reid
            and len(r_tracked_stracks) > 0
            and len(detections_second) > 0
            and any(getattr(d, "curr_feat", None) is not None for d in detections_second)
        ):
            # dùng embedding_distance "gốc": chia 2.0 rồi clamp > apperance_thresh -> 1.0
            reid_cost_mat = self.embedding_distance(r_tracked_stracks, detections_second) / 2.0
            reid_cost_mat[reid_cost_mat > self.appearance_thresh] = 1.0
            # chống NaN/inf
            reid_cost_mat = np.nan_to_num(reid_cost_mat, nan=1.0, posinf=1.0, neginf=1.0)

        long_cost_mat = None
        if self.with_reid and len(r_tracked_stracks) > 0 and len(detections_second) > 0:
            long_cost_mat = self._compute_long_cost(r_tracked_stracks, detections_second)
            long_cost_mat = np.nan_to_num(long_cost_mat, nan=1.0, posinf=1.0, neginf=1.0)
            bad_long = long_cost_mat > self.appearance_thresh
            long_cost_mat[bad_long] = 1.0
            if reid_cost_mat is not None:
                reid_cost_mat[bad_long] = 1.0

        # 5) Cập nhật + log (KHÔNG cập nhật features/EMA ở vòng 2)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det   = detections_second[idet]

            # final_cost = IoU (vì vòng 2 ghép theo IoU)
            track.iou_cost   = float(dists[itracked, idet])
            track.final_cost = track.iou_cost

            # log reid_cost (fallback 1.0 nếu chưa tính)
            track.reid_cost = float(reid_cost_mat[itracked, idet]) if reid_cost_mat is not None else 1.0
            # log long reid (fallback NaN để dễ phân biệt)
            track.long_reid_cost = float(long_cost_mat[itracked, idet]) if long_cost_mat is not None else float("nan")
            
            # chỉ cập nhật motion/state; không cập nhật feature/EMA
            track.frame_id = self.frame_count
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count, img=None, allow_feat=False, allow_long=False)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False, img=None, allow_feat=False, allow_long=False)
                refind_stracks.append(track)

        # 6) Những track còn lại mà không match -> mark lost
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection

    def _update_track_states(self, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(
        self, activated_stracks, refind_stracks, lost_stracks, removed_stracks
    ):
        
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        
        self.lost_stracks  = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks
            if t.is_activated
        ]
        
        logs = []
        for t in self.active_tracks:
            if t.is_activated:
                logs.append({
                    "track_id"      : t.id,
                    "conf"          : getattr(t, "conf", float("nan")),    
                    "reid_cost"     : getattr(t, "reid_cost", float("nan")),
                    "long_reid_cost": getattr(t, "long_reid_cost", float("nan")),
                    "iou_cost"      : getattr(t, "iou_cost", float("nan")),
                    "final_cost"    : getattr(t, "final_cost", float("nan")),
                })
        
        tracks_for_visual = []
        for t in (self.active_tracks + self.lost_stracks):
            if t.latest_image is not None:  # Có ảnh cập nhật khi feat update
                tracks_for_visual.append({
                    "id"   : t.id,
                    "image": t.latest_image
                })

        return np.asarray(outputs), logs, tracks_for_visual
                   
    def _max_overlap_ratio(self, track, other_tracks):
        """Tính max (intersect / area của track) giữa track và các track khác."""
        box1 = track.xyxy
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        max_ratio = 0.0

        for other in other_tracks:
            if other.id == track.id:
                continue
            box2 = other.xyxy
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)
            if inter_area > 0:
                ratio = inter_area / area1
                if ratio > max_ratio:
                    max_ratio = ratio
        return max_ratio

    # def _allow_feature_update(self, external_dist, track=None, feat=None, other_tracks=None, safe_thresh=0.15, drift_thresh=0.15, overlap_thresh=0.5) -> bool:
    #     # 1. Check external_dist (match score giữa detection và track)
    #     if external_dist is None or np.isnan(external_dist) or external_dist > safe_thresh:
    #         return False
        
    #     # 2. Check drift nếu có smooth_feat và feature mới
    #     if track is not None and feat is not None and track.smooth_feat is not None:
    #         drift = compute_cosine_distance(track.smooth_feat, feat)
    #         if drift > drift_thresh:
    #             return False
        
    #     # 3. Kiểm tra overlap
    #     if track is not None and other_tracks is not None:
    #         max_overlap = self._max_overlap_ratio(track, other_tracks)
    #         if max_overlap > overlap_thresh:
    #             return False
        
    #     return True