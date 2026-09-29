"""ByteTrack-style association: high-score match, then low-score recovery."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.track_state import Detection, TrackState, Trajectory


def _center(bbox: np.ndarray) -> tuple[float, float]:
    return float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)


def _iou_matrix(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
    if len(tracks_xyxy) == 0 or len(dets_xyxy) == 0:
        return np.zeros((len(tracks_xyxy), len(dets_xyxy)), dtype=np.float32)
    tl = np.maximum(tracks_xyxy[:, None, :2], dets_xyxy[None, :, :2])
    br = np.minimum(tracks_xyxy[:, None, 2:], dets_xyxy[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_t = np.clip(tracks_xyxy[:, 2] - tracks_xyxy[:, 0], 0, None) * np.clip(
        tracks_xyxy[:, 3] - tracks_xyxy[:, 1], 0, None
    )
    area_d = np.clip(dets_xyxy[:, 2] - dets_xyxy[:, 0], 0, None) * np.clip(
        dets_xyxy[:, 3] - dets_xyxy[:, 1], 0, None
    )
    union = area_t[:, None] + area_d[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def _associate(tracks: list[TrackState], dets: list[Detection], iou_thr: float, predicted: bool) -> list[tuple[int, int]]:
    if not tracks or not dets:
        return []
    track_boxes = np.stack([t.pred_bbox if predicted else t.bbox for t in tracks]).astype(np.float32)
    det_boxes = np.stack([d.bbox for d in dets]).astype(np.float32)
    iou = _iou_matrix(track_boxes, det_boxes)
    cost = 1.0 - iou
    for t_index, track in enumerate(tracks):
        for d_index, det in enumerate(dets):
            if track.class_id != det.class_id:
                cost[t_index, d_index] += 0.55
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for row, col in zip(rows, cols):
        if iou[row, col] >= iou_thr and cost[row, col] < 1.15:
            pairs.append((int(row), int(col)))
    return pairs


class ObjectTracker:
    def __init__(self, lost_frames_timeout: int = 20) -> None:
        self.lost_frames_timeout = lost_frames_timeout
        self.tracks: list[TrackState] = []
        self._next_id = 1
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_index = 0

    def update(
        self,
        detections: list[Detection],
        high_thr: float,
        lost_frames_timeout: int,
        trail_length: int,
        enabled: bool,
    ) -> list[TrackState]:
        self.lost_frames_timeout = max(1, int(lost_frames_timeout))
        self.frame_index += 1
        if not enabled:
            self.reset()
            self.frame_index = 1
            born = [self._spawn(det, trail_length, tracked=False) for det in detections if det.score >= high_thr]
            self.tracks = []
            return born

        self._predict()
        high = [det for det in detections if det.score >= high_thr]
        low = [det for det in detections if det.score < high_thr]
        pool = list(self.tracks)
        pairs = _associate(pool, high, 0.30, predicted=True)
        matched_tracks = {pair[0] for pair in pairs}
        matched_high = {pair[1] for pair in pairs}
        for track_index, det_index in pairs:
            self._apply(pool[track_index], high[det_index], trail_length)

        left_tracks = [pool[index] for index in range(len(pool)) if index not in matched_tracks]
        pairs_low = _associate(left_tracks, low, 0.22, predicted=True)
        matched_low_tracks = set()
        for track_index, det_index in pairs_low:
            self._apply(left_tracks[track_index], low[det_index], trail_length)
            matched_low_tracks.add(track_index)

        for index, track in enumerate(left_tracks):
            if index in matched_low_tracks:
                continue
            track.time_since_update += 1
            track.age += 1

        for index, det in enumerate(high):
            if index in matched_high:
                continue
            self.tracks.append(self._spawn(det, trail_length, tracked=True))

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.lost_frames_timeout]
        visible = [track for track in self.tracks if track.time_since_update == 0]
        visible.sort(key=lambda track: track.score, reverse=True)
        return visible

    def _predict(self) -> None:
        for track in self.tracks:
            pred = track.bbox.astype(np.float32).copy()
            if track.time_since_update <= 0:
                scale = 1.0
            else:
                scale = 0.5 ** min(int(track.time_since_update), 4)
            pred[0] += track.velocity_x * scale
            pred[2] += track.velocity_x * scale
            pred[1] += track.velocity_y * scale
            pred[3] += track.velocity_y * scale
            track.pred_bbox = pred

    def _spawn(self, det: Detection, trail_length: int, tracked: bool) -> TrackState:
        bbox = det.bbox.astype(np.float32).copy()
        cx, cy = _center(bbox)
        track = TrackState(
            track_id=self._next_id,
            class_id=det.class_id,
            class_name=det.class_name,
            score=float(det.score),
            bbox=bbox,
            mask_roi=det.mask_roi,
            trajectory=Trajectory(trail_length),
            cx=cx,
            cy=cy,
            tracked=tracked,
        )
        track.pred_bbox = bbox.copy()
        track.trajectory.add(cx, cy)
        self._motion(track)
        self._next_id += 1
        return track

    def _apply(self, track: TrackState, det: Detection, trail_length: int) -> None:
        new_bbox = det.bbox.astype(np.float32).copy()
        cx, cy = _center(new_bbox)
        vx = cx - track.cx
        vy = cy - track.cy
        track.velocity_x = 0.55 * track.velocity_x + 0.45 * vx
        track.velocity_y = 0.55 * track.velocity_y + 0.45 * vy
        track.bbox = new_bbox
        track.pred_bbox = new_bbox.copy()
        track.cx = cx
        track.cy = cy
        track.class_id = det.class_id
        track.class_name = det.class_name
        track.score = float(det.score)
        track.mask_roi = det.mask_roi
        track.time_since_update = 0
        track.hits += 1
        track.age += 1
        track.tracked = True
        track.trajectory.resize(trail_length)
        track.trajectory.add(cx, cy)
        self._motion(track)

    @staticmethod
    def _motion(track: TrackState) -> None:
        vx = track.velocity_x
        vy = track.velocity_y
        if (vx * vx + vy * vy) ** 0.5 < 1.8:
            track.motion = "STATIC"
            return
        if abs(vx) >= abs(vy):
            track.motion = "MOVING RIGHT" if vx > 0 else "MOVING LEFT"
            return
        track.motion = "MOVING DOWN" if vy > 0 else "MOVING UP"
