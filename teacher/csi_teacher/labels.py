"""프레임 상태 정책 + /labels·사이드카(.labels.npz) 입출력."""
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .body18 import coco17_to_body18

STATUS_OK, STATUS_NO_PERSON, STATUS_MULTI = 0, 1, 2


@dataclass
class LabelResult:
    pose18: np.ndarray          # (F,18,3) f32 — 결측 NaN
    status: np.ndarray          # (F,) u8
    det_score: np.ndarray       # (F,) f32 — no_person/multi NaN
    W: int
    H: int
    attrs: dict = field(default_factory=dict)


def label_frame(runner, frame, det_thr):
    """단일 프레임 정책: det_thr 이상 박스 0=no_person, 1=ok(pose), ≥2=multi(폐기)."""
    dets = np.asarray(runner.detect(frame), np.float32).reshape(-1, 5)
    dets = dets[dets[:, 4] >= det_thr]
    if len(dets) == 0:
        return STATUS_NO_PERSON, None, np.nan
    if len(dets) > 1:
        return STATUS_MULTI, None, np.nan
    kpts17 = runner.pose(frame, dets[0, :4])
    return STATUS_OK, coco17_to_body18(kpts17), float(dets[0, 4])


def run_label(frames, runner, *, det_thr, progress=None):
    """프레임 이터러블 → LabelResult. (W,H)는 첫 프레임에서."""
    pose, stat, score = [], [], []
    W = H = None
    for i, frame in enumerate(frames):
        # 고정 카메라 가정 — W/H는 첫 프레임에서 래치 (가변 해상도 미지원)
        if W is None:
            H, W = frame.shape[:2]
        s, p18, ds = label_frame(runner, frame, det_thr)
        stat.append(s)
        score.append(ds)
        pose.append(p18 if p18 is not None else np.full((18, 3), np.nan, np.float32))
        if progress and (i + 1) % 200 == 0:
            progress(i + 1)
    if W is None:
        raise SystemExit("프레임 0장 — mp4가 비었거나 디코드 실패")
    return LabelResult(np.stack(pose), np.asarray(stat, np.uint8),
                       np.asarray(score, np.float32), W, H)


def base_attrs(*, mp4, det_thr, det_model, pose_model):
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        rev = ""
    return {"mp4": str(Path(mp4).resolve()), "det_thr": float(det_thr), "det_model": det_model,
            "pose_model": pose_model, "git_rev": rev or "unknown"}


def save_npz(path, res):
    np.savez_compressed(path, pose18=res.pose18, status=res.status,
                        det_score=res.det_score, W=res.W, H=res.H,
                        attrs=json.dumps(res.attrs, ensure_ascii=False))


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return LabelResult(z["pose18"], z["status"], z["det_score"],
                       int(z["W"]), int(z["H"]), json.loads(str(z["attrs"])))


def check_video_mapping(h, F, *, warn):
    """mp4 프레임 수 F vs /video — frame_idx 있으면 유실 허용, 없으면 일치 강제."""
    if "video/t_ns" not in h:
        raise SystemExit("/video/t_ns 없음 — 세션 HDF5가 아님 (mp4 단독은 --h5 생략)")
    V = h["video/t_ns"].shape[0]
    if "video/frame_idx" in h:
        fi = h["video/frame_idx"][...]
        if len(fi) != V:
            raise SystemExit(f"/video 손상: t_ns {V} ≠ frame_idx {len(fi)}")
        if V and int(fi.max()) >= F:
            raise SystemExit(f"frame_idx 최대 {int(fi.max())} ≥ mp4 프레임 {F} — 짝이 다른 세션?")
        if V > F:
            raise SystemExit(f"/video {V} > mp4 {F} — 짝이 다른 세션?")
        if V < F:
            warn(f"cam/meta 유실 의심: /video {V} < mp4 {F} — frame_idx 매핑으로 진행")
    elif V != F:
        raise SystemExit(f"/video/t_ns {V} ≠ mp4 프레임 {F} — frame_idx 없는 구세션은 일치 필수")


def write_h5(h5_path, res, *, force=False, warn=print):
    with h5py.File(h5_path, "r+") as h:
        check_video_mapping(h, len(res.status), warn=warn)
        if "labels" in h and not force:
            raise SystemExit("기존 /labels 존재 — --force로 재라벨")
        if "labels_tmp" in h:
            del h["labels_tmp"]                  # 이전 실패 잔재 청소
        g = h.create_group("labels_tmp")
        g.create_dataset("pose18", data=res.pose18.astype(np.float32))
        g.create_dataset("status", data=res.status.astype(np.uint8))
        g.create_dataset("det_score", data=res.det_score.astype(np.float32))
        g.attrs["W"] = int(res.W)
        g.attrs["H"] = int(res.H)
        g.attrs["F"] = int(len(res.status))
        for k, v in res.attrs.items():
            g.attrs[k] = v
        if "labels" in h:                        # 전부 성공한 뒤에만 교체
            del h["labels"]
        h.move("labels_tmp", "labels")
