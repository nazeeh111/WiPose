"""§5.2 빌드 오케스트레이션 — 클록핏 → 링크 스트림 → /grid → /samples.

블록 단위 기록으로 장시간 세션 메모리 제한. 앵커 = /video/t_ns (있으면) 또는
합성 등간격(anchor_rate Hz)."""
import json
from pathlib import Path

import h5py
import numpy as np

from .align import (LinkStream, amplitude, cut_windows, fill_gaps, grid_block,
                    grid_bounds, sanitized_phase, split_epochs, window_indices, WIN)
from .clockfit import fit_board

STEP_NS = 10_000_000  # 100Hz
N_FEAT = 113                          # 값 열: amp[:56] ‖ phase[56:112] ‖ rssi[112]


def anchor_shift_ns(csi_ms, cam_ms):
    """§5.2-3 보정 앵커 시프트: 앵커′ = vid − shift, shift = (cam − csi) ms → int ns.

    규약 '참값 = 스탬프 − 보정값'에서 유도 — 장면 사건 T의 그리드 위치(T + csi_corr)에
    윈도 끝을 맞춘다(스펙 2026-06-11-correction-apply). 비유한·|shift|>1s는 fail-loud."""
    if not (np.isfinite(csi_ms) and np.isfinite(cam_ms)):
        raise SystemExit(f"보정값 비유한 — csi={csi_ms} cam={cam_ms}")
    shift_ms = float(cam_ms) - float(csi_ms)
    if abs(shift_ms) > 1000.0:
        raise SystemExit(f"보정 시프트 |{shift_ms:.1f}ms| > 1000ms — 단위(ms) 착오 의심")
    return int(round(shift_ms * 1e6))


def resolve_corrections(csi_ms=None, cam_ms=None, no_correction=False,
                        config_path=None):
    """CLI 보정값 해석 — 우선순위: no_correction > 개별 플래그 > config 파일.

    반환 (corrections | None, source). corrections = {"csi_ms", "cam_ms"}.
    - config 부재 → (None, "absent") 통과 (합성 세션 허용 — 영상 세션은 build가 fail-loud)
    - config 존재하는데 파싱 실패·필수 키 누락 → SystemExit (조용히 '없음' 취급 금지)
    - 병합 후 한쪽 값만 결정 → SystemExit
    - 플래그가 하나라도 개입하면 source = "cli" (값 자체는 meta에 기록되므로 추적 가능)"""
    if no_correction:
        return {"csi_ms": 0.0, "cam_ms": 0.0}, "off"
    cfg_csi = cfg_cam = None
    if config_path is not None and Path(config_path).exists():
        try:
            cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
            cfg_csi = float(cfg["csi_correction_ms"])
            cfg_cam = float(cfg["cam_correction_ms"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"pairing config 손상({config_path}): {e}")
    c = csi_ms if csi_ms is not None else cfg_csi
    m = cam_ms if cam_ms is not None else cfg_cam
    if c is None and m is None:
        return None, "absent"
    if c is None or m is None:
        raise SystemExit("보정값은 둘 다 필요 — csi/cam 중 하나만 결정됨 "
                         f"(csi={c}, cam={m})")
    src = "cli" if (csi_ms is not None or cam_ms is not None) else "config"
    return {"csi_ms": float(c), "cam_ms": float(m)}, src


def _link_stream(d, model, *, max_gap_run=2, label="?"):
    """링크 원시 배열 + 보드 클록 모델 → LinkStream (에포크 머지, 에포크 간 = break).

    값 열 [n,113] = 진폭‖sanitized phase‖rssi — 열 결합으로 보간·그리드 1패스(M2.5 §1.2).
    새니타이즈는 보간 전 원시 패킷에서(선형 성분이 패킷마다 달라 잔차? 음.... 후에만 보간 유의미)."""
    tf, ok = model.predict(d["esp_us"].astype(np.float64), d["boot_id"])
    vals = np.concatenate([amplitude(d["iq"]), sanitized_phase(d["iq"]),
                           np.asarray(d["rssi"], np.float32)[:, None]], axis=1)
    parts_t, parts_a, parts_m, breaks = [], [], [], []
    for sl in split_epochs(d["seq"], d["boot_id"]):
        keep = ok[sl]
        t = tf[sl][keep].astype(np.int64)
        a = vals[sl][keep]
        sq = np.asarray(d["seq"], np.int64)[sl][keep]
        if len(t) < 2:
            continue
        t2, a2, m2, br = fill_gaps(t, sq, a, max_run=max_gap_run)
        if parts_t:
            breaks.append((int(parts_t[-1][-1]), int(t2[0])))
        parts_t.append(t2)
        parts_a.append(a2)
        parts_m.append(m2)
        breaks.extend(br)
    if not parts_t:
        return None
    t_all = np.concatenate(parts_t)
    if (np.diff(t_all) < 0).any():      # 역행 = 쓰레기 보간 — 조용히 진행 금지
        raise SystemExit(f"링크 {label}: 병합 스트림 t 역행 — 에포크 순서/클록핏 점검 필요")
    return LinkStream(t=t_all, amp=np.concatenate(parts_a),
                      interp=np.concatenate(parts_m), breaks=sorted(breaks))


def _fit_rx(ds, window_s):
    """rx의 링크별 핏 필드(t_ns·esp_us·boot_id) 목록 → (모델, stats). 지역 배열은 반환 시 해제."""
    t = np.concatenate([d["t_ns"] for d in ds]).astype(np.int64)
    esp = np.concatenate([d["esp_us"] for d in ds]).astype(np.float64)
    boot = np.concatenate([d["boot_id"] for d in ds])
    o = np.argsort(t, kind="stable")               # 도착순 (에포크 순서 보존)
    model, rep = fit_board(esp[o], t[o], boot[o], window_s=window_s)
    return model, rep.stats()


def _cut_into(out_ds, grid_ds, mask_ds, starts, *, label, say):
    """그리드 → 윈도 절단을 블록 단위로 out_ds에 기록 + 유한성 fail-loud. valid 반환.

    피처별 순차 패스(메모리 피크 = 슬라이스 1개 유지). rssi처럼 [rows,3,3]이면 K=1 축 부여.
    cut_windows는 K=56 전용이므로 K≠56은 직접 슬라이스(마스크 판정 재사용)."""
    valid_all = np.zeros(len(starts), bool)
    span = 100_000                                  # 그리드 행 단위 배치
    n_done = 0
    while n_done < len(starts):
        lo_row = int(starts[n_done])
        hi_idx = int(np.searchsorted(starts, lo_row + span - WIN, side="right"))
        hi_idx = max(hi_idx, n_done + 1)
        rows0 = lo_row
        rows1 = int(starts[hi_idx - 1]) + WIN
        mem = np.asarray(grid_ds[rows0:rows1], np.float32)
        mask_mem = mask_ds[rows0:rows1][...]
        rel = starts[n_done:hi_idx] - rows0
        if mem.ndim == 3:
            # rssi: [rows,3,3] → slices [N,WIN,3,3] without cut_windows reshape assumption
            idx = rel[:, None] + np.arange(WIN)[None, :]        # [N,WIN]
            X = mem[idx].astype(np.float16)                     # [N,WIN,3,3]
            bad = mask_mem[idx].sum(axis=1)                     # [N,3,3]
            valid = (bad < 2).all(axis=(1, 2))
        else:
            # amp/phase: [rows,K,3,3] → cut_windows → [N,WIN*K,3,3]
            X, valid = cut_windows(mem, mask_mem, rel)
        if not np.isfinite(X).all():
            raise SystemExit(f"{label}: 비유한 값 — 빌드 중단(원시/클록핏 점검)")
        out_ds[n_done:hi_idx] = X
        valid_all[n_done:hi_idx] = valid
        n_done = hi_idx
        say(f"{label} {n_done}/{len(starts)}")
    return valid_all


def build(h5_path, *, anchor_rate=20.0, fit_window_s=600.0, force=False,
          block=100_000, max_gap_run=2, corrections=None, progress=None):
    """세션 HDF5에 /grid·/samples 추가. {"G","N","fit"} 반환.

    corrections: {"csi_ms", "cam_ms"[, "source"]} | None — §5.2-3 보정 앵커
    (스펙 2026-06-11-correction-apply). 영상 앵커 세션은 필수(fail-loud),
    합성 앵커는 무시. /samples/t_ns는 항상 원시 영상 스탬프(pam 조인 계약)."""
    say = progress or (lambda s: None)
    h = h5py.File(h5_path, "r+")
    try:
        if ("grid" in h or "samples" in h):
            if not force:
                raise SystemExit("기존 /grid·/samples 존재 — --force로 재빌드")
            for k in ("grid", "samples"):
                if k in h:
                    del h[k]
        names = sorted(h["links"])
        # 클록핏 필드만 선로드 — 대용량 원시(iq)는 링크 단위 지연 로드 (메모리 피크 ↓)
        per_rx = {}
        for name in names:
            g = h[f"links/{name}"]
            per_rx.setdefault(int(name[0]), []).append(
                {k: g[k][...] for k in ("t_ns", "esp_us", "boot_id")})
        # 보드(rx)별 클록핏 — 같은 보드의 3링크는 같은 클록
        models, fit_stats = {}, {}
        for i in sorted(per_rx):
            models[i], fit_stats[i] = _fit_rx(per_rx.pop(i), fit_window_s)
            say(f"rx{i} 클록핏: {fit_stats[i]}")
        # 링크 스트림 — 링크별 로드 → 진폭 변환 → 다음 링크 로드 전 원시 해제
        streams = {}
        for name in names:
            key = (int(name[0]), int(name[1]))
            g = h[f"links/{name}"]
            d = {k: g[k][...] for k in ("esp_us", "boot_id", "iq", "seq", "rssi")}
            s = _link_stream(d, models[key[0]], max_gap_run=max_gap_run, label=key)
            del d
            if s is None:
                raise SystemExit(f"링크 {key}: 유효 표본 부족 — 빌드 불가")
            streams[key] = s
        g0, g1 = grid_bounds(list(streams.values()), step_ns=STEP_NS)
        G = int((g1 - g0) // STEP_NS)
        if G < WIN:
            raise SystemExit("공통 가용 구간이 윈도보다 짧음")
        gg = h.create_group("grid")
        gg.create_dataset("t_ns", data=(g0 + STEP_NS * np.arange(G, dtype=np.int64))
                          .astype(np.uint64))
        amp_ds = gg.create_dataset("amp", shape=(G, 56, 3, 3), dtype=np.float16,
                                   chunks=(min(4096, G), 56, 3, 3))
        phase_ds = gg.create_dataset("phase", shape=(G, 56, 3, 3), dtype=np.float16,
                                     chunks=(min(4096, G), 56, 3, 3))
        rssi_ds = gg.create_dataset("rssi", shape=(G, 3, 3), dtype=np.float16,
                                    chunks=(min(65536, G), 3, 3))
        mask_ds = gg.create_dataset("mask", shape=(G, 3, 3), dtype=bool,
                                    chunks=(min(65536, G), 3, 3))
        for b0 in range(0, G, block):
            b1 = min(b0 + block, G)
            tb = g0 + STEP_NS * np.arange(b0, b1, dtype=np.int64)
            feat_blk = np.zeros((b1 - b0, N_FEAT, 3, 3), np.float32)
            mask_blk = np.zeros((b1 - b0, 3, 3), bool)
            for (i, j), s in streams.items():
                a, m = grid_block(s, tb)
                feat_blk[:, :, i, j] = a
                mask_blk[:, i, j] = m
            amp_ds[b0:b1] = feat_blk[:, :56].astype(np.float16)
            phase_ds[b0:b1] = feat_blk[:, 56:112].astype(np.float16)
            rssi_ds[b0:b1] = feat_blk[:, 112].astype(np.float16)
            mask_ds[b0:b1] = mask_blk
            say(f"그리드 {b1}/{G}")
        # 앵커 (+§5.2-3 보정 시프트 — 윈도 선택만 보정, 저장 t_ns는 원시 유지)
        vid = h["video/t_ns"][...].astype(np.int64)
        if len(vid):
            if corrections is None:
                raise SystemExit(
                    "영상 앵커 세션 — 보정값 필요: configs/pairing.json 또는 "
                    "--csi-corr-ms/--cam-corr-ms (무보정 빌드는 --no-correction)")
            shift_ns = anchor_shift_ns(corrections["csi_ms"], corrections["cam_ms"])
            corr_meta = {"csi_ms": float(corrections["csi_ms"]),
                         "cam_ms": float(corrections["cam_ms"]),
                         "shift_ms": shift_ns / 1e6,
                         "source": corrections.get("source", "cli")}
            anchors = vid
            src = "video"
        else:
            shift_ns = 0
            corr_meta = {"source": "n/a(synthetic)"}
            anchors = np.arange(g0 + WIN * STEP_NS, g1 + 1,
                                round(1e9 / anchor_rate), dtype=np.int64)
            src = f"synthetic@{anchor_rate}Hz"
        anchors = np.sort(anchors)      # 영상 앵커는 기록순 무보장 — 배칭 searchsorted 전제
        starts, ok = window_indices(g0, STEP_NS, G, anchors - shift_ns)
        starts, anchors = starts[ok], anchors[ok]
        if len(starts) == 0:
            raise SystemExit("유효 앵커 없음 — 앵커 시각이 그리드와 겹치지 않음")
        sg = h.create_group("samples")
        N = len(starts)
        X_ds = sg.create_dataset("X", shape=(N, WIN * 56, 3, 3), dtype=np.float16,
                                 chunks=(min(1024, max(1, N)), WIN * 56, 3, 3))
        XP_ds = sg.create_dataset("X_phase", shape=(N, WIN * 56, 3, 3), dtype=np.float16,
                                  chunks=(min(1024, max(1, N)), WIN * 56, 3, 3))
        RS_ds = sg.create_dataset("rssi", shape=(N, WIN, 3, 3), dtype=np.float16,
                                  chunks=(min(4096, max(1, N)), WIN, 3, 3))
        valid_all = _cut_into(X_ds, amp_ds, mask_ds, starts, label="샘플", say=say)
        v_p = _cut_into(XP_ds, phase_ds, mask_ds, starts, label="샘플(위상)", say=say)
        v_r = _cut_into(RS_ds, rssi_ds, mask_ds, starts, label="샘플(rssi)", say=say)
        if not (np.array_equal(valid_all, v_p) and np.array_equal(valid_all, v_r)):
            raise SystemExit("X/X_phase/rssi valid 불일치 — 빌드 내부 오류")
        sg.create_dataset("t_ns", data=anchors.astype(np.uint64))
        sg.create_dataset("valid", data=valid_all)
        h["meta"].attrs["build"] = json.dumps(
            {"anchors": src, "G": G, "N": len(starts),
             "features": ["amp", "phase", "rssi"],
             "corrections": corr_meta,
             "fit": {str(k): v for k, v in fit_stats.items()},
             "fit_window_s": fit_window_s}, ensure_ascii=False)
        return {"G": G, "N": int(len(starts)), "fit": fit_stats,
                "valid_ratio": float(valid_all.mean()) if len(valid_all) else 0.0}
    finally:
        h.close()
