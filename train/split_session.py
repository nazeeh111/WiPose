#!/usr/bin/env python3
"""세션 h5 시간순 물리 분할 — 단일 세션 train/val.

samples/·labels/의 행 정렬 데이터셋을 t_ns 순서 그대로 앞 frac(train)/뒤
1−frac(val)로 슬라이스해 두 파일을 만든다. 경계 갭(gap-s — 윈도 스팬 ~2.8s
중첩 누수 차단) 구간 행은 train 쪽에서 제외한다(val 보존 우선). grid/links/
video 등 비학습 그룹은 복사하지 않는다(csi_train.data.load_session 비사용).

예) python3 train/split_session.py \\
        --h5 host/sessions/SESSION-YYYYMMDD-HHMMSS.h5 \\
        --out-dir ~/data
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROW_GROUPS = ("samples", "labels")     # 행 정렬 그룹만 복사 — 나머지는 비학습


def split_indices(t_ns, frac, gap_ns):
    """t_ns(단조) → (train bool, val bool, cut_t). 갭 행은 양쪽 모두 False."""
    t = t_ns.astype(np.int64)
    if np.any(np.diff(t) < 0):
        raise SystemExit("samples/t_ns 비단조 — 레코더 규약 위반, 분할 불가")
    idx_cut = int(len(t) * frac)
    if idx_cut <= 0 or idx_cut >= len(t):
        raise SystemExit(f"분할 경계 무효(idx_cut={idx_cut}, N={len(t)}) — frac 확인")
    cut_t = int(t[idx_cut])
    val = np.zeros(len(t), bool)
    val[idx_cut:] = True
    train = (~val) & (t < cut_t - int(gap_ns))
    if not train.any():
        raise SystemExit("train 0행(갭 과대 또는 세션 과소)")
    return train, val, cut_t


def write_split(src, dst, rows, *, role, frac, gap_s, cut_t_ns):
    """src의 행 정렬 그룹(ROW_GROUPS)을 rows 마스크로 슬라이스해 dst에 기록."""
    with h5py.File(src, "r") as hi, h5py.File(dst, "w") as ho:
        for k, v in hi.attrs.items():
            ho.attrs[k] = v
        n = int(rows.sum())
        for gname in ROW_GROUPS:
            if gname not in hi:
                continue
            gi, go = hi[gname], ho.create_group(gname)
            for k, v in gi.attrs.items():
                go.attrs[k] = v
            if "F" in go.attrs:
                go.attrs["F"] = n              # 행수 attr는 슬라이스 후 값으로 정정
            for name, ds in gi.items():
                if not isinstance(ds, h5py.Dataset) or ds.shape[0] != len(rows):
                    raise SystemExit(f"{gname}/{name}: 행 정렬 아님 — 분할 불가")
                go.create_dataset(name, data=ds[...][rows])
        ho.attrs["split_source"] = str(src)
        ho.attrs["split_role"] = role
        ho.attrs["split_frac"] = frac
        ho.attrs["split_gap_s"] = gap_s
        ho.attrs["split_cut_t_ns"] = cut_t_ns
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--gap-s", type=float, default=3.0)
    a = ap.parse_args(argv)
    src = Path(a.h5)
    if not src.exists():
        raise SystemExit(f"입력 h5 없음: {src}")
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as h:
        t_ns = h["samples/t_ns"][...]
    train, val, cut_t = split_indices(t_ns, a.frac, int(a.gap_s * 1e9))
    total = 0
    for role, rows in (("train", train), ("val", val)):
        dst = out_dir / f"{src.stem}-{role}.h5"
        n = write_split(src, dst, rows, role=role, frac=a.frac,
                        gap_s=a.gap_s, cut_t_ns=cut_t)
        total += n
        print(f"{role}: {dst}  rows={n}")
    print(f"갭 제외 {len(t_ns) - total}행 / 전체 {len(t_ns)}행 (cut_t_ns={cut_t})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
