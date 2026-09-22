#!/usr/bin/env python3
"""train/ CLI — fit | eval | baselines.

예) python3 train/train.py fit --config configs/train.yaml --loss-mode pam_full
    python3 train/train.py baselines --config configs/train.yaml
    python3 train/train.py eval --ckpt runs/run0/best.pt --config configs/train.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csi_train import baselines as bl
from csi_train import data, fit
from csi_train.loss import MODES
from csi_train.pck import calibrate_kappa, evaluate


def _device(args):
    import torch
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_fit(args):
    if args.vector_head and args.loss_mode != "diag_only":
        raise SystemExit("--vector-head는 --loss-mode diag_only 전용 (§8.3-3안) — "
                         "loss.py도 막지만 수 GB 적재 후라 조기 차단")
    feats = tuple(f for f, on in (("phase", args.phase), ("rssi", args.rssi)) if on)
    man = data.load_manifest(args.config)
    if args.epochs:
        man["hyper"]["epochs"] = args.epochs
    splits = data.build_splits(man, features=feats)
    name = args.name or (args.loss_mode + ("-vec" if args.vector_head else "")
                         + ("-rssi" if args.rssi else "") + ("-phase" if args.phase else ""))
    res = fit.train_model(splits, man["hyper"], mode=args.loss_mode,
                          vector_head=args.vector_head, augment=args.augment,
                          name=name, out_root=args.out_root,
                          device=_device(args), use_compile=args.compile)
    print(json.dumps(res, ensure_ascii=False))


def cmd_eval(args):
    if not Path(args.ckpt).exists():
        raise SystemExit(f"체크포인트 없음: {args.ckpt}")
    device = _device(args)
    model, ck = fit.load_ckpt(args.ckpt, device=device)
    feats = tuple(ck["config"].get("features", []))        # 구형 ckpt = amp 전용 해석
    man = data.load_manifest(args.config)
    rows = data.load_role(man, args.split, feats)
    mu_p = ck["mu_phase"].cpu().numpy() if "mu_phase" in ck else None
    sg_p = ck["sigma_phase"].cpu().numpy() if "sigma_phase" in ck else None
    X = data.normalize_rows(rows, feats, ck["mu"].cpu().numpy(),
                            ck["sigma"].cpu().numpy(), mu_p, sg_p)
    pred_xy, _ = fit.predict(model, X, device=device)
    m = rows.presence
    gt_xy, gt_c = data.diag_pose(rows.Y[m])
    rep = evaluate(pred_xy[m], gt_xy, gt_c, rows.WH[m], kappa=float(ck["kappa"]),
                   stype=rows.stype[m])
    out = Path(args.ckpt).parent / f"report-{args.split}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"저장: {out}")


def cmd_baselines(args):
    device = _device(args)
    man = data.load_manifest(args.config)
    sp = data.build_splits(man)
    tr, va = sp["train"], sp["val"]
    if not tr.presence.any() or not va.presence.any():
        raise SystemExit(f"presence=1 행 없음 (train {int(tr.presence.sum())}, "
                         f"val {int(va.presence.sum())}) — 베이스라인 평가 불가")
    Xtr = tr.X[tr.presence]
    gt_tr, c_tr = data.diag_pose(tr.Y[tr.presence])
    mq = va.presence
    Xq = va.X[mq]
    gt_q, c_q = data.diag_pose(va.Y[mq])
    WHq, stq = va.WH[mq], va.stype[mq]
    try:
        kappa = calibrate_kappa(gt_tr, c_tr, tr.WH[tr.presence])
    except ValueError:
        kappa = 0.5
    k = man["hyper"]["knn_k"]
    preds = {"mean_pose": bl.predict_mean(bl.mean_pose(gt_tr), len(Xq)),
             "knn_centroid": bl.predict_knn_centroid(Xtr, gt_tr, Xq, k=k, device=device),
             "knn_pose": bl.predict_knn_pose(Xtr, gt_tr, Xq, device=device),
             "oracle_centroid": bl.predict_oracle_centroid(gt_tr, gt_q)}
    rep = {n: evaluate(p, gt_q, c_q, WHq, kappa=kappa, stype=stq)
           for n, p in preds.items()}
    rep["gate_baseline_pck02"] = max(rep[n]["pck"]["0.2"] or 0.0
                                     for n in ("mean_pose", "knn_centroid", "knn_pose"))
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({n: rep[n]["pck"] for n in preds} |
                     {"gate_baseline_pck02": rep["gate_baseline_pck02"],
                      "주의": "oracle_centroid는 진단 전용 — 게이트 제외(설계 §9)"},
                     ensure_ascii=False, indent=1))
    print(f"저장: {out / 'baselines.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="WiSPPN-ESP 학습/평가 (M2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--config", default="configs/train.yaml")
    f.add_argument("--loss-mode", choices=MODES, default="pam_full")
    f.add_argument("--vector-head", action="store_true")
    f.add_argument("--rssi", action="store_true", help="RSSI 재스케일(§6.2 정석) — M2.5")
    f.add_argument("--phase", action="store_true", help="sanitized phase 결합 560ch — M2.5")
    f.add_argument("--augment", action="store_true",
                   help="GPU 텐서 증강 4종 (docs/research-20260612-low-data-techniques.md)")
    f.add_argument("--name", default=None)
    f.add_argument("--out-root", default="runs")
    f.add_argument("--epochs", type=int, default=None, help="hyper.epochs 오버라이드")
    f.add_argument("--compile", action="store_true")
    f.add_argument("--device", default=None)
    f.set_defaults(fn=cmd_fit)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--config", default="configs/train.yaml")
    e.add_argument("--split", default="val")
    e.add_argument("--device", default=None)
    e.set_defaults(fn=cmd_eval)
    b = sub.add_parser("baselines")
    b.add_argument("--config", default="configs/train.yaml")
    b.add_argument("--out-root", default="runs")
    b.add_argument("--device", default=None)
    b.set_defaults(fn=cmd_baselines)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
