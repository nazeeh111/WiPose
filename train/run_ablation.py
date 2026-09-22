#!/usr/bin/env python3
"""M2 ablation 야간 배치 러너 — baselines → fit×4 → eval×4 → §13 게이트 요약.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # train/ → csi_train
from csi_train.data import load_manifest  # noqa: E402

# 게이트 임계 — train/README "게이트 판정"·M2 스펙 §13
GATE_ABS = 0.35        # val PCK@0.2 절대 하한
GATE_MARGIN = 0.10     # gate_baseline_pck02 대비 요구 마진
GATE_LYING = 0.25      # lying PCK@0.2 하한
KAPPA_WARN_LT = 0.4    # κ < 이 값이면 lying 분모 보정 무효 의심(§6) — 경고만, 불합격 아님

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"

RUNS = [  # §8.3 고정 3안 + 증강 1런 — 목록 순서가 best_run 동률 시 우선순위
    {"name": "pam_full", "extra": ["--loss-mode", "pam_full"]},
    {"name": "diag_balanced", "extra": ["--loss-mode", "diag_balanced"]},
    {"name": "diag_only_vec", "extra": ["--loss-mode", "diag_only", "--vector-head"]},
    # 소표본 증강 ablation (docs/research-20260612-low-data-techniques.md) — 동률 시 무증강 우선
    {"name": "pam_full_aug", "extra": ["--loss-mode", "pam_full", "--augment"]},
]


def judge_run(report, gate_baseline):
    """report-val.json dict(없으면 None) + 기준치 → 런 판정 dict.

    gate 3치 논리: abs/lying 미달이나 error 런은 margin을 몰라도 확정 False,
    margin만 미지(gate_baseline None)면 None(판정 불가), 전부 충족 시 True.

    lying 미측정(pck_lying["0.2"] == null): lying02=None, lying_ok=None.
    abs/margin이 확정 False면 그대로 False; abs·margin 통과 시 gate=None(판정 불가).
    """
    if report is None:
        return {"pck02": None, "lying02": None, "kappa": None,
                "abs_ok": None, "margin_ok": None, "lying_ok": None,
                "gate": False, "kappa_warn": False}
    pck02 = float(report["pck"]["0.2"])
    lying_raw = report["pck_lying"]["0.2"]
    lying02 = None if lying_raw is None else float(lying_raw)
    kappa = float(report["kappa"])
    abs_ok = pck02 >= GATE_ABS
    lying_ok = None if lying02 is None else lying02 >= GATE_LYING
    margin_ok = None if gate_baseline is None else pck02 >= gate_baseline + GATE_MARGIN
    if not abs_ok or lying_ok is False or margin_ok is False:
        gate = False
    elif lying_ok is None or margin_ok is None:
        gate = None
    else:
        gate = True
    return {"pck02": pck02, "lying02": lying02, "kappa": kappa,
            "abs_ok": abs_ok, "margin_ok": margin_ok, "lying_ok": lying_ok,
            "gate": gate, "kappa_warn": kappa < KAPPA_WARN_LT}


def summarize(judged):
    """judge_run 결과 리스트(각 항목에 'name' 키 포함, RUNS 순서) → best_run·m2_gate.

    best = pck02 최고(동률 시 앞 순서 — max는 첫 최대 유지). report 없는 런 제외.
    전부 제외면 best_run=None·m2_gate=False.
    """
    with_report = [j for j in judged if j["pck02"] is not None]
    if not with_report:
        return {"best_run": None, "m2_gate": False}
    best = max(with_report, key=lambda j: j["pck02"])
    return {"best_run": best["name"], "m2_gate": best["gate"]}


def plan_step(out_root, name):
    """멱등 재실행 판정 — 스펙 §4. 부분 best.pt는 완료 증거가 아니다(fit-done 마커 기준)."""
    if _read_report(out_root, name) is not None:   # 파손 report는 미완료 취급 → 재평가로 자가복구
        return "skip"
    if (out_root / "ablation-logs" / f"{name}.fit-done").exists():
        return "eval_only"
    return "full"


def _common(config, device):
    return ["--config", str(config)] + (["--device", device] if device is not None else [])


def build_baselines_cmd(config, out_root, device):
    return [sys.executable, str(TRAIN_SCRIPT), "baselines",
            *_common(config, device), "--out-root", str(out_root)]


def build_fit_cmd(config, out_root, run, epochs, device):
    """--name은 항상 명시 — out_root/<name> 경로 계약을 train.py 자동명에 의존하지 않는다."""
    cmd = [sys.executable, str(TRAIN_SCRIPT), "fit", *_common(config, device),
           "--out-root", str(out_root), *run["extra"]]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    return cmd + ["--name", run["name"]]


def build_eval_cmd(config, out_root, name, device):
    # --split 생략 = 'val' 기본값 고정 (report-val.json 명세, 스펙 §5)
    return [sys.executable, str(TRAIN_SCRIPT), "eval", *_common(config, device),
            "--ckpt", str(out_root / name / "best.pt")]


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def preflight(config, device, out_root):
    """시작 전 전수 검사 — 에러 문자열 리스트(빈 리스트 = 통과). 스펙 §3-1."""
    if not config.exists():
        return [f"설정 파일 없음: {config}"]
    errors = []
    try:
        man = load_manifest(config)
    except Exception as e:                      # yaml 파손·빈 파일 → 깨끗한 에러로
        return [f"설정 파일 파싱 실패: {config} — {type(e).__name__}: {e}"]
    sessions = man.get("sessions") or []
    if not sessions:
        errors.append(f"매니페스트 sessions 비어 있음: {config}")
    for s in sessions:
        h5 = Path(s["h5"])
        if str(h5).startswith("/mnt/"):
            errors.append(f"/mnt/ 경로 학습 금지(9P, 설계 §15) — ext4로 복사할 것: {h5}")
        elif not h5.exists():
            errors.append(f"세션 HDF5 없음: {h5}")
    if device != "cpu" and not _cuda_available():
        errors.append("CUDA 불가 — train.py의 cpu 묵시 폴백으로 야간 런이 무의미해짐. "
                      "CPU로 강행하려면 --device cpu 명시")
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        probe = out_root / ".write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as e:
        errors.append(f"out-root 쓰기 불가: {out_root} ({e})")
    return errors


def _read_report(out_root, name):
    p = out_root / name / "report-val.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):    # 부분 기록(전원 단절 등) → 없음으로 격하
        return None


def _log_header(logf, label, cmd):
    logf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label}: {' '.join(cmd)}\n")
    logf.flush()


def _run_logged(cmd, logf, label):
    _log_header(logf, label, cmd)
    return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode == 0


def run_one(run, config, out_root, epochs, device):
    """한 런의 fit→eval — 멱등 스킵·fit-done 마커·로그. 반환: done|skipped|error."""
    name = run["name"]
    step = plan_step(out_root, name)
    if step == "skip":
        return "skipped"
    logs = out_root / "ablation-logs"
    with open(logs / f"{name}.log", "a", encoding="utf-8") as logf:
        if step == "full":
            if not _run_logged(build_fit_cmd(config, out_root, run, epochs, device),
                               logf, "fit"):
                return "error"
            (logs / f"{name}.fit-done").write_text("")
        if not _run_logged(build_eval_cmd(config, out_root, name, device),
                           logf, "eval"):
            return "error"
    return "done"


def print_plan(config, out_root, args):
    """--dry-run: 실행될 커맨드 시퀀스·스킵 판정만 출력."""
    baselines_path = out_root / "baselines.json"
    if args.force_baselines or (not baselines_path.exists() and not args.skip_baselines):  # main need_baselines와 동일 조건(동기 유지)
        print("baselines:", " ".join(build_baselines_cmd(config, out_root, args.device)))
    else:
        reason = "--skip-baselines" if args.skip_baselines else "기존 재사용"
        print(f"baselines: 스킵({reason})")
    for run in RUNS:
        step = plan_step(out_root, run["name"])
        if step == "skip":
            print(f"{run['name']}: 스킵(report 존재)")
            continue
        if step == "full":
            print(f"{run['name']} fit:",
                  " ".join(build_fit_cmd(config, out_root, run, args.epochs, args.device)))
        print(f"{run['name']} eval:",
              " ".join(build_eval_cmd(config, out_root, run["name"], args.device)))


def print_table(judged, overall, gate_baseline):
    """아침 확인용 콘솔 표 — summary와 동일 내용."""
    def b(v):
        return "—" if v is None else ("✓" if v else "✗")

    print(f"\n{'run':<16}{'status':<9}{'pck@0.2':<9}{'lying':<8}{'κ':<8}"
          f"{'abs':<5}{'margin':<8}{'ly_ok':<7}GATE")
    for j in judged:
        pck = "—" if j["pck02"] is None else f"{j['pck02']:.3f}"
        ly = "—" if j["lying02"] is None else f"{j['lying02']:.3f}"
        ka = "—" if j["kappa"] is None else (
            f"{j['kappa']:.2f}" + ("⚠" if j["kappa_warn"] else ""))
        gate = {True: "PASS", False: "FAIL", None: "N/A"}[j["gate"]]
        print(f"{j['name']:<16}{j['status']:<9}{pck:<9}{ly:<8}{ka:<8}"
              f"{b(j['abs_ok']):<5}{b(j['margin_ok']):<8}{b(j['lying_ok']):<7}{gate}")
    base = "—" if gate_baseline is None else f"{gate_baseline:.3f}"
    # gate None 사유 파생: baselines 있으면 margin_ok는 None일 수 없음 → 남는 원인은 lying 미측정
    reason = "baselines 부재" if gate_baseline is None else "lying 미측정"
    m2 = {True: "PASS", False: "FAIL", None: f"판정 불가({reason})"}[overall["m2_gate"]]
    print(f"\nM2 게이트: {m2}  (best={overall['best_run']}, 기준치 {base}+{GATE_MARGIN})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="M2 ablation 야간 배치 러너 (스펙: "
                                 "docs/superpowers/specs/2026-06-11-ablation-runner-design.md)")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--epochs", type=int, default=None, help="단축 스모크용 fit 패스스루")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--skip-baselines", action="store_true")
    g.add_argument("--force-baselines", action="store_true")
    args = ap.parse_args(argv)
    config = Path(args.config).resolve()
    out_root = Path(args.out_root).resolve()
    logs = out_root / "ablation-logs"

    errs = preflight(config, args.device, out_root)
    if errs:
        for e in errs:
            print(f"프리플라이트 실패: {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        print_plan(config, out_root, args)
        return 0
    logs.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    baselines_path = out_root / "baselines.json"
    need_baselines = args.force_baselines or (
        not baselines_path.exists() and not args.skip_baselines)
    if need_baselines:
        with open(logs / "baselines.log", "a", encoding="utf-8") as logf:
            if not _run_logged(build_baselines_cmd(config, out_root, args.device),
                               logf, "baselines"):
                print("baselines 실패 — 중단. 마진 기준치 없는 야간 배치는 무의미 "
                      f"(로그: {logs / 'baselines.log'})", file=sys.stderr)
                return 2
    elif (baselines_path.exists()
          and config.stat().st_mtime > baselines_path.stat().st_mtime):
        print(f"경고: {config.name}이 baselines.json보다 최신 — 매니페스트 불일치 위험"
              "(README ⚠). --force-baselines 고려", file=sys.stderr)

    gate_baseline = None
    if baselines_path.exists():
        try:
            gate_baseline = float(json.loads(
                baselines_path.read_text(encoding="utf-8"))["gate_baseline_pck02"])
        except (ValueError, KeyError, TypeError, OSError):
            print(f"baselines.json 파손 — --force-baselines로 재생성 필요: {baselines_path}",
                  file=sys.stderr)
            return 2

    judged = []
    for run in RUNS:
        status = run_one(run, config, out_root, args.epochs, args.device)
        report = _read_report(out_root, run["name"])
        judged.append({"name": run["name"], "status": status,
                       "log": str(logs / f"{run['name']}.log"),
                       **judge_run(report, gate_baseline)})
    overall = summarize(judged)

    summary = {"config": str(config),
               "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
               "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "argv": list(argv) if argv is not None else sys.argv[1:],
               "gate_baseline_pck02": gate_baseline, "runs": judged, **overall}
    (out_root / "ablation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print_table(judged, overall, gate_baseline)
    return 1 if any(j["status"] == "error" for j in judged) else 0


if __name__ == "__main__":
    sys.exit(main())
