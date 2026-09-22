"""소크 rawlog → 컬럼 배열 + 링크/에포크 통계.

rawlog는 /tmp 복사본을 권장 (9P 직접 읽기는 수 배 느림 — 분석 노하우).
"""
import array
from dataclasses import dataclass

import numpy as np

from csi_host.framing import StreamParser
from csi_host.rawlog import read_rawlog
from csi_host.unwrap import TimeUnwrapper


@dataclass
class BoardLog:
    path: str
    t_ns: np.ndarray      # i64 — 브리지 수신 시각
    esp_us: np.ndarray    # u64 — unwrap 적용
    seq: np.ndarray       # i64
    tx: np.ndarray        # i8
    rssi: np.ndarray      # i8
    noise: np.ndarray     # i8
    boot: np.ndarray      # u8
    rx_ids: np.ndarray    # u8
    frames: int
    crc_errors: int
    junk_bytes: int
    texts: int
    wraps: int
    reboots: int


def collect_rawlog(path):
    """rawlog 1개(=rx 보드 1개) 재생 → 컬럼 배열. unwrap은 보드 단위로 여기서 적용.

    crc_errors/junk_bytes/texts는 파일 전체 카운터 — --window 필터 비적용 (보수 채점).
    """
    t_a, e_a, s_a = array.array("q"), array.array("Q"), array.array("q")
    tx_a, rs_a, nf_a = array.array("b"), array.array("b"), array.array("b")
    bt_a, rx_a = array.array("B"), array.array("B")
    parser, unwrap = StreamParser(), TimeUnwrapper()
    texts = 0
    for t_ns, chunk in read_rawlog(path):
        for kind, val in parser.feed(chunk):
            if kind == "frame":
                u, _ = unwrap.update(boot_id=val.boot_id, t_us=val.esp_timer_us)
                t_a.append(t_ns)
                e_a.append(u)
                s_a.append(val.seq)
                tx_a.append(val.tx_idx)
                rs_a.append(val.rssi)
                nf_a.append(val.noise_floor)
                bt_a.append(val.boot_id)
                rx_a.append(val.rx_id)
            elif kind == "text":
                texts += 1

    def np_(a, dt):
        return np.frombuffer(a, dtype=dt).copy() if len(a) else np.empty(0, dt)

    return BoardLog(
        path=str(path),
        t_ns=np_(t_a, np.int64), esp_us=np_(e_a, np.uint64), seq=np_(s_a, np.int64),
        tx=np_(tx_a, np.int8), rssi=np_(rs_a, np.int8), noise=np_(nf_a, np.int8),
        boot=np_(bt_a, np.uint8), rx_ids=np_(rx_a, np.uint8),
        frames=len(t_a), crc_errors=parser.crc_errors, junk_bytes=parser.junk_bytes,
        texts=texts, wraps=unwrap.wraps, reboots=unwrap.reboots,
    )


def time_window_mask(t_ns, window):
    """window "HH:MM-HH:MM" (KST, 자정 걸침 허용) → bool 마스크. None이면 전체 True."""
    t = np.asarray(t_ns, np.int64)
    if not window:
        return np.ones(len(t), bool)
    s, e = window.split("-")

    def mins(x):
        h, m = x.split(":")
        return int(h) * 60 + int(m)

    sm, em = mins(s), mins(e)
    tod = ((t // 60_000_000_000) + 9 * 60) % (24 * 60)  # KST = UTC+9
    if sm > em:  # 자정 걸침
        return (tod >= sm) | (tod < em)
    return (tod >= sm) & (tod < em)


def _epoch_bounds(seq, boot):
    cut = np.flatnonzero((np.diff(seq) <= 0) | (np.diff(boot.astype(np.int64)) != 0)) + 1
    return np.concatenate(([0], cut, [len(seq)]))


def link_stats(b, mask):
    """BoardLog(+윈도 마스크) → {tx: 통계 dict}. 에포크(seq 후퇴/리부트) 분리 후 집계.

    마스크는 링크당 연속 구간 가정 — 앞뒤 잘림은 seq 갭을 안 만들지만 중간 구멍은 가짜 갭 생성.
    """
    out = {}
    for tx in sorted(set(np.asarray(b.tx).tolist())):
        m = (b.tx == tx) & mask
        seq, t = b.seq[m], b.t_ns[m]
        rssi, boot = b.rssi[m], b.boot[m]
        n = len(seq)
        if n < 2:
            out[int(tx)] = {"received": n, "lost": 0, "loss_pct": 0.0, "resets": 0,
                            "bursts": {"1": 0, "2": 0, "3-5": 0, "6+": 0},
                            "discard_pct": 0.0,             # n<2 폴백 — §5.2 폐기율 0
                            "pps10": {"best": 0.0, "worst": 0.0, "median": 0.0,
                                      "full10": False},   # n≥2 폴백과 동일 형태 — 렌더 일관
                            "worst_minutes": [], "rssi": None,
                            "span_s": 0.0, "avg_pps": 0.0}
            continue
        bounds = _epoch_bounds(seq, boot)
        lost = 0
        bursts_all = []
        loss_by_min = {}
        for a, z in zip(bounds[:-1], bounds[1:]):
            d = np.diff(seq[a:z])
            gi = np.flatnonzero(d > 1)
            gaps = d[gi] - 1
            lost += int(gaps.sum())
            bursts_all.append(gaps)
            mins_right = t[a:z][gi + 1] // 60_000_000_000   # 갭을 우변 샘플 분에 귀속
            for mm, g in zip(mins_right.tolist(), gaps.tolist()):
                loss_by_min[mm] = loss_by_min.get(mm, 0) + int(g)
        gaps = np.concatenate(bursts_all) if bursts_all else np.empty(0, np.int64)
        bursts = {"1": int((gaps == 1).sum()), "2": int((gaps == 2).sum()),
                  "3-5": int(((gaps >= 3) & (gaps <= 5)).sum()),
                  "6+": int((gaps >= 6).sum())}
        # §5.2 폐기율: >2연속 결손 프레임은 보간 불가 → 폐기 처리
        total = n + lost
        discard_pct = float(100.0 * gaps[gaps > 2].sum() / total) if total else 0.0
        span = float((t[-1] - t[0]) / 1e9)
        # 10분 롤링 pps (60s 빈)
        pps10 = None
        if span >= 600:
            edges = np.arange(t[0], t[-1] + 1, 60_000_000_000)
            if len(edges) >= 11:
                cnt, _ = np.histogram(t, bins=edges)
                roll = np.convolve(cnt, np.ones(10), "valid") / 600.0
                pps10 = {"best": float(roll.max()), "worst": float(roll.min()),
                         "median": float(np.median(roll)), "full10": True}
        if pps10 is None:
            pps10 = {"best": n / max(span, 1e-9), "worst": n / max(span, 1e-9),
                     "median": n / max(span, 1e-9), "full10": False}
        recv_min, recv_cnt = np.unique(t // 60_000_000_000, return_counts=True)
        recv_map = dict(zip(recv_min.tolist(), recv_cnt.tolist()))
        worst = sorted(loss_by_min.items(), key=lambda kv: -kv[1])[:5]
        worst_minutes = [{"kst": _min_to_kst(mm), "lost": lo,
                          "expected": lo + recv_map.get(mm, 0)} for mm, lo in worst]
        out[int(tx)] = {
            "received": n, "lost": lost,
            "loss_pct": 100.0 * lost / (n + lost) if (n + lost) else 0.0,
            "resets": int(len(bounds) - 2), "bursts": bursts,
            "discard_pct": discard_pct,                     # §5.2 보간 불가 폐기율
            "pps10": pps10,
            "worst_minutes": worst_minutes,
            "rssi": {"mean": float(rssi.mean()), "p5": float(np.percentile(rssi, 5)),
                     "p95": float(np.percentile(rssi, 95)),
                     "noise_mean": float(b.noise[m].mean())},
            "span_s": span, "avg_pps": n / max(span, 1e-9),
        }
    return out


def _min_to_kst(minute_utc):
    mm = int(minute_utc) + 9 * 60
    return f"{(mm // 60) % 24:02d}:{mm % 60:02d}"


def grade_m0(link, board):
    """링크 통계 + 보드 요약 → §13 M0 ('≥95pps 10분 유지' = 롤링 worst 기준).

    v1.4 수용 기준 (§13 M0):
      - loss_ok   : 손실 < 5%
      - burst6_ok : 6+연속 결손 ≤ 2회/시간
      - discard_ok: §5.2 폐기 샘플 비율 < 1%

    링크별 crc_ok/wrap_ok/all_ok는 보드 문맥 참고치(advisory) — 최종 기준은 전역 verdict.
    """
    pps10_ok = bool(link["pps10"] and link["pps10"]["full10"]
                    and link["pps10"]["worst"] >= 95.0)
    loss_ok = bool(link["loss_pct"] < 5.0)                  # v1.4: 3% → 5%
    crc_ok = board["crc_errors"] == 0
    wrap_ok = bool(board["span_min"] >= 75.0 and board["wrap_checks"]
                   and all(w["ok"] for w in board["wrap_checks"]))
    # v1.4: 6+연속 결손 시간당 ≤2회 — span_s=0이면 6+가 없을 때만 OK
    span_s = link.get("span_s", 0.0)
    burst6 = link["bursts"]["6+"]
    if span_s > 0:
        burst6_ok = bool((burst6 / (span_s / 3600)) <= 2.0)
    else:
        burst6_ok = bool(burst6 == 0)
    # v1.4: §5.2 폐기율 < 1%
    discard_ok = bool(link.get("discard_pct", 0.0) < 1.0)
    return {"pps10_ok": pps10_ok, "loss_ok": loss_ok, "crc_ok": crc_ok,
            "wrap_ok": wrap_ok, "burst6_ok": burst6_ok, "discard_ok": discard_ok,
            "all_ok": pps10_ok and loss_ok and crc_ok and wrap_ok and burst6_ok and discard_ok}


def render_report(report):
    """analyze_soak() 결과 dict → 사람용 텍스트."""
    L = []
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    for _, br in sorted(report["boards"].items()):
        rx = br["rx"]
        L.append(f"== rx{rx}  {br['path']}")
        L.append(f"   span={br['span_min']:.1f}min frames={br['frames']} "
                 f"crc(파일전체)={br['crc_errors']} junk={br['junk_bytes']}B "
                 f"wraps={br['wraps']} reboots={br['reboots']}")
        cf = br["clockfit"]
        if cf.get("n"):
            L.append(f"   클록핏: slope={['%.1f' % s for s in cf['slope_ppm']]}ppm "
                     f"잔차 p5/p50/p95/max = {cf['resid_p5_ms']:.2f}/{cf['resid_p50_ms']:.2f}/"
                     f"{cf['resid_p95_ms']:.2f}/{cf['resid_max_ms']:.2f} ms")
        for w in br["wrap_checks"]:
            d = "n/a" if w["delta_ms"] is None else f"{w['delta_ms']:.3f}ms"
            L.append(f"   랩 {w['wrap_at_min']:.1f}min: Δ잔차={d} → {ok(w['ok'])}"
                     + (f" ({w.get('note')})" if w.get("note") else ""))
        for tx, s in sorted(br["links"].items()):
            g = s["grade"]
            r = s["rssi"]
            span_h = s["span_s"] / 3600.0 if s.get("span_s", 0) > 0 else 0.0
            burst6_per_h = s["bursts"]["6+"] / span_h if span_h > 0 else 0.0
            L.append(f"   rx{rx}-tx{tx}: rx={s['received']} lost={s['lost']} "
                     f"loss={s['loss_pct']:.2f}%({ok(g['loss_ok'])}) "
                     f"discard={s.get('discard_pct', 0.0):.2f}%({ok(g['discard_ok'])}) "
                     f"6+/h={burst6_per_h:.1f}({ok(g['burst6_ok'])}) "
                     f"pps10 best/med/worst={s['pps10']['best']:.1f}/"
                     f"{s['pps10']['median']:.1f}/{s['pps10']['worst']:.1f}"
                     f"({ok(g['pps10_ok'])}) resets={s['resets']}")
            L.append(f"      버스트 {s['bursts']} rssi={r['mean']:.1f}"
                     f"[{r['p5']:.0f},{r['p95']:.0f}] noise={r['noise_mean']:.1f}"
                     if r else "      (표본 부족)")
            if s["worst_minutes"]:
                wm = ", ".join(f"{w['kst']}({w['lost']}/{w['expected']})"
                               for w in s["worst_minutes"])
                L.append(f"      워스트 분: {wm}")
    v = report["verdict"]
    L.append(f"== M0 판정: 링크 {v['links_pass']}/{v['links_total']} PASS, "
             f"CRC {ok(v['crc_ok'])}, 랩(75분+) {ok(v['wrap_ok'])} "
             f"→ 전체 {ok(v['all_ok'])}")
    return "\n".join(L)


def analyze_soak(paths, *, window=None, fit_window_s=600.0):
    """rawlog 경로들 → 리포트 dict (boards/{파일경로}/links/{tx} + verdict).

    boards 키는 파일 경로 — 같은 rx 파일 다수(사전 테스트+본 테스트)도 전부 판정 반영.
    """
    from csi_pipe.clockfit import fit_board, wrap_continuity
    boards = {}
    for p in paths:
        b = collect_rawlog(p)
        if b.frames == 0:
            continue
        rx = int(np.bincount(b.rx_ids).argmax())  # 지배 rx_id (불일치는 채록만)
        m = time_window_mask(b.t_ns, window)
        cf = {"n": 0}
        wraps = []
        if m.sum() >= 200:
            model, rep = fit_board(b.esp_us[m].astype(np.float64), b.t_ns[m],
                                   b.boot[m], window_s=fit_window_s)
            cf = rep.stats()
            wraps = wrap_continuity(b.esp_us[m].astype(np.float64),
                                    rep.resid_ns, rep.valid)
        t_w = b.t_ns[m]
        span_min = float((t_w[-1] - t_w[0]) / 6e10) if m.sum() >= 2 else 0.0
        board = {"path": b.path, "rx": rx, "frames": int(m.sum()),
                 "crc_errors": b.crc_errors,
                 "junk_bytes": b.junk_bytes, "texts": b.texts, "wraps": b.wraps,
                 "reboots": b.reboots, "span_min": span_min,
                 "clockfit": cf, "wrap_checks": wraps}
        links = link_stats(b, m)
        for tx, s in links.items():
            s["grade"] = grade_m0(s, board)
        board["links"] = links
        boards[str(p)] = board
    links_all = [(s["grade"]) for br in boards.values() for s in br["links"].values()]
    verdict = {
        "links_total": len(links_all),
        "links_pass": sum(1 for g in links_all if g["all_ok"]),
        "crc_ok": all(br["crc_errors"] == 0 for br in boards.values()) if boards else False,
        "wrap_ok": all(br["span_min"] >= 75.0 and br["wrap_checks"]
                       and all(w["ok"] for w in br["wrap_checks"])
                       for br in boards.values()) if boards else False,
    }
    verdict["all_ok"] = (boards != {} and verdict["links_pass"] == verdict["links_total"]
                         and verdict["crc_ok"] and verdict["wrap_ok"])
    return {"boards": boards, "verdict": verdict, "window": window}
