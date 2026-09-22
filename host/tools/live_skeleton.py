#!/usr/bin/env python3
"""웹캠 라이브 스켈레톤 뷰어 — 화각·트래킹 즉석 점검용 (Windows에서 실행).

teacher의 RTMDet→RTMPose 러너를 그대로 재사용해 COCO-17 골격을 실시간 오버레이.
수집 파이프라인 무관(MQTT·저장 없음).

    python host\\tools\\live_skeleton.py                # 기본: cam0·MSMF·720p·CPU
    python host\\tools\\live_skeleton.py --device cuda  # GPU EP 설치 시

키: ESC/q 종료, m 미러 토글. 첫 실행은 모델 ~120MB 자동 다운로드.
⚠ MSMF 카메라 단일 점유 — cam_capture.py 기동 전 반드시 종료.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 추적 파라미터 (스펙 §요구 — CPU 실시간화 2-모드 루프)
SEARCH_DET_EVERY = 10   # 탐색 모드: det 주기 (프레임)
TRACK_DET_PERIOD = 1.0  # 추적 모드: det 재확인 주기 (초)
KPT_THR = 0.3           # 유효 관절·인물 드랍 판정 score
MIN_VALID_KPTS = 4      # 외접박스 최소 유효 관절 수
BBOX_MARGIN = 0.2       # bbox 승계 마진 비율
MAX_PERSONS = 3
READ_FAIL_MAX = 30      # 연속 read 실패 한도


def kpts_to_bbox(kpts, score_thr, margin, frame_wh):
    """(17,3)[x,y,score] → 유효 관절 외접박스+마진(프레임 클립). frame_wh=(width,height). 유효 <4 → None."""
    k = np.asarray(kpts, np.float32)
    pts = k[k[:, 2] >= score_thr, :2]
    if len(pts) < MIN_VALID_KPTS:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    w, h = frame_wh
    return np.array([max(0.0, x1 - mx), max(0.0, y1 - my),
                     min(w - 1.0, x2 + mx), min(h - 1.0, y2 + my)], np.float32)


class ViewerCore:
    """탐색/추적 2-모드 스케줄러 — runner 주입식 (cv2 GUI·카메라 무의존).

    탐색(추적 인원 0): det를 SEARCH_DET_EVERY 프레임마다.
    추적: 매 프레임 pose-only(bbox는 직전 키포인트 외접박스 승계),
          det 재확인 TRACK_DET_PERIOD마다 — 신규 인물 반영·추적 종료 판단 겸함.
    """

    def __init__(self, runner, det_thr=0.5):
        self.runner = runner
        self.det_thr = det_thr
        self.bboxes = []          # 추적 중 인물별 (4,) bbox — 비면 탐색 모드
        self.det_ms = 0.0
        self.pose_ms = 0.0
        self._frame_idx = 0
        self._last_det_t = None

    @property
    def tracking(self):
        return bool(self.bboxes)

    def _det_due(self, now):
        if not self.tracking:
            return self._frame_idx % SEARCH_DET_EVERY == 0
        return now - self._last_det_t >= TRACK_DET_PERIOD

    def step(self, frame, now):
        """1프레임 처리 → (인물별 (17,3) kpts 리스트, hud dict)."""
        if self._det_due(now):
            t0 = time.perf_counter()
            dets = self.runner.detect(frame)
            self.det_ms = (time.perf_counter() - t0) * 1e3
            self._last_det_t = now
            # dets는 runner가 score 내림차순 보장(runner.py detect) — [:MAX_PERSONS]가 상위 N
            self.bboxes = [d[:4] for d in dets if d[4] >= self.det_thr][:MAX_PERSONS]

        h, w = frame.shape[:2]
        persons, next_bboxes = [], []
        t0 = time.perf_counter()
        for bbox in self.bboxes:
            kpts = self.runner.pose(frame, bbox)
            if kpts[:, 2].mean() < KPT_THR:
                continue                      # 저신뢰 인물 드랍 — 전원 소실 시 탐색 복귀
            nb = kpts_to_bbox(kpts, KPT_THR, BBOX_MARGIN, (w, h))
            if nb is None:
                continue
            persons.append(kpts)
            next_bboxes.append(nb)
        self.pose_ms = (time.perf_counter() - t0) * 1e3
        self.bboxes = next_bboxes
        self._frame_idx += 1
        return persons, {"det_ms": self.det_ms, "pose_ms": self.pose_ms,
                         "n": len(persons), "tracking": self.tracking}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--backend", choices=["msmf", "dshow", "any"], default="msmf",
                    help="cv2 캡처 백엔드 (cam_capture 동형 — MSMF가 30fps 협상)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                    help="추론 디바이스 — Windows 기본 cpu (auto는 CUDA 시도 후 "
                         "폴백이라 모델 이중 초기화)")
    ap.add_argument("--det-thr", type=float, default=0.5, dest="det_thr",
                    help="사람 검출 score 임계 (기본 0.5)")
    ap.add_argument("--mirror", action="store_true", help="미러 표시로 시작 (m 토글)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        import cv2
        from rtmlib import draw_skeleton
    except ImportError as e:                  # 미설치 안내 (스펙 §사전 설치)
        print(f"오류: 의존성 미설치({e.name}) — Windows PowerShell에서:\n"
              "  pip install --no-deps rtmlib==0.0.15\n"
              "  pip install onnxruntime tqdm", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "teacher"))
    from csi_teacher.runner import make_runner

    print("[live] 러너 초기화 — 첫 실행은 모델 ~120MB 자동 다운로드", flush=True)
    core = ViewerCore(make_runner(device=args.device), det_thr=args.det_thr)

    # 카메라 오픈 — cam_capture.py §③④ 동형 (MSMF 기본·MJPG 선설정·버퍼 1)
    backends = {"msmf": "CAP_MSMF", "dshow": "CAP_DSHOW", "any": None}
    bk = backends[args.backend]
    cap = (cv2.VideoCapture(args.camera, getattr(cv2, bk))
           if bk and hasattr(cv2, bk) else cv2.VideoCapture(args.camera))
    if not cap.isOpened():
        print(f"오류: 카메라 {args.camera} 오픈 실패 — --camera/--backend 확인, "
              "점유 프로세스(cam_capture 등) 종료", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 최신 프레임 우선
    except Exception:
        pass

    mirror = args.mirror
    fps = 0.0
    t_prev = time.monotonic()
    fails = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= READ_FAIL_MAX:
                    print(f"오류: 프레임 read 연속 {READ_FAIL_MAX}회 실패",
                          file=sys.stderr)
                    return 1
                continue
            fails = 0
            if mirror:
                frame = cv2.flip(frame, 1)    # 추론 전에 뒤집어 좌표계 일치

            now = time.monotonic()
            persons, hud = core.step(frame, now)
            dt = max(now - t_prev, 1e-6)
            t_prev = now
            fps = (1.0 / dt) if fps == 0.0 else fps * 0.9 + (1.0 / dt) * 0.1

            if persons:
                k = np.stack(persons)         # draw_skeleton은 배치 (N,17,…)만 허용
                frame = draw_skeleton(frame, k[:, :, :2], k[:, :, 2],
                                      openpose_skeleton=False, kpt_thr=KPT_THR)
            txt = (f"fps {fps:4.1f}  det {hud['det_ms']:5.1f}ms  "
                   f"pose {hud['pose_ms']:5.1f}ms  N={hud['n']}  {args.device}"
                   f"{'  MIRROR' if mirror else ''}")
            cv2.putText(frame, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "[ESC/q] quit  [m] mirror",
                        (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("live_skeleton", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
                core.bboxes = []              # 좌표계 반전 — 승계 bbox 무효화
            if cv2.getWindowProperty("live_skeleton", cv2.WND_PROP_VISIBLE) < 1:
                break                         # X 버튼 닫힘 — 다음 imshow가 창 재생성하기 전 종료
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
