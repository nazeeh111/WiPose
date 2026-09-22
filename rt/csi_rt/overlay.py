"""오버레이 렌더 — 캔버스 기본, --video 시 [영상+골격 | 캔버스] 나란히.

관절 순서 = teacher BODY-18(coco17_to_body18 산출 순서). 엣지는 OpenPose BODY-18
표준 18연결 — teacher에 엣지 상수가 없어 여기 1회 정의(이중 정의 금지)."""
import cv2
import numpy as np

CANVAS_WH = (640, 480)
C_MIN = 0.3
EDGES = ((1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17))


def project(xy_norm, W, H):
    xy = np.nan_to_num(np.asarray(xy_norm, np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    px = xy * [W, H]
    return np.clip(px, [0, 0], [W - 1, H - 1]).astype(np.int32)


def _draw_skeleton(img, xy, c, W, H):
    xy = np.asarray(xy, np.float32)
    finite = np.isfinite(xy).all(axis=1)  # per-joint 좌표 NaN/Inf 게이트 — 코너 점·선 차단
    px = project(xy, W, H)
    for a, b in EDGES:
        if finite[a] and finite[b] and c[a] >= C_MIN and c[b] >= C_MIN:
            cv2.line(img, tuple(px[a]), tuple(px[b]), (0, 200, 0), 2)
    for j in range(len(px)):
        if finite[j] and c[j] >= C_MIN:
            g = int(np.clip(c[j], 0, 1) * 255)
            cv2.circle(img, tuple(px[j]), 4, (0, g, 255 - g), -1)


def _banner(img, present, fall_state, hud):
    # fall.py FSM 상태 어휘(IDLE/IMPACT/ALARM)와 결합 — 미지 상태 = KeyError(fail-loud)
    color = {"IDLE": (80, 80, 80), "IMPACT": (0, 160, 255),
             "ALARM": (0, 0, 230)}[fall_state]
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), color, -1)
    tag = " RANDOM" if hud["random"] else ""
    cv2.putText(img, f"{'PRESENT' if present else 'ABSENT'} | {fall_state}{tag}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    txt = (f"fps {hud['fps']:.1f}  infer {hud['infer_ms']:.1f}ms  "
           f"e2e {hud['e2e_ms']:.0f}ms  drop {hud['drop']}  motion {hud['motion']:.2f}")
    cv2.putText(img, txt, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1)


def render(video_frame, xy_norm, c, *, present, fall_state, hud):
    """video_frame=None → 캔버스 단독. ndarray → [영상+골격 | 캔버스] 가로 연결.

    hud 필수 6키: fps·infer_ms·e2e_ms·drop·motion·random."""
    W, H = CANVAS_WH
    canvas = np.zeros((H, W, 3), np.uint8)
    if present and xy_norm is not None:
        _draw_skeleton(canvas, xy_norm, c, W, H)
    _banner(canvas, present, fall_state, hud)
    if video_frame is None:
        return canvas
    vf = cv2.resize(video_frame, (W, H))
    if present and xy_norm is not None:
        _draw_skeleton(vf, xy_norm, c, W, H)
    return np.hstack([vf, canvas])
