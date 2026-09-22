"""낙상 규칙 + 상태기계 — 3중 2 발화, 유지 확인은 pose+CSI 결합.

aspect = 픽셀 공간(pck.frame_geometry 재사용, 코어·c<0.3 제외).
R1·R3·theta_still = 정규화 공간. θ 전부 rt.yaml 잠정값(캘리브는 라운드1 낙상으로만).
theta_csi_still ≤ 0 = 미캘리브 → CSI 조건 스킵 + 경고 1회 전이 상태."""
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np

from csi_train.pck import C_MIN, CORE, frame_geometry

NOSE, RHIP, LHIP = 0, 8, 11


@dataclass(frozen=True)
class FallOut:
    state: str                                      # IDLE | IMPACT | ALARM
    fired: bool                                     # ALARM 진입 틱에만 True


class FallDetector:
    def __init__(self, cfg, WH):
        self.c = dict(cfg)
        self.WH = np.asarray(WH, np.float32)
        if self.c["theta_csi_still"] <= 0:
            print("[fall] theta_csi_still 미캘리브(≤0) — CSI 정지 조건 스킵 "
                  "(configs/rt.yaml 주석 참조)", file=sys.stderr)
        self.state = "IDLE"
        self._hip = deque()                         # (t, hip_y) ≤0.3s
        self._aspect = deque()                      # (t, aspect) ≤transition_s
        self._impact_t = None
        self._confirm = []                          # (lying, still)
        self._prev_core = None
        self._refract_until = -1e9
        self._stand_since = None
        self._absent_since = None

    def _rules(self, t, xy, c, aspect, n_core):
        hips = [j for j in (RHIP, LHIP) if c[j] >= C_MIN]
        r1 = r2 = r3 = False
        if hips:
            self._hip.append((t, float(np.mean(xy[hips, 1]))))
            while self._hip and self._hip[0][0] < t - 0.3:
                self._hip.popleft()
            if len(self._hip) >= 4:
                ts, ys = np.array(self._hip).T
                r1 = np.polyfit(ts - ts[0], ys, 1)[0] > self.c["theta_v"]
        if np.isfinite(aspect) and n_core >= 3:
            self._aspect.append((t, aspect))
        while self._aspect and self._aspect[0][0] < t - self.c["transition_s"]:
            self._aspect.popleft()
        if self._aspect and np.isfinite(aspect):
            r2 = (max(a for _, a in self._aspect) >= self.c["aspect_hi"]
                  and aspect <= self.c["aspect_lo"])
        if c[NOSE] >= C_MIN:
            r3 = xy[NOSE, 1] > self.c["head_y"]
        return r1, r2, r3

    def _pose_still(self, xy, c):
        core = [j for j in CORE if c[j] >= C_MIN]
        cur = xy[core] if len(core) >= 3 else None
        moved = np.inf
        if cur is not None and self._prev_core is not None \
                and len(cur) == len(self._prev_core):
            moved = float(np.sqrt(np.mean((cur - self._prev_core) ** 2)))
        self._prev_core = cur
        return moved < self.c["theta_still"]

    def update(self, t_s, xy_norm, c, present, motion) -> FallOut:
        fired = False
        if not present:
            self._absent_since = self._absent_since or t_s
            if t_s - self._absent_since < self.c["grace_s"]:
                return FallOut(self.state, False)    # 유예 — 낙상 순간 ĉ 일시 급락(<1s 실측,
                                                     # rt-handoff-20260612): 이력·IMPACT 보존, 프레임만 스킵
            self._hip.clear()
            self._aspect.clear()                     # R2 = '관찰된 전환' — 유예 초과 absent 너머 이력 무효
            self._prev_core = None
            if (self.state == "ALARM" and t_s >= self._refract_until
                    and t_s - self._absent_since >= self.c["absent_release_s"]):
                self.state = "IDLE"                  # 해제 조건은 불응 경과 후에만
            elif self.state == "IMPACT":             # 유예 초과 확인 불능 — 보수적으로 철회
                self.state = "IDLE"
            return FallOut(self.state, False)
        self._absent_since = None
        xy = np.asarray(xy_norm, np.float32)
        _, _, aspect, n_core = frame_geometry(xy * self.WH, c)
        r1, r2, r3 = self._rules(t_s, xy, c, aspect, n_core)
        lying = np.isfinite(aspect) and aspect <= self.c["aspect_lo"]
        still_pose = self._pose_still(xy, c)
        still_csi = (self.c["theta_csi_still"] <= 0
                     or motion < self.c["theta_csi_still"])

        if self.state == "IDLE":
            if t_s >= self._refract_until and int(r1) + int(r2) + int(r3) >= 2:
                self.state, self._impact_t, self._confirm = "IMPACT", t_s, []
        elif self.state == "IMPACT":
            if np.isfinite(aspect) and n_core >= 3:  # 확인 가능 프레임만 적재 — 저신뢰는 분모
                self._confirm.append((lying, still_pose and still_csi))  # 제외(핸드오프 수정 ②)
            if t_s - self._impact_t >= self.c["confirm_s"]:
                if not self._confirm:                # 확인 가능 프레임 전무 — 판정 불능 철회
                    self.state = "IDLE"
                else:
                    arr = np.array(self._confirm, bool)
                    ok = (arr[:, 0].mean() >= self.c["lying_ratio"]
                          and arr[:, 1].mean() >= self.c["lying_ratio"])
                    self.state = "ALARM" if ok else "IDLE"
                    fired = self.state == "ALARM"
                    if fired:
                        self._refract_until = t_s + self.c["refractory_s"]
        elif self.state == "ALARM":
            standing = (np.isfinite(aspect) and aspect >= self.c["aspect_hi"]
                        and len([j for j in (RHIP, LHIP) if c[j] >= C_MIN]) > 0
                        and float(np.mean(xy[[j for j in (RHIP, LHIP)
                                              if c[j] >= C_MIN], 1]))
                        < self.c["release_hip_y"])
            if t_s >= self._refract_until:
                self._stand_since = (self._stand_since or t_s) if standing else None
                if self._stand_since is not None \
                        and t_s - self._stand_since >= self.c["release_hold_s"]:
                    self.state, self._stand_since = "IDLE", None
        return FallOut(self.state, fired)
