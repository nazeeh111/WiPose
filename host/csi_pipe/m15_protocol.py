""" M1.5 미니캡처 프로토콜 — 플랜·세그먼트 상태기계.

stdlib 전용 — 윈도우 운영 콘솔(tools/m15_protocol.py)에서 구동, numpy/h5py 비의존.
격자 1~9 = TX측→RX측 행우선, 매트 = 5번 칸 고정. 캡처2 순회는 고정 셔플
(캡처 내 시간 드리프트가 위치 신호로 가장되는 leak 차단)."""

PLAN_VERSION = "m15-v1"
ORDERS = {1: (1, 2, 3, 4, 5, 6, 7, 8, 9), 2: (5, 1, 8, 3, 7, 2, 9, 4, 6)}


def plan_segments(capture):
    """캡처 번호(1|2) → 13세그먼트 [{idx,pos,posture,empty,hint_s}]."""
    if capture not in ORDERS:
        raise SystemExit(f"capture는 1|2 — 입력 {capture}")
    segs = [{"pos": None, "posture": None, "empty": True, "hint_s": 60}]
    segs += [{"pos": p, "posture": "stand", "empty": False, "hint_s": 40}
             for p in ORDERS[capture]]
    segs += [{"pos": 5, "posture": "sit", "empty": False, "hint_s": 40},
             {"pos": 5, "posture": "lie", "empty": False, "hint_s": 60},
             {"pos": None, "posture": None, "empty": True, "hint_s": 60}]
    return [{"idx": i, **s} for i, s in enumerate(segs)]


class M15Session:
    """경계 키 이벤트 → 세그먼트 확정. mark 토글: 이동→정착(시작)/정착→이동(끝).

    undo = 직전 경계 1개 취소(반복 가능): 시작 취소 → 이동 복귀, 끝 취소 → 재개.
    done이어도 undo로 마지막 경계 되돌리기 가능(마지막 끝을 너무 빨리 누른 실수 복구).
    abort 후에는 mark/undo 불가(종료 상태) — 확정 경계 보존.
    시각은 호출측 주입(time.time_ns) — 비증가는 ValueError(시계 이상 방어, CLI가
    잡아 무시). result는 완주·중단 상태에서만 — 미확정(열린) 세그먼트는 버린다."""

    def __init__(self, capture, session):
        self.capture = capture
        self.session = session
        self.plan = plan_segments(capture)
        self.k = 0                # 현재(미확정) 세그먼트 인덱스
        self.settled = False      # True = 시작 경계 찍힘(진행 중)
        self._open_t = None
        self.bounds = []          # 확정 [(t_start, t_end)] — plan[i] 인덱스 대응
        self.aborted = False

    @property
    def done(self):
        return self.k >= len(self.plan)

    def current(self):
        return None if self.done else self.plan[self.k]

    def _last_t(self):
        if self.settled:
            return self._open_t
        return self.bounds[-1][1] if self.bounds else 0

    def mark(self, t_ns):
        if self.aborted:
            raise ValueError("중단됨 — 추가 조작 불가")
        if self.done:
            raise ValueError("플랜 종료 — 추가 mark 불가")
        if t_ns <= self._last_t():
            raise ValueError(f"시각 역행: {t_ns} ≤ {self._last_t()}")
        if not self.settled:
            self._open_t = t_ns
            self.settled = True
            return "start"
        self.bounds.append((self._open_t, t_ns))
        self._open_t = None
        self.settled = False
        self.k += 1
        return "end"

    def undo(self):
        if self.aborted:
            raise ValueError("중단됨 — 추가 조작 불가")
        if self.settled:
            self.settled = False
            self._open_t = None
            return "start_cancelled"
        if self.bounds:
            start, _ = self.bounds.pop()
            self.k -= 1
            self.settled = True
            self._open_t = start
            return "end_cancelled"
        return None

    def abort(self):
        self.aborted = True
        self.settled = False
        self._open_t = None

    def result(self):
        if not (self.done or self.aborted):
            raise ValueError("미완료·미중단 — result 불가 (done 또는 abort 후 호출)")
        segments = []
        for i, (s, e) in enumerate(self.bounds):
            p = self.plan[i]
            segments.append({"idx": p["idx"], "pos": p["pos"],
                             "posture": p["posture"], "empty": p["empty"],
                             "t_start_ns": s, "t_end_ns": e})
        return {"session": self.session, "capture": self.capture,
                "plan_version": PLAN_VERSION, "segments": segments,
                "aborted": self.aborted}
