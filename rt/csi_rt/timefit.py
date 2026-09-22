"""인과 시간 재구성 — 브리지 청크 스탬프(동일 t_ns 런 ~11pkt·주기 ~105ms 실측) 보정.

t̂ = esp_ns + offset(RX별 rolling-min) — min = '가장 덜 늦은' 표본(청크 끝 프레임)'t_ns 직사용'은 실측으로 기각 — 본 모듈이 예약된 폴백."""
from collections import deque


class CausalOffset:
    """RX 1대의 esp→호스트 오프셋 rolling-min 추정기. boot 변화 시 reset."""

    def __init__(self, *, window_ns=3_000_000_000):
        self._w = window_ns
        self._d = deque()                            # (t_ns, off) — off 단조 min-deque
        self._boot = None

    def update(self, boot_id, t_ns, esp_ns):
        """패킷 1개 관측 → 현재 오프셋(ns) 반환. 워밍업(첫 패킷)부터 사용 가능."""
        if self._boot is not None and boot_id != self._boot:
            self._d.clear()                          # 리부트 — esp 클록 재시작
        self._boot = boot_id
        off = t_ns - esp_ns
        # 동률은 신규(늦은 t_ns) 표본 유지 — 윈도 잔류 극대화(rolling-min 정확성)
        while self._d and self._d[-1][1] >= off:     # min-deque 유지
            self._d.pop()
        self._d.append((t_ns, off))
        while self._d and self._d[0][0] < t_ns - self._w:
            self._d.popleft()
        return self._d[0][1]

    def estimate(self, boot_id, t_ns, esp_ns):
        """update + t̂ 산출."""
        return esp_ns + self.update(boot_id, t_ns, esp_ns)
