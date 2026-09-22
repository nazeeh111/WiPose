class LinkTracker:
    """링크(rx_i, tx_j)별 seq 결손 추적 (RF 손실과 전송 손실 분리 진단).

    update(seq) 반환: None(정상) | "gap"(결손 가산) | "reset"(seq 후퇴 — TX 리부트
    또는 START 재시작으로 간주, 결손 비가산·baseline 재설정).
    """

    def __init__(self):
        self.received = 0
        self.lost = 0
        self.resets = 0
        self._last = None

    def update(self, seq: int):
        event = None
        if self._last is not None:
            if seq > self._last + 1:
                self.lost += seq - self._last - 1
                event = "gap"
            elif seq <= self._last:
                self.resets += 1
                event = "reset"
        self.received += 1
        self._last = seq
        return event

    def rebaseline(self):
        """다음 update를 baseline으로 — RX 리부트 공백을 RF 손실로 오인 방지."""
        self._last = None

    @property
    def loss_ratio(self) -> float:
        total = self.received + self.lost
        return self.lost / total if total else 0.0
