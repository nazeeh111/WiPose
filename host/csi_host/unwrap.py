class TimeUnwrapper:
    """esp_timer_us(u32, 71.58분 랩)를 보드 단위로 단조 시각으로 복원.

    update() 반환: (unwrapped_us, event)  event ∈ {"", "wrap", "reboot"}.
    boot_id 변경 = 리부트 → 에포크 리셋 (랩 누적과 구별 — 클록 핏도 리셋 대상).
    """
    WRAP = 1 << 32

    def __init__(self):
        self.boot_id = None
        self.last_raw = None
        self.epoch = 0
        self.wraps = 0
        self.reboots = 0

    def update(self, *, boot_id: int, t_us: int):
        event = ""
        if self.boot_id is None:
            self.boot_id = boot_id
        elif boot_id != self.boot_id:
            self.boot_id, self.epoch, self.last_raw = boot_id, 0, None
            self.reboots += 1
            event = "reboot"
        if self.last_raw is not None and t_us < self.last_raw:
            self.epoch += 1
            self.wraps += 1
            event = "wrap"
        self.last_raw = t_us
        return self.epoch * self.WRAP + t_us, event
