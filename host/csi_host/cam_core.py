"""CamCore — 프레임 시각 → cam/meta 발행 (송신측; 영상 저장은 CLI 담당).

sink 주입(bridge NullSink/MqttSink 덕타입)으로 cv2·MQTT 없이 테스트."""
try:
    import msgpack
except ImportError:
    msgpack = None


class CamCore:
    """웹캠 프레임 시각을 cam/meta 토픽으로 발행하는 코어 로직.

    Parameters
    ----------
    sink:
        ``publish(topic, payload)`` 덕타입. NullSink 또는 MqttSink.
    on_event:
        선택 콜백 ``(kind: str, val) -> None`` — 확장용, 현재 미사용.
    """

    def __init__(self, sink, *, on_event=None):
        if msgpack is None:
            raise RuntimeError("cam_capture는 msgpack 필수 — pip install msgpack")
        self._sink = sink
        self._on_event = on_event
        self._frames = 0        # 성공 handle_frame 횟수
        self._drops = 0         # 누적 드롭 수
        self._frame_idx = 0     # 다음에 발행할 frame_idx (단조 증가)
        self._consec_drops = 0  # 연속 드롭 카운터 — handle_frame 성공 시 0 리셋

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def handle_frame(self, t_ns) -> int:
        """프레임 grab 시각을 cam/meta로 발행하고 사용한 frame_idx를 반환.

        Parameters
        ----------
        t_ns:
            grab 직후 호스트 시계 (``time.time_ns()``).

        Returns
        -------
        int
            이번 프레임에 할당된 frame_idx (0부터 단조 증가).
        """
        idx = self._frame_idx
        payload = msgpack.packb({"frame_idx": idx, "t_ns": int(t_ns)})
        self._sink.publish("cam/meta", payload)
        self._frames += 1
        self._frame_idx += 1
        self._consec_drops = 0  # 성공 → 연속 드롭 리셋
        return idx

    def note_drop(self) -> int:
        """프레임 read 실패를 기록하고 현재 연속 드롭 횟수를 반환.

        Returns
        -------
        int
            현재까지의 연속 드롭 수 (handle_frame 성공 시 리셋).
            CLI가 이 값으로 임계 초과 여부를 판단한다.
        """
        self._drops += 1
        self._consec_drops += 1
        return self._consec_drops

    def status(self) -> dict:
        """현재 집계 상태를 반환.

        Returns
        -------
        dict
            ``{"frames": int, "drops": int, "frame_idx": int}``
        """
        return {
            "frames": self._frames,
            "drops": self._drops,
            "frame_idx": self._frame_idx,
        }
