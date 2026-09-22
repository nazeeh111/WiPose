"""rawlog(브리지 원본) → HDF5 세션 재구축.

unwrap은 rx 보드 단위로 파일 경계를 넘어 연속 (브리지 재시작으로 로그가 쪼개져도
boot_id가 같으면 같은 에포크). StreamParser는 파일마다 새로 (리싱크 경계)."""
from csi_host.framing import StreamParser
from csi_host.rawlog import read_rawlog
from csi_host.unwrap import TimeUnwrapper

from .store import SessionWriter


def rebuild_session(rx_paths, out_path, *, session, extra_meta=None, progress=None):
    """rx_paths: {rx_id: [rawlog 경로 (시간순)]} → out_path HDF5. {rx: {frames, crc, mismatch}} 반환."""
    meta = {"session": session,
            "sources": {str(rx): [str(p) for p in ps] for rx, ps in rx_paths.items()}}
    meta.update(extra_meta or {})
    w = SessionWriter(out_path, meta=meta)
    stats = {}
    try:
        for rx_id, paths in sorted(rx_paths.items()):
            unwrap = TimeUnwrapper()
            frames = crc = mismatch = 0
            for p in paths:
                parser = StreamParser()
                for t_ns, chunk in read_rawlog(p):
                    for kind, val in parser.feed(chunk):
                        if kind != "frame":
                            continue
                        if val.rx_id != rx_id:
                            mismatch += 1
                            continue
                        u, _ = unwrap.update(boot_id=val.boot_id,
                                             t_us=val.esp_timer_us)
                        w.append(rx_id=val.rx_id, tx_idx=val.tx_idx, t_ns=t_ns,
                                 esp_us=u, iq=val.iq, seq=val.seq, rssi=val.rssi,
                                 noise=val.noise_floor, boot_id=val.boot_id)
                        frames += 1
                crc += parser.crc_errors
                if progress:
                    progress(f"rx{rx_id} {p}: 누적 {frames}프레임")
            stats[rx_id] = {"frames": frames, "crc": crc, "mismatch": mismatch}
            w.set_meta(f"rx{rx_id}_crc_errors", crc)
    except BaseException:
        try:
            w.close()       # 파손 상태의 close 실패가 원인 예외를 가리지 않게
        except Exception:
            pass
        raise
    else:
        w.close()           # 정상 경로에서는 close 오류도 표면화
    return stats
