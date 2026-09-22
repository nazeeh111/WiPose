#include "csi_link/wire.h"

/* CRC16-CCITT-FALSE — 호스트 csi_host.crc16.crc16_ccitt와 동일 (교차검증: test_crc_cross.py) */
uint16_t csil_crc16(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++)
            crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
    return crc;
}
