/* 와이어 포맷 (호스트 csi_host.framing과 단일 규약).
 * 주의: 이 헤더는 gcc 교차검증 테스트가 호스트에서 컴파일한다 — ESP 헤더 include 금지.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#define CSIL_PAYLOAD_MAGIC  0xC51Eu
#define CSIL_FRAME_MAGIC    0xC51Du
#define CSIL_RAW_MAGIC      0xC51Fu
#define CSIL_ESPNOW_HDR_LEN 15   /* Category1+OUI3+Random4+EID1+Len1+OUI3+Type1+Ver1 — 기본 오프셋, 런타임 magic 스캔으로 잠금 */
#define CSIL_NUM_SC         56
#define CSIL_FRAME_LEN      130
#define CSIL_CSI_BUF_MAX    384  /* LLTF128 + HT-LTF128 + STBC128 상한 */

/* ESP-NOW 페이로드 16B (§4.2) — CSI는 프리앰블에서 측정, 페이로드는 식별용 최소화 */
typedef struct __attribute__((packed)) {
    uint16_t magic;     /* 0xC51E */
    uint8_t  tx_idx;    /* 0..2 */
    uint8_t  rsv0;
    uint32_t seq;       /* 시간 정렬·손실 계측 기준 */
    uint8_t  rsv[8];
} csil_payload_t;
_Static_assert(sizeof(csil_payload_t) == 16, "payload must be 16B");

/* 시리얼 CSI 프레임 130B (§4.3 표) */
typedef struct __attribute__((packed)) {
    uint16_t magic;        /* 0xC51D */
    uint8_t  rx_id;
    uint8_t  tx_idx;
    uint32_t seq;          /* 페이로드에서 파싱 */
    uint32_t esp_timer_us; /* u32 — 71.58분 랩, 호스트 unwrap (§5.2-0, v1.3) */
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  len;          /* = 56 */
    uint8_t  boot_id;      /* 부팅마다 +1 — seq/esp_timer 리셋 식별 (§4.3) */
    int8_t   iq[112];      /* I,Q 교호 56쌍 — 디바이스 buf는 [imag,real] → 펌웨어가 스왑 */
    uint16_t crc;          /* CRC16-CCITT-FALSE, 바이트 [0,128) 대상 */
} csil_frame_t;
_Static_assert(sizeof(csil_frame_t) == CSIL_FRAME_LEN, "frame must be 130B");

/* RAW 덤프 프레임 헤더 18B (§16-6 SC 인덱스 실측 지원) — 이후 buf[buf_len] + crc16(헤더+buf) */
typedef struct __attribute__((packed)) {
    uint16_t magic;        /* 0xC51F */
    uint8_t  rx_id;
    uint8_t  tx_idx;
    uint32_t seq;
    uint32_t esp_timer_us;
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  flags;        /* b0=first_word_invalid, b1..2=sig_mode */
    uint8_t  boot_id;
    uint16_t buf_len;
} csil_raw_hdr_t;
_Static_assert(sizeof(csil_raw_hdr_t) == 18, "raw header must be 18B");

uint16_t csil_crc16(const uint8_t *data, size_t len);
