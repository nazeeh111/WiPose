#pragma once
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

extern const uint8_t CSIL_BCAST[6]; /* FF:FF:FF:FF:FF:FF */

/* STA 모드 브링업: PS off, 11b rate 비활성(DSSS는 OFDM CSI 없음), 채널 고정 */
esp_err_t csil_wifi_start(uint8_t ch);

/* 순서 엄수: esp_now_init → 브로드캐스트 피어 add_peer 선등록 →
 * esp_now_set_peer_rate_config(HT20/MCS0_LGI). TX 전용. */
esp_err_t csil_espnow_tx_init(void);

/* RX 측 ESP-NOW 수신 경로 활성 (CSI payload 파싱용 — recv 콜백은 불요) */
esp_err_t csil_espnow_rx_init(void);

/* 정지 상태 전용: 전 채널 스캔 → "SCAN ch=n ap=k best=r" 라인들 작성 후 채널 원복 */
esp_err_t csil_wifi_scan_report(char *out, size_t cap, uint8_t restore_ch);
