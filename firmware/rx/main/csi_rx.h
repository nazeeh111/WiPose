/* CSI 수신 모듈 — 콜백(WiFi 태스크)에서 페이로드 오프셋 잠금·MAC 핀·큐 복사만 수행.
 * 프레임 구성/시리얼 송출/NVS 기록은 프레이머 태스크 몫 (콜백 최소화).
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_err.h"

#include "csi_link/wire.h"

typedef struct {
    uint32_t seq;
    uint32_t t_us;      /* (u32)esp_timer_get_time() — 호스트 unwrap */
    uint8_t  tx_idx;
    int8_t   rssi;
    int8_t   noise;
    uint8_t  flags;     /* b0=first_word_invalid, b1..2=sig_mode */
    uint16_t buf_len;
    int8_t   buf[CSIL_CSI_BUF_MAX];
} csi_item_t;

typedef struct {
    uint32_t cb_total, magic_reject, mac_reject, q_drop;
    uint32_t per_tx[3];
    uint32_t pay_scans;
    int16_t  pay_off;   /* -1 = 미잠금 (magic 스캔) */
} csi_rx_stats_t;

esp_err_t csi_rx_init(QueueHandle_t q);
void csi_rx_get_stats(csi_rx_stats_t *out);
bool csi_rx_get_pinned(int idx, uint8_t mac[6]);
void csi_rx_clear_macs(void);   /* RAM+NVS 핀 삭제 */
void csi_rx_persist_macs(void); /* dirty 핀을 NVS 기록 — 프레이머 태스크 컨텍스트에서 호출 */
