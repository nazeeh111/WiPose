/* RX (설계 §4.3): CSI 콜백 → 큐 → 프레이머 태스크 → 시리얼 921600.
 * STREAM 모드 = 130B CSI 프레임, RAW 모드 = 전체 CSI buf 덤프 (§16-6 SC 표 확정용).
 * 시리얼 명령: HELLO MAC SET_IDX SET_CH START STOP STATUS RAW n CLEAR_MACS SCAN
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_mac.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "csi_link/cfg.h"
#include "csi_link/console.h"
#include "csi_link/serial_io.h"
#include "csi_link/wifi.h"
#include "csi_link/wire.h"

#include "csi_rx.h"
#include "sc_table.h"

#define FW_VER "m0.1"
#define QUEUE_DEPTH 64

typedef enum { M_IDLE, M_STREAM, M_RAW } run_mode_t;

static QueueHandle_t s_q;
static volatile run_mode_t s_mode = M_IDLE;
static volatile uint32_t s_raw_left;
static uint8_t s_rx_id = 0xFF, s_ch, s_boot_id;
static uint32_t s_framed, s_not_ht, s_short_buf;

static void framer_task(void *arg)
{
    csi_item_t it;
    for (;;) {
        if (xQueueReceive(s_q, &it, pdMS_TO_TICKS(100)) != pdTRUE) {
            csi_rx_persist_macs(); /* 한가할 때 MAC 핀 영속화 */
            continue;
        }
        if (s_mode == M_STREAM) {
            if (((it.flags >> 1) & 3u) != 1u) { /* HT 아님 — 11b/g CSI는 56SC 불가 */
                s_not_ht++;
                continue;
            }
            const uint8_t *tab;
            if (it.buf_len >= 256)
                tab = SC_WORD_HTLTF;
            else if (it.buf_len >= 128)
                tab = SC_WORD_SINGLE;
            else {
                s_short_buf++;
                continue;
            }
            csil_frame_t f = {
                .magic = CSIL_FRAME_MAGIC,
                .rx_id = s_rx_id,
                .tx_idx = it.tx_idx,
                .seq = it.seq,
                .esp_timer_us = it.t_us,
                .rssi = it.rssi,
                .noise_floor = it.noise,
                .len = CSIL_NUM_SC,
                .boot_id = s_boot_id,
            };
            for (int k = 0; k < CSIL_NUM_SC; k++) {
                int w = tab[k];
                f.iq[2 * k]     = it.buf[2 * w + 1]; /* I = real — 디바이스 [imag,real] 스왑 */
                f.iq[2 * k + 1] = it.buf[2 * w];     /* Q = imag */
            }
            f.crc = csil_crc16((const uint8_t *)&f, 128);
            csil_serial_write(&f, sizeof f);
            s_framed++;
        } else if (s_mode == M_RAW && s_raw_left > 0) {
            static uint8_t tmp[sizeof(csil_raw_hdr_t) + CSIL_CSI_BUF_MAX + 2];
            csil_raw_hdr_t h = {
                .magic = CSIL_RAW_MAGIC,
                .rx_id = s_rx_id,
                .tx_idx = it.tx_idx,
                .seq = it.seq,
                .esp_timer_us = it.t_us,
                .rssi = it.rssi,
                .noise_floor = it.noise,
                .flags = it.flags,
                .boot_id = s_boot_id,
                .buf_len = it.buf_len,
            };
            memcpy(tmp, &h, sizeof h);
            memcpy(tmp + sizeof h, it.buf, it.buf_len);
            uint16_t crc = csil_crc16(tmp, sizeof h + it.buf_len);
            memcpy(tmp + sizeof h + it.buf_len, &crc, 2);
            csil_serial_write(tmp, sizeof h + it.buf_len + 2);
            if (--s_raw_left == 0) {
                s_mode = M_IDLE;
                csil_set_binary(false);
                csil_reply("OK RAW done\n");
            }
        }
        /* M_IDLE: 큐 드레인만 */
    }
}

static void banner(void)
{
    uint8_t m[6];
    esp_read_mac(m, ESP_MAC_WIFI_STA);
    csil_reply("BOOT role=rx idx=%d mac=%02X:%02X:%02X:%02X:%02X:%02X ch=%u boot_id=%u fw=%s\n",
               s_rx_id == 0xFF ? -1 : (int)s_rx_id, m[0], m[1], m[2], m[3], m[4], m[5],
               (unsigned)s_ch, (unsigned)s_boot_id, FW_VER);
}

static void reply_status(void)
{
    csi_rx_stats_t st;
    csi_rx_get_stats(&st);
    char macs[3][18];
    for (int i = 0; i < 3; i++) {
        uint8_t m[6];
        if (csi_rx_get_pinned(i, m))
            snprintf(macs[i], sizeof macs[i], "%02X:%02X:%02X:%02X:%02X:%02X",
                     m[0], m[1], m[2], m[3], m[4], m[5]);
        else
            snprintf(macs[i], sizeof macs[i], "-");
    }
    csil_reply("STATUS role=rx idx=%d ch=%u mode=%d framed=%" PRIu32 " cb=%" PRIu32
               " magic_rej=%" PRIu32 " mac_rej=%" PRIu32 " qdrop=%" PRIu32
               " not_ht=%" PRIu32 " short=%" PRIu32 " per_tx=%" PRIu32 "/%" PRIu32 "/%" PRIu32
               " pay_off=%d scans=%" PRIu32 " m0=%s m1=%s m2=%s boot_id=%u\n",
               s_rx_id == 0xFF ? -1 : (int)s_rx_id, (unsigned)s_ch, (int)s_mode,
               s_framed, st.cb_total, st.magic_reject, st.mac_reject, st.q_drop,
               s_not_ht, s_short_buf, st.per_tx[0], st.per_tx[1], st.per_tx[2],
               (int)st.pay_off, st.pay_scans, macs[0], macs[1], macs[2],
               (unsigned)s_boot_id);
}

static void handle(const char *line)
{
    if (!strcmp(line, "HELLO")) {
        banner();
        return;
    }
    if (!strcmp(line, "MAC")) {
        uint8_t m[6];
        esp_read_mac(m, ESP_MAC_WIFI_STA);
        csil_reply("MAC %02X:%02X:%02X:%02X:%02X:%02X\n", m[0], m[1], m[2], m[3], m[4], m[5]);
        return;
    }
    if (!strncmp(line, "SET_IDX", 7)) {
        int v = atoi(line + 7);
        if (v < 0 || v > 2) {
            csil_reply("ERR idx range 0..2\n");
            return;
        }
        s_rx_id = (uint8_t)v;
        csil_cfg_set_u8("idx", s_rx_id);
        csil_reply("OK SET_IDX %d\n", v);
        return;
    }
    if (!strncmp(line, "SET_CH", 6)) {
        int v = atoi(line + 6);
        if (v < 1 || v > 13) {
            csil_reply("ERR ch range 1..13\n");
            return;
        }
        s_ch = (uint8_t)v;
        csil_cfg_set_u8("ch", s_ch);
        if (s_mode == M_IDLE)
            esp_wifi_set_channel(s_ch, WIFI_SECOND_CHAN_NONE);
        csil_reply("OK SET_CH %d\n", v);
        return;
    }
    if (!strncmp(line, "START", 5)) {
        if (s_rx_id == 0xFF) {
            csil_reply("ERR idx unset (SET_IDX first)\n");
            return;
        }
        csil_reply("OK START\n");
        csil_set_binary(true); /* 응답 후 바이너리 모드 — 이후 프레임만 송출 */
        s_mode = M_STREAM;
        return;
    }
    if (!strcmp(line, "STOP")) {
        s_mode = M_IDLE;
        s_raw_left = 0;
        csil_set_binary(false);
        reply_status();
        return;
    }
    if (!strcmp(line, "STATUS")) {
        reply_status();
        return;
    }
    if (!strncmp(line, "RAW", 3)) {
        if (s_rx_id == 0xFF) {
            csil_reply("ERR idx unset (SET_IDX first)\n");
            return;
        }
        int n = atoi(line + 3);
        if (n <= 0)
            n = 100;
        if (n > 5000)
            n = 5000;
        csil_reply("OK RAW n=%d\n", n);
        csil_set_binary(true);
        s_raw_left = (uint32_t)n;
        s_mode = M_RAW;
        return;
    }
    if (!strcmp(line, "CLEAR_MACS")) {
        csi_rx_clear_macs();
        csil_reply("OK CLEAR_MACS\n");
        return;
    }
    if (!strcmp(line, "SCAN")) {
        if (s_mode != M_IDLE) {
            csil_reply("ERR scan while running\n");
            return;
        }
        static char buf[512];
        csil_wifi_scan_report(buf, sizeof buf, s_ch);
        csil_serial_write(buf, strlen(buf));
        return;
    }
    csil_reply("ERR unknown cmd\n");
}

void app_main(void)
{
    ESP_ERROR_CHECK(csil_cfg_init());
    s_boot_id = csil_cfg_next_boot_id();
    s_rx_id = csil_cfg_get_u8("idx", 0xFF);
    s_ch = csil_cfg_get_u8("ch", CONFIG_CSI_LINK_DEFAULT_CHANNEL);

    csil_serial_init();
    banner();

    ESP_ERROR_CHECK(csil_wifi_start(s_ch));

    s_q = xQueueCreate(QUEUE_DEPTH, sizeof(csi_item_t));
    configASSERT(s_q);
    ESP_ERROR_CHECK(csi_rx_init(s_q));

    BaseType_t ok = xTaskCreate(framer_task, "framer", 4096, NULL, 5, NULL);
    configASSERT(ok == pdPASS);

    csil_console_run(handle); /* 복귀하지 않음 */
}
