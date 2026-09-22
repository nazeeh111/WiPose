/* TX 비컨 (설계 §4.2): START 후 esp_timer 주기(기본 10ms=100pps)로 ESP-NOW 브로드캐스트.
 * 페이로드 16B {magic, tx_idx, seq} — CSI는 프리앰블에서 측정되므로 식별용 최소화.
 * 시리얼 명령: HELLO MAC SET_IDX SET_CH START[rate=N] STOP STATUS SCAN
 */
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "esp_mac.h"
#include "esp_now.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "sdkconfig.h"

#include "csi_link/cfg.h"
#include "csi_link/console.h"
#include "csi_link/serial_io.h"
#include "csi_link/wifi.h"
#include "csi_link/wire.h"

#define FW_VER "m0.1"

static esp_timer_handle_t s_tick;
static volatile bool s_running;
static uint32_t s_seq, s_sent_ok, s_sent_fail;
static uint32_t s_rate = 100;
static uint8_t s_idx = 0xFF, s_ch, s_boot_id;

/* 평균 주기 ±15% 균일 지터로 재예약 — 3TX의 등주기 위상 고정이 만드는 주기적
 * 충돌 폭풍 차단 (M0 실측: 위상 정렬 구간에서 링크 손실 5~8%).
 * 호스트가 공통 100Hz 그리드로 리샘플 하므로 지터는 무해. */
static void schedule_next(void)
{
    uint32_t period = 1000000UL / s_rate;
    uint32_t span = period * 30 / 100;
    int32_t jit = (int32_t)(esp_random() % (span + 1)) - (int32_t)(span / 2);
    esp_timer_start_once(s_tick, (uint64_t)((int64_t)period + jit));
}

/* 브로드캐스트는 ACK가 없어 send 콜백 status가 무의미 — esp_now_send 반환만 계수.
 * 손실 계측의 기준은 어디까지나 RX측 seq 갭. */
static void tick_cb(void *arg)
{
    if (!s_running)
        return;
    csil_payload_t p = {
        .magic = CSIL_PAYLOAD_MAGIC,
        .tx_idx = s_idx,
        .seq = s_seq++,
    };
    if (esp_now_send(CSIL_BCAST, (const uint8_t *)&p, sizeof p) == ESP_OK)
        s_sent_ok++;
    else
        s_sent_fail++;
    schedule_next();
}

static void banner(void)
{
    uint8_t m[6];
    esp_read_mac(m, ESP_MAC_WIFI_STA);
    csil_reply("BOOT role=tx idx=%d mac=%02X:%02X:%02X:%02X:%02X:%02X ch=%u boot_id=%u fw=%s\n",
               s_idx == 0xFF ? -1 : (int)s_idx, m[0], m[1], m[2], m[3], m[4], m[5],
               (unsigned)s_ch, (unsigned)s_boot_id, FW_VER);
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
        s_idx = (uint8_t)v;
        csil_cfg_set_u8("idx", s_idx);
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
        if (!s_running)
            esp_wifi_set_channel(s_ch, WIFI_SECOND_CHAN_NONE);
        csil_reply("OK SET_CH %d\n", v);
        return;
    }
    if (!strncmp(line, "START", 5)) {
        if (s_idx == 0xFF) {
            csil_reply("ERR idx unset (SET_IDX first)\n");
            return;
        }
        int rate = csil_cmd_arg_int(line, "rate", 100);
        if (rate < 1 || rate > 500) {
            csil_reply("ERR rate range 1..500\n");
            return;
        }
        if (s_running)
            esp_timer_stop(s_tick);
        s_rate = (uint32_t)rate;
        s_seq = s_sent_ok = s_sent_fail = 0;  /* START마다 seq 리셋 — 호스트는 reset 이벤트로 처리 */
        s_running = true;
        schedule_next();
        csil_reply("OK START rate=%d\n", rate);
        return;
    }
    if (!strcmp(line, "STOP")) {
        if (s_running) {
            s_running = false;   /* 콜백 재예약 차단 후 정지 */
            esp_timer_stop(s_tick);
        }
        csil_reply("OK STOP sent=%" PRIu32 " fail=%" PRIu32 "\n", s_sent_ok, s_sent_fail);
        return;
    }
    if (!strcmp(line, "STATUS")) {
        csil_reply("STATUS role=tx idx=%d ch=%u running=%d rate=%" PRIu32
                   " seq=%" PRIu32 " ok=%" PRIu32 " fail=%" PRIu32 " boot_id=%u\n",
                   s_idx == 0xFF ? -1 : (int)s_idx, (unsigned)s_ch, (int)s_running,
                   s_rate, s_seq, s_sent_ok, s_sent_fail, (unsigned)s_boot_id);
        return;
    }
    if (!strcmp(line, "SCAN")) {
        if (s_running) {
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
    s_idx = csil_cfg_get_u8("idx", 0xFF);
    s_ch = csil_cfg_get_u8("ch", CONFIG_CSI_LINK_DEFAULT_CHANNEL);

    csil_serial_init();
    banner();

    ESP_ERROR_CHECK(csil_wifi_start(s_ch));
    ESP_ERROR_CHECK(csil_espnow_tx_init());

    const esp_timer_create_args_t targs = {.callback = tick_cb, .name = "txbeacon"};
    ESP_ERROR_CHECK(esp_timer_create(&targs, &s_tick));

    csil_console_run(handle); /* 복귀하지 않음 */
}
