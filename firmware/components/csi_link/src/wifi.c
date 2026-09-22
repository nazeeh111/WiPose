#include "csi_link/wifi.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "esp_event.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_wifi.h"

const uint8_t CSIL_BCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

esp_err_t csil_wifi_start(uint8_t ch)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));
    /* 11b 금지 보강 (§4.1) — esp_wifi_start() 이전에만 호출 가능 */
    ESP_ERROR_CHECK(esp_wifi_config_11b_rate(WIFI_IF_STA, true));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));            /* 타이밍 일관성 */
    ESP_ERROR_CHECK(esp_wifi_set_channel(ch, WIFI_SECOND_CHAN_NONE));
    return ESP_OK;
}

esp_err_t csil_espnow_tx_init(void)
{
    ESP_ERROR_CHECK(esp_now_init());
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, CSIL_BCAST, 6);
    peer.ifidx = WIFI_IF_STA;
    peer.channel = 0; /* 현재 채널 추종 */
    ESP_ERROR_CHECK(esp_now_add_peer(&peer)); /* rate config 전 선등록 필수 (§4.1) */
    esp_now_rate_config_t rc = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI, /* HT 프레임이어야 HT-LTF 56SC 확보 (§4.1) */
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(CSIL_BCAST, &rc));
    return ESP_OK;
}

esp_err_t csil_espnow_rx_init(void)
{
    return esp_now_init();
}

esp_err_t csil_wifi_scan_report(char *out, size_t cap, uint8_t restore_ch)
{
    wifi_scan_config_t sc = {0}; /* 전 채널 액티브 스캔 */
    esp_err_t err = esp_wifi_scan_start(&sc, true);
    if (err != ESP_OK) {
        snprintf(out, cap, "ERR scan %d\n", (int)err);
        return err;
    }
    uint16_t n = 0;
    esp_wifi_scan_get_ap_num(&n);
    if (n > 64)
        n = 64;
    wifi_ap_record_t *recs = calloc(n ? n : 1, sizeof(*recs));
    if (!recs) {
        snprintf(out, cap, "ERR scan oom\n");
        return ESP_ERR_NO_MEM;
    }
    esp_wifi_scan_get_ap_records(&n, recs);

    int cnt[14] = {0};
    int best[14];
    for (int i = 0; i < 14; i++)
        best[i] = -127;
    for (uint16_t i = 0; i < n; i++) {
        uint8_t c = recs[i].primary;
        if (c >= 1 && c <= 13) {
            cnt[c]++;
            if (recs[i].rssi > best[c])
                best[c] = recs[i].rssi;
        }
    }
    free(recs);

    size_t off = 0;
    for (int c = 1; c <= 13 && off < cap; c++)
        off += (size_t)snprintf(out + off, cap - off,
                                "SCAN ch=%d ap=%d best=%d\n", c, cnt[c], best[c]);
    esp_wifi_set_channel(restore_ch, WIFI_SECOND_CHAN_NONE);
    return ESP_OK;
}
