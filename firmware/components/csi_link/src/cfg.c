#include "csi_link/cfg.h"

#include <stdio.h>
#include <string.h>

#include "nvs.h"
#include "nvs_flash.h"

static nvs_handle_t s_nvs;

esp_err_t csil_cfg_init(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK)
        return err;
    return nvs_open("csil", NVS_READWRITE, &s_nvs);
}

uint8_t csil_cfg_get_u8(const char *key, uint8_t def)
{
    uint8_t v = def;
    if (nvs_get_u8(s_nvs, key, &v) != ESP_OK)
        return def;
    return v;
}

void csil_cfg_set_u8(const char *key, uint8_t v)
{
    nvs_set_u8(s_nvs, key, v);
    nvs_commit(s_nvs);
}

uint8_t csil_cfg_next_boot_id(void)
{
    uint32_t cnt = 0;
    nvs_get_u32(s_nvs, "bootcnt", &cnt);
    cnt++;
    nvs_set_u32(s_nvs, "bootcnt", cnt);
    nvs_commit(s_nvs);
    return (uint8_t)(cnt & 0xFF);
}

static void mac_key(int idx, char out[8])
{
    snprintf(out, 8, "txm%d", idx);
}

bool csil_cfg_get_mac(int idx, uint8_t mac[6])
{
    char key[8];
    mac_key(idx, key);
    size_t len = 6;
    return nvs_get_blob(s_nvs, key, mac, &len) == ESP_OK && len == 6;
}

void csil_cfg_set_mac(int idx, const uint8_t mac[6])
{
    char key[8];
    mac_key(idx, key);
    nvs_set_blob(s_nvs, key, mac, 6);
    nvs_commit(s_nvs);
}

void csil_cfg_erase_macs(void)
{
    for (int i = 0; i < 3; i++) {
        char key[8];
        mac_key(i, key);
        nvs_erase_key(s_nvs, key);
    }
    nvs_commit(s_nvs);
}
