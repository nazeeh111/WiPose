/* NVS 영속 설정 — 네임스페이스 "csil".
 * 키: idx(보드 인덱스 u8), ch(채널 u8), bootcnt(u32), txm0..2(MAC blob 6B).
 */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

esp_err_t csil_cfg_init(void);

uint8_t csil_cfg_get_u8(const char *key, uint8_t def);
void csil_cfg_set_u8(const char *key, uint8_t v);

/* 부팅 카운터 ++ 후 하위 8비트 반환 — 프레임 boot_id (§4.3) */
uint8_t csil_cfg_next_boot_id(void);

/* RX의 TX MAC 핀 영속화 (idx 0..2). get은 존재 시 true */
bool csil_cfg_get_mac(int idx, uint8_t mac[6]);
void csil_cfg_set_mac(int idx, const uint8_t mac[6]);
void csil_cfg_erase_macs(void);
