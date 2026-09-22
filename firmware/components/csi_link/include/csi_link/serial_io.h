/* 시리얼 I/O 추상화 — UART0 / USB-Serial-JTAG 백엔드 (Kconfig 선택).
 * 바이너리 프레임과 텍스트 응답이 같은 포트를 쓰므로 write는 호출 단위 원자성 보장.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

void csil_serial_init(void);
/* 전량 블로킹 송신 (드라이버 TX 링버퍼 경유, 호출 단위 원자적) */
void csil_serial_write(const void *buf, size_t len);
/* timeout_ms 내 최대 max 바이트 수신, 반환 = 읽은 바이트 수 (0 = 타임아웃) */
int csil_serial_read(uint8_t *buf, size_t max, uint32_t timeout_ms);
