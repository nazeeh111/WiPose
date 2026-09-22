/* 라인 콘솔 — 부팅 = 텍스트 모드, START 후 = 바이너리 모드 (로그 OFF).
 * 응답은 반드시 csil_reply/csil_serial_write 경유 (printf 금지 — 프레임 오염 방지).
 */
#pragma once
#include <stdbool.h>

typedef void (*csil_cmd_handler_t)(const char *line);

/* 블로킹 명령 루프 — app_main에서 호출 (복귀하지 않음) */
void csil_console_run(csil_cmd_handler_t handler);

/* "KEY=value" 정수 인자 파싱 (예: csil_cmd_arg_int("START rate=100", "rate", 100)) */
int csil_cmd_arg_int(const char *line, const char *key, int def);

/* printf형 텍스트 응답 (단일 write — 원자적). fmt에 \n 포함할 것 */
void csil_reply(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* true: ESP_LOG 전부 억제 (바이너리 모드), false: 기본 레벨 복원 */
void csil_set_binary(bool on);
