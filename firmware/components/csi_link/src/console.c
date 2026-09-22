#include "csi_link/console.h"
#include "csi_link/serial_io.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "esp_log.h"
#include "sdkconfig.h"

void csil_console_run(csil_cmd_handler_t handler)
{
    static char line[160];
    size_t n = 0;
    uint8_t ch;
    for (;;) {
        if (csil_serial_read(&ch, 1, 100) <= 0)
            continue;
        if (ch == '\r')
            continue;
        if (ch == '\n') {
            line[n] = '\0';
            if (n)
                handler(line);
            n = 0;
            continue;
        }
        if (n < sizeof line - 1)
            line[n++] = (char)ch;
        else
            n = 0; /* 과대 라인 폐기 */
    }
}

int csil_cmd_arg_int(const char *line, const char *key, int def)
{
    size_t klen = strlen(key);
    for (const char *p = line; (p = strstr(p, key)) != NULL; p += klen) {
        bool at_start = (p == line) || isspace((unsigned char)p[-1]);
        if (at_start && p[klen] == '=')
            return atoi(p + klen + 1);
    }
    return def;
}

void csil_reply(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0)
        return;
    if ((size_t)n >= sizeof buf)
        n = sizeof buf - 1;
    csil_serial_write(buf, (size_t)n);
}

void csil_set_binary(bool on)
{
    esp_log_level_set("*", on ? ESP_LOG_NONE : (esp_log_level_t)CONFIG_LOG_DEFAULT_LEVEL);
}
