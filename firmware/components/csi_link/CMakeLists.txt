idf_component_register(
    SRCS "src/crc16.c"
         "src/serial_uart0.c"
         "src/serial_usb_jtag.c"
         "src/console.c"
         "src/cfg.c"
         "src/wifi.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_wifi esp_timer nvs_flash esp_event esp_netif vfs
)
