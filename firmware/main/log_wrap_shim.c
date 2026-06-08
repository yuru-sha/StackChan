#include <stdarg.h>

#include "esp_log.h"

void __real_esp_log_writev(esp_log_level_t level, const char* tag, const char* format, va_list args);

void __attribute__((weak)) __wrap_esp_log_writev(esp_log_level_t level,
                                                 const char* tag,
                                                 const char* format,
                                                 va_list args)
{
    __real_esp_log_writev(level, tag, format, args);
}

void __attribute__((weak)) __wrap_esp_log_write(esp_log_level_t level,
                                                const char* tag,
                                                const char* format,
                                                ...)
{
    va_list args;
    va_start(args, format);
    __real_esp_log_writev(level, tag, format, args);
    va_end(args);
}
