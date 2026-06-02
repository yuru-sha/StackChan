/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "face_detect_wakeup.h"

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32S3

#include <board.h>
#include <lvgl_display.h>
#include <hal/board/config.h>
#include <hal/board/hal_bridge.h>
#include <esp_camera.h>
#include <esp_heap_caps.h>
#include <esp_lvgl_port.h>
#include <human_face_detect.hpp>
#include <mooncake_log.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace {

constexpr const char* kTag               = "AI.AGENT.FaceDetect";
constexpr uint32_t kIdlePollIntervalMs   = 1000;
constexpr uint32_t kDetectIntervalMs     = 1500;
constexpr uint32_t kWakeCooldownMs       = 10000;
constexpr uint32_t kUnsupportedBackoffMs = 3000;
constexpr uint32_t kStateLogIntervalMs   = 10000;
constexpr float kWakeScoreThreshold      = 0.97F;
constexpr int kMinFaceSidePx             = 80;
constexpr int kMaxFaceSidePx             = 220;
constexpr int kFaceEdgeMarginPx          = 8;
constexpr int kMaxStableCenterShiftPx    = 32;
constexpr int kMaxStableSideShiftPx      = 35;
constexpr int kRequiredLandmarkValues    = 10;
constexpr int kRequiredConsecutiveHits   = 5;
constexpr int kCoreS3CameraXclkHz        = 10000000;
constexpr const char* kEspCameraVariant  = "esp_camera_rgb565";
constexpr const char* kSwappedVariant    = "esp_camera_rgb565_byteswapped";
constexpr const char* kSwappedDimsVariant = "esp_camera_rgb565_swapped_dims";
constexpr const char* kRotCwVariant      = "esp_camera_rgb565_rot90_cw";
constexpr const char* kRotCcwVariant     = "esp_camera_rgb565_rot90_ccw";
constexpr const char* kRgb888LeVariant   = "esp_camera_rgb888_from_le";
constexpr const char* kRgb888BeVariant   = "esp_camera_rgb888_from_be";
#if CONFIG_CAMERA_PSRAM_DMA
constexpr bool kDefaultCameraPsramDma = true;
#else
constexpr bool kDefaultCameraPsramDma = false;
#endif
#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_RODATA
constexpr const char* kModelStorage = "flash_rodata";
#elif CONFIG_HUMAN_FACE_DETECT_MODEL_IN_FLASH_PARTITION
constexpr const char* kModelStorage = "flash_partition";
#elif CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
constexpr const char* kModelStorage = "sdcard";
#else
constexpr const char* kModelStorage = "unknown";
#endif

std::atomic_bool s_started{false};

class LcdBusQuietGuard {
public:
    LcdBusQuietGuard()
    {
        stopped_ = lvgl_port_stop() == ESP_OK;
        locked_  = lvgl_port_lock(1000);
        if (!locked_) {
            mclog::tagWarn(kTag, "failed to lock LVGL before camera capture");
            return;
        }

        // CoreS3 shares several LCD and camera pins. Give any in-flight LCD DMA transfer time to drain.
        vTaskDelay(pdMS_TO_TICKS(80));
    }

    ~LcdBusQuietGuard()
    {
        if (locked_) {
            lvgl_port_unlock();
        }
        if (stopped_) {
            lvgl_port_resume();
        }
    }

private:
    bool stopped_ = false;
    bool locked_  = false;
};

static void log_camera_heap(const char* phase)
{
    mclog::tagInfo(kTag,
                   "camera heap {}: dma_free={}, dma_largest={}, internal_free={}, internal_largest={}, spiram_free={}, spiram_largest={}",
                   phase,
                   heap_caps_get_free_size(MALLOC_CAP_DMA),
                   heap_caps_get_largest_free_block(MALLOC_CAP_DMA),
                   heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                   heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
                   heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                   heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
}

static camera_config_t make_camera_config(size_t fb_count)
{
    return camera_config_t{
        .pin_pwdn = CAMERA_PIN_PWDN,
        .pin_reset = CAMERA_PIN_RESET,
        .pin_xclk = CAMERA_PIN_XCLK,
        .pin_sccb_sda = -1,
        .pin_sccb_scl = -1,
        .pin_d7 = CAMERA_PIN_D7,
        .pin_d6 = CAMERA_PIN_D6,
        .pin_d5 = CAMERA_PIN_D5,
        .pin_d4 = CAMERA_PIN_D4,
        .pin_d3 = CAMERA_PIN_D3,
        .pin_d2 = CAMERA_PIN_D2,
        .pin_d1 = CAMERA_PIN_D1,
        .pin_d0 = CAMERA_PIN_D0,
        .pin_vsync = CAMERA_PIN_VSYNC,
        .pin_href = CAMERA_PIN_HREF,
        .pin_pclk = CAMERA_PIN_PCLK,
        .xclk_freq_hz = kCoreS3CameraXclkHz,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB565,
        .frame_size = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count = fb_count,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        .sccb_i2c_port = 1,
    };
}

static bool try_init_esp_camera(bool psram_dma, size_t fb_count)
{
    mclog::tagInfo(kTag,
                   "esp_camera_init attempt: psram_dma={}, fb_count={}, fb_location=PSRAM, dma_buffer_max={}, sdkconfig_psram_dma={}",
                   psram_dma, fb_count, CONFIG_CAMERA_DMA_BUFFER_SIZE_MAX, kDefaultCameraPsramDma);
    log_camera_heap("before init");
    esp_camera_set_psram_mode(psram_dma);
    camera_config_t camera_config = make_camera_config(fb_count);
    const esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        mclog::tagError(kTag, "esp_camera_init failed: psram_dma={}, fb_count={}, err=0x{:X}", psram_dma,
                        fb_count, static_cast<int>(err));
        log_camera_heap("after failed init");
        esp_camera_deinit();
        return false;
    }
    log_camera_heap("after init");

    auto sensor = esp_camera_sensor_get();
    if (sensor != nullptr) {
        sensor->set_hmirror(sensor, 0);
    }

    mclog::tagInfo(kTag, "esp_camera initialized: RGB565 QVGA, xclk={}Hz, psram_dma={}, fb_count={}",
                   kCoreS3CameraXclkHz, psram_dma, fb_count);
    return true;
}

static bool init_esp_camera()
{
    if (try_init_esp_camera(false, 1)) {
        return true;
    }

    mclog::tagWarn(kTag, "retry esp_camera_init with PSRAM DMA enabled");
    return try_init_esp_camera(true, 2);
}

struct FaceDetectionSummary {
    float best_score = 0.0F;
    const char* variant = "none";
    int best_box[4] = {0, 0, 0, 0};
    int best_keypoints = 0;
    int best_side = 0;
    const char* reject_reason = "none";
    int accepted_box[4] = {0, 0, 0, 0};
    int accepted_keypoints = 0;
    int center_x = 0;
    int center_y = 0;
    int side = 0;
};

static bool has_confident_face(const std::list<dl::detect::result_t>& results,
                               int frame_width,
                               int frame_height,
                               const char* variant,
                               FaceDetectionSummary& summary);

static void log_raw_face_results(const char* variant, const std::list<dl::detect::result_t>& results)
{
    int index = 0;
    for (const auto& result : results) {
        if (result.box.size() < 4) {
            mclog::tagInfo(kTag, "raw face result {}: variant={}, score={:.2f}, box_values={}, keypoints={}",
                           index, variant, result.score, result.box.size(), result.keypoint.size());
            index++;
            continue;
        }

        if (result.keypoint.size() >= kRequiredLandmarkValues) {
            mclog::tagInfo(kTag,
                           "raw face result {}: variant={}, score={:.2f}, box=[{},{},{},{}], keypoints=[{},{},{},{},{},{},{},{},{},{}]",
                           index, variant, result.score, result.box[0], result.box[1], result.box[2],
                           result.box[3], result.keypoint[0], result.keypoint[1], result.keypoint[2],
                           result.keypoint[3], result.keypoint[4], result.keypoint[5], result.keypoint[6],
                           result.keypoint[7], result.keypoint[8], result.keypoint[9]);
        } else {
            mclog::tagInfo(kTag, "raw face result {}: variant={}, score={:.2f}, box=[{},{},{},{}], keypoints={}",
                           index, variant, result.score, result.box[0], result.box[1], result.box[2],
                           result.box[3], result.keypoint.size());
        }
        index++;
    }
}

static void swap_rgb565_bytes(uint8_t* dst, const uint8_t* src, size_t len)
{
    const size_t even_len = len & ~static_cast<size_t>(1);
    for (size_t i = 0; i < even_len; i += 2) {
        dst[i] = src[i + 1];
        dst[i + 1] = src[i];
    }
    if (even_len != len) {
        dst[even_len] = src[even_len];
    }
}

static void rotate_rgb565_90_cw(uint8_t* dst, const uint8_t* src, uint16_t src_width, uint16_t src_height)
{
    for (uint16_t y = 0; y < src_height; ++y) {
        for (uint16_t x = 0; x < src_width; ++x) {
            const size_t src_index = (static_cast<size_t>(y) * src_width + x) * 2;
            const uint16_t dst_x   = src_height - 1 - y;
            const uint16_t dst_y   = x;
            const size_t dst_index = (static_cast<size_t>(dst_y) * src_height + dst_x) * 2;
            dst[dst_index]         = src[src_index];
            dst[dst_index + 1]     = src[src_index + 1];
        }
    }
}

static void rotate_rgb565_90_ccw(uint8_t* dst, const uint8_t* src, uint16_t src_width, uint16_t src_height)
{
    for (uint16_t y = 0; y < src_height; ++y) {
        for (uint16_t x = 0; x < src_width; ++x) {
            const size_t src_index = (static_cast<size_t>(y) * src_width + x) * 2;
            const uint16_t dst_x   = y;
            const uint16_t dst_y   = src_width - 1 - x;
            const size_t dst_index = (static_cast<size_t>(dst_y) * src_height + dst_x) * 2;
            dst[dst_index]         = src[src_index];
            dst[dst_index + 1]     = src[src_index + 1];
        }
    }
}

static void rgb565_to_rgb888(uint8_t* dst, const uint8_t* src, size_t pixel_count, bool big_endian)
{
    for (size_t i = 0; i < pixel_count; ++i) {
        const uint8_t b0 = src[i * 2];
        const uint8_t b1 = src[i * 2 + 1];
        const uint16_t pixel = big_endian ? (static_cast<uint16_t>(b0) << 8) | b1
                                          : (static_cast<uint16_t>(b1) << 8) | b0;
        const uint8_t r5 = (pixel >> 11) & 0x1F;
        const uint8_t g6 = (pixel >> 5) & 0x3F;
        const uint8_t b5 = pixel & 0x1F;
        dst[i * 3]     = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
        dst[i * 3 + 1] = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
        dst[i * 3 + 2] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
    }
}

static bool run_face_detection(HumanFaceDetect* detect,
                               const char* variant,
                               uint8_t* data,
                               uint16_t width,
                               uint16_t height,
                               FaceDetectionSummary& detection_summary)
{
    dl::image::img_t img = {
        .data = data,
        .width = width,
        .height = height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565,
    };

    auto& results = detect->run(img);
    log_raw_face_results(variant, results);
    const bool has_face = has_confident_face(results, width, height, variant, detection_summary);
    if (!has_face && !results.empty()) {
        mclog::tagInfo(kTag,
                       "ignored face detection: variant={}, count={}, best_score={:.2f}, box=[{},{},{},{}], side={}, keypoints={}, reason={}",
                       detection_summary.variant, results.size(), detection_summary.best_score,
                       detection_summary.best_box[0], detection_summary.best_box[1], detection_summary.best_box[2],
                       detection_summary.best_box[3], detection_summary.best_side, detection_summary.best_keypoints,
                       detection_summary.reject_reason);
    }
    return has_face;
}

static void show_camera_preview_once(const camera_fb_t* frame)
{
    if (frame == nullptr || frame->format != PIXFORMAT_RGB565 || frame->len == 0) {
        return;
    }

    auto display = dynamic_cast<LvglDisplay*>(Board::GetInstance().GetDisplay());
    if (display == nullptr) {
        mclog::tagWarn(kTag, "skip camera preview because display is unavailable");
        return;
    }

    auto data = static_cast<uint8_t*>(heap_caps_malloc(frame->len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (data == nullptr) {
        mclog::tagWarn(kTag, "failed to allocate camera preview frame");
        return;
    }

    std::memcpy(data, frame->buf, frame->len);
    auto image = std::make_unique<LvglAllocatedImage>(data,
                                                      frame->len,
                                                      static_cast<int>(frame->width),
                                                      static_cast<int>(frame->height),
                                                      static_cast<int>(frame->width * 2),
                                                      LV_COLOR_FORMAT_RGB565);
    display->SetPreviewImage(std::move(image));
    mclog::tagInfo(kTag, "showing one camera preview frame on display");
}

static bool has_confident_face(const std::list<dl::detect::result_t>& results,
                               int frame_width,
                               int frame_height,
                               const char* variant,
                               FaceDetectionSummary& summary)
{
    summary = {};
    summary.variant = variant;
    for (const auto& result : results) {
        if (result.score > summary.best_score) {
            summary.best_score = result.score;
            summary.reject_reason = "score_or_keypoints";
            summary.best_keypoints = static_cast<int>(result.keypoint.size());
            if (result.box.size() >= 4) {
                for (int i = 0; i < 4; ++i) {
                    summary.best_box[i] = result.box[i];
                }
                const int width = result.box[2] - result.box[0];
                const int height = result.box[3] - result.box[1];
                summary.best_side = std::max(width, height);
            }
        }
        if (result.score < kWakeScoreThreshold || result.box.size() < 4 ||
            static_cast<int>(result.keypoint.size()) < kRequiredLandmarkValues) {
            continue;
        }

        const bool touches_edge = result.box[0] <= kFaceEdgeMarginPx ||
                                  result.box[1] <= kFaceEdgeMarginPx ||
                                  result.box[2] >= frame_width - kFaceEdgeMarginPx ||
                                  result.box[3] >= frame_height - kFaceEdgeMarginPx;
        if (touches_edge) {
            if (result.score >= summary.best_score) {
                summary.reject_reason = "edge";
            }
            continue;
        }

        const int width = result.box[2] - result.box[0];
        const int height = result.box[3] - result.box[1];
        const int side = std::max(width, height);
        if (side < kMinFaceSidePx || side > kMaxFaceSidePx) {
            if (result.score >= summary.best_score) {
                summary.reject_reason = "size";
            }
            continue;
        }

        const int center_x = (result.box[0] + result.box[2]) / 2;
        const int center_y = (result.box[1] + result.box[3]) / 2;
        const bool near_center = center_x >= frame_width / 5 &&
                                 center_x <= (frame_width * 4) / 5 &&
                                 center_y >= frame_height / 6 &&
                                 center_y <= (frame_height * 5) / 6;
        if (!near_center) {
            if (result.score >= summary.best_score) {
                summary.reject_reason = "off_center";
            }
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            summary.accepted_box[i] = result.box[i];
        }
        summary.accepted_keypoints = static_cast<int>(result.keypoint.size());
        summary.center_x = center_x;
        summary.center_y = center_y;
        summary.side = side;
        return true;
    }

    return false;
}

static bool is_stable_face_hit(const FaceDetectionSummary& current, const FaceDetectionSummary& previous)
{
    if (previous.side == 0) {
        return true;
    }

    return std::abs(current.center_x - previous.center_x) <= kMaxStableCenterShiftPx &&
           std::abs(current.center_y - previous.center_y) <= kMaxStableCenterShiftPx &&
           std::abs(current.side - previous.side) <= kMaxStableSideShiftPx;
}

static void face_detect_wakeup_task(void*)
{
    mclog::tagInfo(kTag, "human_face_detect model storage={}, location={}", kModelStorage,
                   CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION);
    TickType_t last_wake_tick = 0;
    TickType_t last_state_log_tick = 0;
    int consecutive_hits = 0;
    bool logged_frame_info = false;
    bool previewed_frame = false;
    bool last_ready = false;
    bool last_idle = false;
    bool has_logged_state = false;
    FaceDetectionSummary previous_detection;

    if (!init_esp_camera()) {
        mclog::tagError(kTag, "face detect wakeup disabled because camera init failed");
        vTaskDelete(nullptr);
        return;
    }

    auto detect = new HumanFaceDetect();
    mclog::tagInfo(kTag, "face detect wakeup task started");
    mclog::tagInfo(kTag, "detector uses human_face_detect default score thresholds; wake score threshold={:.2f}",
                   kWakeScoreThreshold);

    while (true) {
        const bool is_ready = hal_bridge::is_xiaozhi_ready();
        const bool is_idle = hal_bridge::is_xiaozhi_idle();
        if (!is_ready || !is_idle) {
            const TickType_t now = xTaskGetTickCount();
            const bool state_changed = !has_logged_state || is_ready != last_ready || is_idle != last_idle;
            const bool should_log_periodically =
                last_state_log_tick == 0 || (now - last_state_log_tick) >= pdMS_TO_TICKS(kStateLogIntervalMs);
            if (state_changed || should_log_periodically) {
                mclog::tagInfo(kTag, "waiting for idle conversation: ready={}, idle={}", is_ready, is_idle);
                last_ready = is_ready;
                last_idle = is_idle;
                has_logged_state = true;
                last_state_log_tick = now;
            }
            vTaskDelay(pdMS_TO_TICKS(kIdlePollIntervalMs));
            continue;
        }

        camera_fb_t* frame = nullptr;
        {
            LcdBusQuietGuard lcd_bus_quiet;
            frame = esp_camera_fb_get();
        }
        if (frame == nullptr) {
            mclog::tagWarn(kTag, "esp_camera_fb_get failed");
            vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
            continue;
        }

        if (!logged_frame_info) {
            mclog::tagInfo(kTag, "camera frame: {}x{}, len={}, format={}", frame->width, frame->height, frame->len,
                           static_cast<int>(frame->format));
            logged_frame_info = true;
        }

        if (!previewed_frame) {
            show_camera_preview_once(frame);
            previewed_frame = true;
        }

        FaceDetectionSummary detection_summary;
        bool has_face = run_face_detection(detect,
                                           kEspCameraVariant,
                                           frame->buf,
                                           static_cast<uint16_t>(frame->width),
                                           static_cast<uint16_t>(frame->height),
                                           detection_summary);

        uint8_t* converted_frame = nullptr;
        if (!has_face) {
            converted_frame = static_cast<uint8_t*>(heap_caps_malloc(frame->len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
            if (converted_frame == nullptr) {
                mclog::tagWarn(kTag, "failed to allocate converted RGB565 frame");
            } else {
                swap_rgb565_bytes(converted_frame, frame->buf, frame->len);
                FaceDetectionSummary swapped_summary;
                has_face = run_face_detection(detect,
                                              kSwappedVariant,
                                              converted_frame,
                                              static_cast<uint16_t>(frame->width),
                                              static_cast<uint16_t>(frame->height),
                                              swapped_summary);
                if (has_face) {
                    detection_summary = swapped_summary;
                }
            }
        }

        if (!has_face && converted_frame != nullptr && frame->width == 320 && frame->height == 240) {
            FaceDetectionSummary swapped_dims_summary;
            has_face = run_face_detection(detect,
                                          kSwappedDimsVariant,
                                          frame->buf,
                                          static_cast<uint16_t>(frame->height),
                                          static_cast<uint16_t>(frame->width),
                                          swapped_dims_summary);
            if (has_face) {
                detection_summary = swapped_dims_summary;
            }
        }

        if (!has_face && converted_frame != nullptr && frame->width == 320 && frame->height == 240) {
            rotate_rgb565_90_cw(converted_frame,
                                frame->buf,
                                static_cast<uint16_t>(frame->width),
                                static_cast<uint16_t>(frame->height));
            FaceDetectionSummary rot_cw_summary;
            has_face = run_face_detection(detect,
                                          kRotCwVariant,
                                          converted_frame,
                                          static_cast<uint16_t>(frame->height),
                                          static_cast<uint16_t>(frame->width),
                                          rot_cw_summary);
            if (has_face) {
                detection_summary = rot_cw_summary;
            }
        }

        if (!has_face && converted_frame != nullptr && frame->width == 320 && frame->height == 240) {
            rotate_rgb565_90_ccw(converted_frame,
                                 frame->buf,
                                 static_cast<uint16_t>(frame->width),
                                 static_cast<uint16_t>(frame->height));
            FaceDetectionSummary rot_ccw_summary;
            has_face = run_face_detection(detect,
                                          kRotCcwVariant,
                                          converted_frame,
                                          static_cast<uint16_t>(frame->height),
                                          static_cast<uint16_t>(frame->width),
                                          rot_ccw_summary);
            if (has_face) {
                detection_summary = rot_ccw_summary;
            }
        }

        uint8_t* rgb888_frame = nullptr;
        if (!has_face) {
            const size_t rgb888_len = static_cast<size_t>(frame->width) * frame->height * 3;
            rgb888_frame = static_cast<uint8_t*>(heap_caps_malloc(rgb888_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
            if (rgb888_frame == nullptr) {
                mclog::tagWarn(kTag, "failed to allocate RGB888 frame");
            } else {
                rgb565_to_rgb888(rgb888_frame, frame->buf, static_cast<size_t>(frame->width) * frame->height, false);
                dl::image::img_t img = {
                    .data = rgb888_frame,
                    .width = static_cast<uint16_t>(frame->width),
                    .height = static_cast<uint16_t>(frame->height),
                    .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888,
                };

                auto& results = detect->run(img);
                log_raw_face_results(kRgb888LeVariant, results);
                FaceDetectionSummary rgb888_summary;
                has_face = has_confident_face(results,
                                              static_cast<int>(frame->width),
                                              static_cast<int>(frame->height),
                                              kRgb888LeVariant,
                                              rgb888_summary);
                if (!has_face && !results.empty()) {
                    mclog::tagInfo(kTag,
                                   "ignored face detection: variant={}, count={}, best_score={:.2f}, box=[{},{},{},{}], side={}, keypoints={}, reason={}",
                                   rgb888_summary.variant, results.size(), rgb888_summary.best_score,
                                   rgb888_summary.best_box[0], rgb888_summary.best_box[1], rgb888_summary.best_box[2],
                                   rgb888_summary.best_box[3], rgb888_summary.best_side, rgb888_summary.best_keypoints,
                                   rgb888_summary.reject_reason);
                }
                if (has_face) {
                    detection_summary = rgb888_summary;
                }
            }
        }

        if (!has_face && rgb888_frame != nullptr) {
            rgb565_to_rgb888(rgb888_frame, frame->buf, static_cast<size_t>(frame->width) * frame->height, true);
            dl::image::img_t img = {
                .data = rgb888_frame,
                .width = static_cast<uint16_t>(frame->width),
                .height = static_cast<uint16_t>(frame->height),
                .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888,
            };

            auto& results = detect->run(img);
            log_raw_face_results(kRgb888BeVariant, results);
            FaceDetectionSummary rgb888_summary;
            has_face = has_confident_face(results,
                                          static_cast<int>(frame->width),
                                          static_cast<int>(frame->height),
                                          kRgb888BeVariant,
                                          rgb888_summary);
            if (!has_face && !results.empty()) {
                mclog::tagInfo(kTag,
                               "ignored face detection: variant={}, count={}, best_score={:.2f}, box=[{},{},{},{}], side={}, keypoints={}, reason={}",
                               rgb888_summary.variant, results.size(), rgb888_summary.best_score,
                               rgb888_summary.best_box[0], rgb888_summary.best_box[1], rgb888_summary.best_box[2],
                               rgb888_summary.best_box[3], rgb888_summary.best_side, rgb888_summary.best_keypoints,
                               rgb888_summary.reject_reason);
            }
            if (has_face) {
                detection_summary = rgb888_summary;
            }
        }

        if (has_face) {
            if (is_stable_face_hit(detection_summary, previous_detection)) {
                consecutive_hits++;
            } else {
                consecutive_hits = 1;
                mclog::tagInfo(kTag, "reset face hits because bbox moved too much");
            }
            previous_detection = detection_summary;
            mclog::tagInfo(kTag,
                           "accepted face hit {}/{}: variant={}, score={:.2f}, box=[{},{},{},{}], center=[{},{}], side={}, keypoints={}",
                           consecutive_hits, kRequiredConsecutiveHits, detection_summary.variant,
                           detection_summary.best_score, detection_summary.accepted_box[0], detection_summary.accepted_box[1],
                           detection_summary.accepted_box[2], detection_summary.accepted_box[3],
                           detection_summary.center_x, detection_summary.center_y, detection_summary.side,
                           detection_summary.accepted_keypoints);
        } else {
            consecutive_hits = 0;
            previous_detection = {};
        }

        if (consecutive_hits >= kRequiredConsecutiveHits) {
            const TickType_t now = xTaskGetTickCount();
            if (last_wake_tick == 0 ||
                (now - last_wake_tick) >= pdMS_TO_TICKS(kWakeCooldownMs)) {
                last_wake_tick = now;
                consecutive_hits = 0;
                mclog::tagInfo(kTag, "face detected, turning conversation on");
                if (!hal_bridge::start_xiaozhi_chat_from_idle()) {
                    mclog::tagInfo(kTag, "skip face wakeup because Xiaozhi is no longer idle");
                }
            }
        }

        esp_camera_fb_return(frame);
        if (converted_frame != nullptr) {
            heap_caps_free(converted_frame);
        }
        if (rgb888_frame != nullptr) {
            heap_caps_free(rgb888_frame);
        }
        vTaskDelay(pdMS_TO_TICKS(kDetectIntervalMs));
    }
}

}  // namespace

namespace ai_agent {

void start_face_detect_wakeup()
{
    bool expected = false;
    if (!s_started.compare_exchange_strong(expected, true)) {
        return;
    }

    BaseType_t ok = xTaskCreatePinnedToCore(face_detect_wakeup_task, "face_detect_wakeup", 8192, nullptr, 2, nullptr, 1);
    if (ok != pdPASS) {
        s_started.store(false);
        mclog::tagError(kTag, "failed to start face detect wakeup task");
    }
}

}  // namespace ai_agent

#else

namespace ai_agent {

void start_face_detect_wakeup() {}

}  // namespace ai_agent

#endif  // CONFIG_IDF_TARGET_ESP32S3
