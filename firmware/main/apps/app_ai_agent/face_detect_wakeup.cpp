/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "face_detect_wakeup.h"

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32S3

#include <hal/board/hal_bridge.h>
#include <human_face_detect.hpp>
#include <mooncake_log.h>

#include <algorithm>
#include <atomic>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <linux/videodev2.h>
#include <esp_heap_caps.h>

namespace {

constexpr const char* kTag               = "AI.AGENT.FaceDetect";
constexpr uint32_t kIdlePollIntervalMs   = 1000;
constexpr uint32_t kDetectIntervalMs     = 1500;
constexpr uint32_t kWakeCooldownMs       = 10000;
constexpr uint32_t kUnsupportedBackoffMs = 3000;
constexpr float kMsrScoreThreshold       = 0.65F;
constexpr float kMnpScoreThreshold       = 0.75F;
constexpr int kMinFaceSidePx             = 32;
constexpr int kMaxFaceSidePx             = 220;
constexpr int kRequiredLandmarkValues    = 10;
constexpr int kRequiredConsecutiveHits   = 2;
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

static inline uint8_t clamp_u8(int value)
{
    return static_cast<uint8_t>(std::min(255, std::max(0, value)));
}

static inline uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b)
{
    return static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

static void yuv_to_rgb(uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b)
{
    int c = static_cast<int>(y) - 16;
    int d = static_cast<int>(u) - 128;
    int e = static_cast<int>(v) - 128;

    r = clamp_u8((298 * c + 409 * e + 128) >> 8);
    g = clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
    b = clamp_u8((298 * c + 516 * d + 128) >> 8);
}

static bool convert_yuyv_to_rgb565(const uint8_t* src, size_t src_len, uint16_t* dst, int width, int height)
{
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (src == nullptr || dst == nullptr || src_len < pixel_count * 2) {
        return false;
    }

    for (size_t i = 0, out = 0; out + 1 < pixel_count; i += 4, out += 2) {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        const uint8_t y0 = src[i + 0];
        const uint8_t u  = src[i + 1];
        const uint8_t y1 = src[i + 2];
        const uint8_t v  = src[i + 3];

        yuv_to_rgb(y0, u, v, r, g, b);
        dst[out] = rgb888_to_rgb565(r, g, b);
        yuv_to_rgb(y1, u, v, r, g, b);
        dst[out + 1] = rgb888_to_rgb565(r, g, b);
    }

    return true;
}

static bool has_confident_face(const std::list<dl::detect::result_t>& results, float& best_score)
{
    best_score = 0.0F;
    for (const auto& result : results) {
        best_score = std::max(best_score, result.score);
        if (result.score < kMnpScoreThreshold || result.box.size() < 4 ||
            static_cast<int>(result.keypoint.size()) < kRequiredLandmarkValues) {
            continue;
        }

        const int width = result.box[2] - result.box[0];
        const int height = result.box[3] - result.box[1];
        const int side = std::max(width, height);
        if (side >= kMinFaceSidePx && side <= kMaxFaceSidePx) {
            return true;
        }
    }

    return false;
}

static void face_detect_wakeup_task(void*)
{
    mclog::tagInfo(kTag, "human_face_detect model storage={}, location={}", kModelStorage,
                   CONFIG_HUMAN_FACE_DETECT_MODEL_LOCATION);
    auto detect = new HumanFaceDetect();
    detect->set_score_thr(kMsrScoreThreshold, 0);
    detect->set_score_thr(kMnpScoreThreshold, 1);
    uint8_t* rgb565_buffer = nullptr;
    size_t rgb565_buffer_len = 0;
    TickType_t last_wake_tick = 0;
    int consecutive_hits = 0;

    mclog::tagInfo(kTag, "face detect wakeup task started");

    while (true) {
        if (!hal_bridge::is_xiaozhi_ready() || !hal_bridge::is_xiaozhi_idle()) {
            vTaskDelay(pdMS_TO_TICKS(kIdlePollIntervalMs));
            continue;
        }

        auto camera = hal_bridge::board_get_camera();
        if (camera == nullptr || !camera->StreamCaptures()) {
            vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
            continue;
        }

        const auto frame_data = camera->GetFrameData();
        const auto frame_len = camera->GetFrameSize();
        const int width = camera->GetFrameWidth();
        const int height = camera->GetFrameHeight();
        const int format = camera->GetFrameFormat();

        dl::image::img_t img = {
            .data = nullptr,
            .width = static_cast<uint16_t>(width),
            .height = static_cast<uint16_t>(height),
            .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565,
        };

        if (format == V4L2_PIX_FMT_RGB565) {
            img.data = const_cast<uint8_t*>(frame_data);
        } else if (format == V4L2_PIX_FMT_YUYV) {
            const size_t required_len = static_cast<size_t>(width) * static_cast<size_t>(height) * 2;
            if (rgb565_buffer_len < required_len) {
                if (rgb565_buffer != nullptr) {
                    heap_caps_free(rgb565_buffer);
                    rgb565_buffer = nullptr;
                }
                rgb565_buffer = static_cast<uint8_t*>(
                    heap_caps_malloc(required_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
                rgb565_buffer_len = rgb565_buffer == nullptr ? 0 : required_len;
            }

            if (rgb565_buffer == nullptr ||
                !convert_yuyv_to_rgb565(frame_data, frame_len, reinterpret_cast<uint16_t*>(rgb565_buffer), width, height)) {
                mclog::tagError(kTag, "failed to convert YUYV frame to RGB565");
                vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
                continue;
            }
            img.data = rgb565_buffer;
        } else {
            mclog::tagWarn(kTag, "unsupported camera frame format: 0x{:08X}", static_cast<uint32_t>(format));
            vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
            continue;
        }

        auto& results = detect->run(img);
        float best_score = 0.0F;
        if (has_confident_face(results, best_score)) {
            consecutive_hits++;
        } else {
            if (!results.empty()) {
                mclog::tagInfo(kTag, "ignored weak face detection: count={}, best_score={:.2f}", results.size(),
                               best_score);
            }
            consecutive_hits = 0;
        }

        if (consecutive_hits >= kRequiredConsecutiveHits) {
            const TickType_t now = xTaskGetTickCount();
            if (last_wake_tick == 0 ||
                (now - last_wake_tick) >= pdMS_TO_TICKS(kWakeCooldownMs)) {
                last_wake_tick = now;
                consecutive_hits = 0;
                mclog::tagInfo(kTag, "face detected, turning conversation on");
                hal_bridge::toggle_xiaozhi_chat_state();
            }
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
