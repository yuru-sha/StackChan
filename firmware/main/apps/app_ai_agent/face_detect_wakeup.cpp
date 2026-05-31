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
constexpr float kMsrScoreThreshold       = 0.70F;
constexpr float kMnpScoreThreshold       = 0.97F;
constexpr int kMinFaceSidePx             = 80;
constexpr int kMaxFaceSidePx             = 220;
constexpr int kFaceEdgeMarginPx          = 8;
constexpr int kMaxStableCenterShiftPx    = 32;
constexpr int kMaxStableSideShiftPx      = 35;
constexpr int kRequiredLandmarkValues    = 10;
constexpr int kRequiredConsecutiveHits   = 5;
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

static void rgb565_to_rgb888(uint16_t pixel, uint8_t& r, uint8_t& g, uint8_t& b)
{
    r = static_cast<uint8_t>(((pixel >> 11) & 0x1F) << 3);
    g = static_cast<uint8_t>(((pixel >> 5) & 0x3F) << 2);
    b = static_cast<uint8_t>((pixel & 0x1F) << 3);
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

static bool convert_rgb565_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst, int width, int height)
{
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (src == nullptr || dst == nullptr || src_len < pixel_count * 2) {
        return false;
    }

    auto src16 = reinterpret_cast<const uint16_t*>(src);
    for (size_t i = 0; i < pixel_count; ++i) {
        rgb565_to_rgb888(src16[i], dst[i * 3], dst[i * 3 + 1], dst[i * 3 + 2]);
    }

    return true;
}

static bool convert_yuyv_to_rgb888(const uint8_t* src, size_t src_len, uint8_t* dst, int width, int height)
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
        dst[out * 3] = r;
        dst[out * 3 + 1] = g;
        dst[out * 3 + 2] = b;
        yuv_to_rgb(y1, u, v, r, g, b);
        dst[(out + 1) * 3] = r;
        dst[(out + 1) * 3 + 1] = g;
        dst[(out + 1) * 3 + 2] = b;
    }

    return true;
}

struct FaceDetectionSummary {
    float best_score = 0.0F;
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
                               FaceDetectionSummary& summary)
{
    summary = {};
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
        if (result.score < kMnpScoreThreshold || result.box.size() < 4 ||
            static_cast<int>(result.keypoint.size()) < kRequiredLandmarkValues) {
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
    auto detect = new HumanFaceDetect();
    detect->set_score_thr(kMsrScoreThreshold, 0);
    detect->set_score_thr(kMnpScoreThreshold, 1);
    uint8_t* detect_buffer = nullptr;
    size_t detect_buffer_len = 0;
    TickType_t last_wake_tick = 0;
    int consecutive_hits = 0;
    bool logged_frame_info = false;
    FaceDetectionSummary previous_detection;

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
        if (!logged_frame_info) {
            mclog::tagInfo(kTag, "camera frame: {}x{}, len={}, format=0x{:08X}", width, height, frame_len,
                           static_cast<uint32_t>(format));
            logged_frame_info = true;
        }

        const size_t required_len = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
        if (detect_buffer_len < required_len) {
            if (detect_buffer != nullptr) {
                heap_caps_free(detect_buffer);
                detect_buffer = nullptr;
            }
            detect_buffer =
                static_cast<uint8_t*>(heap_caps_malloc(required_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
            detect_buffer_len = detect_buffer == nullptr ? 0 : required_len;
        }

        dl::image::img_t img = {
            .data = nullptr,
            .width = static_cast<uint16_t>(width),
            .height = static_cast<uint16_t>(height),
            .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888,
        };

        if (format == V4L2_PIX_FMT_RGB565) {
            if (detect_buffer == nullptr ||
                !convert_rgb565_to_rgb888(frame_data, frame_len, detect_buffer, width, height)) {
                mclog::tagError(kTag, "failed to convert RGB565 frame to RGB888");
                vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
                continue;
            }
            img.data = detect_buffer;
        } else if (format == V4L2_PIX_FMT_YUYV) {
            if (detect_buffer == nullptr ||
                !convert_yuyv_to_rgb888(frame_data, frame_len, detect_buffer, width, height)) {
                mclog::tagError(kTag, "failed to convert YUYV frame to RGB888");
                vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
                continue;
            }
            img.data = detect_buffer;
        } else {
            mclog::tagWarn(kTag, "unsupported camera frame format: 0x{:08X}", static_cast<uint32_t>(format));
            vTaskDelay(pdMS_TO_TICKS(kUnsupportedBackoffMs));
            continue;
        }

        auto& results = detect->run(img);
        FaceDetectionSummary detection_summary;
        if (has_confident_face(results, width, height, detection_summary)) {
            if (is_stable_face_hit(detection_summary, previous_detection)) {
                consecutive_hits++;
            } else {
                consecutive_hits = 1;
                mclog::tagInfo(kTag, "reset face hits because bbox moved too much");
            }
            previous_detection = detection_summary;
            mclog::tagInfo(kTag,
                           "accepted face hit {}/{}: score={:.2f}, box=[{},{},{},{}], center=[{},{}], side={}, keypoints={}",
                           consecutive_hits, kRequiredConsecutiveHits, detection_summary.best_score,
                           detection_summary.accepted_box[0], detection_summary.accepted_box[1],
                           detection_summary.accepted_box[2], detection_summary.accepted_box[3],
                           detection_summary.center_x, detection_summary.center_y, detection_summary.side,
                           detection_summary.accepted_keypoints);
        } else {
            if (!results.empty()) {
                mclog::tagInfo(kTag,
                               "ignored face detection: count={}, best_score={:.2f}, box=[{},{},{},{}], side={}, keypoints={}, reason={}",
                               results.size(), detection_summary.best_score, detection_summary.best_box[0],
                               detection_summary.best_box[1], detection_summary.best_box[2],
                               detection_summary.best_box[3], detection_summary.best_side,
                               detection_summary.best_keypoints, detection_summary.reject_reason);
            }
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
