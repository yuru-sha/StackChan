/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "face_detect_wakeup.h"

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32S3

#include <hal/board/config.h>
#include <hal/board/hal_bridge.h>
#include <esp_camera.h>
#include <human_face_detect.hpp>
#include <mooncake_log.h>

#include <algorithm>
#include <atomic>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace {

constexpr const char* kTag               = "AI.AGENT.FaceDetect";
constexpr uint32_t kIdlePollIntervalMs   = 1000;
constexpr uint32_t kDetectIntervalMs     = 1500;
constexpr uint32_t kWakeCooldownMs       = 10000;
constexpr uint32_t kUnsupportedBackoffMs = 3000;
constexpr uint32_t kStateLogIntervalMs   = 10000;
constexpr float kMsrScoreThreshold       = 0.70F;
constexpr float kMnpScoreThreshold       = 0.97F;
constexpr int kMinFaceSidePx             = 80;
constexpr int kMaxFaceSidePx             = 220;
constexpr int kFaceEdgeMarginPx          = 8;
constexpr int kMaxStableCenterShiftPx    = 32;
constexpr int kMaxStableSideShiftPx      = 35;
constexpr int kRequiredLandmarkValues    = 10;
constexpr int kRequiredConsecutiveHits   = 5;
constexpr int kCoreS3CameraXclkHz        = 10000000;
constexpr const char* kEspCameraVariant  = "esp_camera_rgb565";
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

static bool init_esp_camera()
{
    camera_config_t camera_config = {
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
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
        .sccb_i2c_port = 1,
    };

    const esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        mclog::tagError(kTag, "esp_camera_init failed: 0x{:X}", static_cast<int>(err));
        return false;
    }

    auto sensor = esp_camera_sensor_get();
    if (sensor != nullptr) {
        sensor->set_hmirror(sensor, 0);
    }

    mclog::tagInfo(kTag, "esp_camera initialized: RGB565 QVGA, xclk={}Hz", kCoreS3CameraXclkHz);
    return true;
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
    TickType_t last_wake_tick = 0;
    TickType_t last_state_log_tick = 0;
    int consecutive_hits = 0;
    bool logged_frame_info = false;
    bool last_ready = false;
    bool last_idle = false;
    bool has_logged_state = false;
    FaceDetectionSummary previous_detection;

    if (!init_esp_camera()) {
        mclog::tagError(kTag, "face detect wakeup disabled because camera init failed");
        vTaskDelete(nullptr);
        return;
    }

    mclog::tagInfo(kTag, "face detect wakeup task started");

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

        auto frame = esp_camera_fb_get();
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

        dl::image::img_t img = {
            .data = frame->buf,
            .width = static_cast<uint16_t>(frame->width),
            .height = static_cast<uint16_t>(frame->height),
            .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565,
        };

        FaceDetectionSummary detection_summary;
        auto& results = detect->run(img);
        bool has_face = has_confident_face(results, frame->width, frame->height, kEspCameraVariant, detection_summary);
        if (!has_face) {
            if (!results.empty()) {
                mclog::tagInfo(kTag,
                               "ignored face detection: variant={}, count={}, best_score={:.2f}, box=[{},{},{},{}], side={}, keypoints={}, reason={}",
                               detection_summary.variant, results.size(), detection_summary.best_score,
                               detection_summary.best_box[0], detection_summary.best_box[1],
                               detection_summary.best_box[2], detection_summary.best_box[3],
                               detection_summary.best_side, detection_summary.best_keypoints,
                               detection_summary.reject_reason);
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
