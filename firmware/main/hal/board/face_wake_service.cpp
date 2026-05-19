#include "face_wake_service.h"

#if CONFIG_STACKCHAN_FACE_WAKE_ENABLE

#include "application.h"
#include "hal_bridge.h"
#include "human_face_detect.hpp"
#include "linux/videodev2.h"

#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_timer.h>

#include <cstdlib>
#include <cstring>
#include <memory>

#ifndef CONFIG_STACKCHAN_FACE_WAKE_CENTER_TOLERANCE_PERCENT
#define CONFIG_STACKCHAN_FACE_WAKE_CENTER_TOLERANCE_PERCENT 45
#endif

namespace {
constexpr const char* TAG = "FaceWakeService";

struct HeapCapsDeleter {
    void operator()(uint8_t* ptr) const
    {
        if (ptr != nullptr) {
            heap_caps_free(ptr);
        }
    }
};

using FramePtr = std::unique_ptr<uint8_t, HeapCapsDeleter>;

bool ToDlPixelType(int frame_format, dl::image::pix_type_t& pix_type)
{
    switch (frame_format) {
        case V4L2_PIX_FMT_RGB565:
            pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE;
            return true;
        case V4L2_PIX_FMT_RGB24:
            pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;
            return true;
        case V4L2_PIX_FMT_YUYV:
        case V4L2_PIX_FMT_YUV422P:
            pix_type = dl::image::DL_IMAGE_PIX_TYPE_YUYV;
            return true;
        case V4L2_PIX_FMT_GREY:
            pix_type = dl::image::DL_IMAGE_PIX_TYPE_GRAY;
            return true;
        default:
            return false;
    }
}

bool IsConversationState(DeviceState state)
{
    return state == kDeviceStateConnecting || state == kDeviceStateListening || state == kDeviceStateSpeaking;
}

TickType_t GetDetectionDelayTicks(bool audio_active)
{
    const int interval_ms = audio_active ? CONFIG_STACKCHAN_FACE_WAKE_ACTIVE_INTERVAL_MS :
                                           CONFIG_STACKCHAN_FACE_WAKE_INTERVAL_MS;
    return pdMS_TO_TICKS(interval_ms);
}

bool IsCenteredFace(const dl::detect::result_t& result, uint16_t width, uint16_t height)
{
    if (result.box.size() < 4) {
        return false;
    }

    const int center_x = (result.box[0] + result.box[2]) / 2;
    const int center_y = (result.box[1] + result.box[3]) / 2;
    const int max_dx = static_cast<int>(width) * CONFIG_STACKCHAN_FACE_WAKE_CENTER_TOLERANCE_PERCENT / 100;
    const int max_dy = static_cast<int>(height) * CONFIG_STACKCHAN_FACE_WAKE_CENTER_TOLERANCE_PERCENT / 100;

    return std::abs(center_x - static_cast<int>(width) / 2) <= max_dx &&
           std::abs(center_y - static_cast<int>(height) / 2) <= max_dy;
}
}  // namespace

FaceWakeService::FaceWakeService(StackChanCamera* camera) : camera_(camera)
{
}

FaceWakeService::~FaceWakeService()
{
    if (task_handle_ != nullptr) {
        vTaskDelete(task_handle_);
        task_handle_ = nullptr;
    }
}

void FaceWakeService::Start()
{
    if (camera_ == nullptr || task_handle_ != nullptr) {
        return;
    }

    BaseType_t ok = xTaskCreate(TaskEntry, "face_wake", 8192, this, 1, &task_handle_);
    if (ok != pdPASS) {
        task_handle_ = nullptr;
        ESP_LOGE(TAG, "Failed to create face wake task");
    }
}

void FaceWakeService::TaskEntry(void* arg)
{
    static_cast<FaceWakeService*>(arg)->Run();
}

void FaceWakeService::Run()
{
    ESP_LOGI(TAG, "Face wake task started");

    while (true) {
        auto& app = Application::GetInstance();
        const auto state = app.GetDeviceState();
        const bool audio_active = IsConversationState(state);

        if (audio_active) {
            last_face_time_ms_ = esp_timer_get_time() / 1000;
            vTaskDelay(GetDetectionDelayTicks(true));
            continue;
        }

        if (face_present_) {
            face_present_ = false;
            consecutive_face_frames_ = 0;
        }

        const bool detected = DetectFace();
        const int64_t now_ms = esp_timer_get_time() / 1000;
        UpdateConversation(detected, now_ms);
        vTaskDelay(GetDetectionDelayTicks(audio_active));
    }
}

bool FaceWakeService::DetectFace()
{
    if (!hal_bridge::is_xiaozhi_ready()) {
        return false;
    }

    if (!camera_->StreamCaptures()) {
        ESP_LOGW(TAG, "Failed to capture camera frame for face detection");
        return false;
    }

    const size_t frame_size = camera_->GetFrameSize();
    if (camera_->GetFrameData() == nullptr || frame_size == 0 || camera_->GetFrameWidth() <= 0 ||
        camera_->GetFrameHeight() <= 0) {
        return false;
    }

    dl::image::pix_type_t pix_type;
    if (!ToDlPixelType(camera_->GetFrameFormat(), pix_type)) {
        ESP_LOGW(TAG, "Unsupported frame format for face detection: 0x%08x", camera_->GetFrameFormat());
        return false;
    }

    FramePtr frame_copy(static_cast<uint8_t*>(heap_caps_malloc(frame_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
    if (frame_copy == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate face detection frame copy (%u bytes)", static_cast<unsigned>(frame_size));
        return false;
    }
    memcpy(frame_copy.get(), camera_->GetFrameData(), frame_size);

    static HumanFaceDetect detector(HumanFaceDetect::MSRMNP_S8_V1, true);
    const float score_threshold = static_cast<float>(CONFIG_STACKCHAN_FACE_WAKE_MIN_SCORE) / 100.0f;
    detector.set_score_thr(score_threshold, 0);
    detector.set_score_thr(score_threshold, 1);

    dl::image::img_t img = {
        .data = frame_copy.get(),
        .width = static_cast<uint16_t>(camera_->GetFrameWidth()),
        .height = static_cast<uint16_t>(camera_->GetFrameHeight()),
        .pix_type = pix_type,
    };

    auto& results = detector.run(img);
    for (const auto& result : results) {
        if (result.score >= score_threshold && IsCenteredFace(result, img.width, img.height)) {
            ESP_LOGD(TAG, "Face detected score=%.2f", result.score);
            return true;
        }
    }
    return false;
}

void FaceWakeService::UpdateConversation(bool detected, int64_t now_ms)
{
    auto& app = Application::GetInstance();
    const auto state = app.GetDeviceState();

    if (detected) {
        last_face_time_ms_ = now_ms;
        if (consecutive_face_frames_ < CONFIG_STACKCHAN_FACE_WAKE_ON_CONSECUTIVE_FRAMES) {
            consecutive_face_frames_++;
        }
        if (!face_present_ && consecutive_face_frames_ >= CONFIG_STACKCHAN_FACE_WAKE_ON_CONSECUTIVE_FRAMES) {
            if (state == kDeviceStateIdle) {
                face_present_ = true;
                ESP_LOGI(TAG, "Face present, starting conversation");
                hal_bridge::toggle_xiaozhi_chat_state();
            } else {
                ESP_LOGD(TAG, "Face present but device is not idle: %d", static_cast<int>(state));
            }
        }
        return;
    }

    consecutive_face_frames_ = 0;
    if (!face_present_) {
        return;
    }

    if ((now_ms - last_face_time_ms_) >= CONFIG_STACKCHAN_FACE_WAKE_OFF_TIMEOUT_MS) {
        face_present_ = false;
        ESP_LOGI(TAG, "Face missing, stopping conversation");
        if (IsConversationState(state)) {
            app.StopConversation();
        }
    }
}

#endif  // CONFIG_STACKCHAN_FACE_WAKE_ENABLE
