#pragma once

#include "sdkconfig.h"

#if CONFIG_STACKCHAN_FACE_WAKE_ENABLE

#include "stackchan_camera.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <atomic>
#include <mutex>

class FaceWakeService {
public:
    explicit FaceWakeService(StackChanCamera* camera);
    ~FaceWakeService();

    void Start();
    void SetEnabled(bool enabled);
    bool IsEnabled() const;
    bool ToggleEnabled();

private:
    StackChanCamera* camera_ = nullptr;
    TaskHandle_t task_handle_ = nullptr;
    std::atomic<bool> enabled_{true};
    std::mutex state_mutex_;
    bool face_present_ = false;
    int consecutive_face_frames_ = 0;
    int64_t last_face_time_ms_ = 0;

    static void TaskEntry(void* arg);
    void Run();
    void ResetState();
    bool DetectFace();
    void UpdateConversation(bool detected, int64_t now_ms);
};

#else

class StackChanCamera;

class FaceWakeService {
public:
    explicit FaceWakeService(StackChanCamera*) {}
    void Start() {}
    void SetEnabled(bool) {}
    bool IsEnabled() const { return false; }
    bool ToggleEnabled() { return false; }
};

#endif  // CONFIG_STACKCHAN_FACE_WAKE_ENABLE
