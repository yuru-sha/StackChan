#pragma once

#include "sdkconfig.h"

#if CONFIG_STACKCHAN_FACE_WAKE_ENABLE

#include "stackchan_camera.h"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

class FaceWakeService {
public:
    explicit FaceWakeService(StackChanCamera* camera);
    ~FaceWakeService();

    void Start();

private:
    StackChanCamera* camera_ = nullptr;
    TaskHandle_t task_handle_ = nullptr;
    bool face_present_ = false;
    int consecutive_face_frames_ = 0;
    int64_t last_face_time_ms_ = 0;

    static void TaskEntry(void* arg);
    void Run();
    bool DetectFace();
    void UpdateConversation(bool detected, int64_t now_ms);
};

#else

class StackChanCamera;

class FaceWakeService {
public:
    explicit FaceWakeService(StackChanCamera*) {}
    void Start() {}
};

#endif  // CONFIG_STACKCHAN_FACE_WAKE_ENABLE
