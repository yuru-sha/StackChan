/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */

#include "sdkconfig.h"

#if CONFIG_IDF_TARGET_ESP32S3

#include <M5Unified.h>
#include <bsp/esp-bsp.h>
#include <esp_camera.h>
#include <esp_log.h>
#include <human_face_detect.hpp>

namespace {

const char* TAG = "AI.AGENT.FaceDetect";
M5Canvas canvas(&M5.Lcd);
HumanFaceDetect* detect = new HumanFaceDetect();

constexpr i2c_port_t kCoreS3I2cPort = I2C_NUM_1;
constexpr int kCoreS3I2cSda = 12;
constexpr int kCoreS3I2cScl = 11;

esp_err_t camera_power_on()
{
    if (!M5.In_I2C.begin(kCoreS3I2cPort, kCoreS3I2cSda, kCoreS3I2cScl)) {
        ESP_LOGE(TAG, "CoreS3 internal I2C init failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG,
             "CoreS3 camera I2C: port=%d, sda=%d, scl=%d",
             static_cast<int>(M5.In_I2C.getPort()),
             static_cast<int>(M5.In_I2C.getSDA()),
             static_cast<int>(M5.In_I2C.getSCL()));

    if (!M5.In_I2C.writeRegister8(0x58, 0x11, 0x10, CONFIG_BSP_I2C_CLK_SPEED_HZ) ||
        !M5.In_I2C.bitOn(0x58, 0x03, 0b00000001, CONFIG_BSP_I2C_CLK_SPEED_HZ)) {
        ESP_LOGE(TAG, "CoreS3 camera power enable failed");
        return ESP_FAIL;
    }

    vTaskDelay(pdMS_TO_TICKS(50));
    return ESP_OK;
}

esp_err_t camera_init()
{
    ESP_LOGI(TAG, "Camera Init");

    esp_err_t power_err = camera_power_on();
    if (power_err != ESP_OK) {
        return power_err;
    }

    camera_config_t camera_config_for_core_s3 = BSP_CAMERA_DEFAULT_CONFIG;
    camera_config_for_core_s3.pixel_format = PIXFORMAT_RGB565;
    camera_config_for_core_s3.frame_size = FRAMESIZE_QVGA;
    camera_config_for_core_s3.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&camera_config_for_core_s3);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera Init Failed: 0x%x", err);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Camera Init OK");
    return ESP_OK;
}

esp_err_t camera_capture_and_infer()
{
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera Capture Failed");
        M5.Display.println("Camera Capture Failed");
        return ESP_FAIL;
    }

    dl::image::img_t image;
    image.data = fb->buf;
    image.height = fb->height;
    image.width = fb->width;
    image.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;

    canvas.writePixels(reinterpret_cast<uint16_t*>(fb->buf), static_cast<int>(fb->len / 2));

    auto& detect_results = detect->run(image);
    int index = 0;
    for (const auto& result : detect_results) {
        if (result.box.size() < 4 || result.keypoint.size() < 10) {
            continue;
        }

        ESP_LOGI(TAG,
                 "raw face result %d: score=%f, box=[%d,%d,%d,%d]",
                 index,
                 result.score,
                 result.box[0],
                 result.box[1],
                 result.box[2],
                 result.box[3]);
        ESP_LOGI(TAG,
                 "keypoints: [%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]",
                 result.keypoint[0],
                 result.keypoint[1],
                 result.keypoint[2],
                 result.keypoint[3],
                 result.keypoint[4],
                 result.keypoint[5],
                 result.keypoint[6],
                 result.keypoint[7],
                 result.keypoint[8],
                 result.keypoint[9]);

        int width = result.box[2] - result.box[0];
        int height = result.box[3] - result.box[1];
        canvas.drawRect(result.box[0], result.box[1], width, height, TFT_RED);
        for (int point = 0; point <= 9; point += 2) {
            canvas.fillCircle(result.keypoint[point], result.keypoint[point + 1], 8, TFT_GREEN);
        }
        ++index;
    }

    canvas.pushSprite(0, 0);

    esp_camera_fb_return(fb);
    return ESP_OK;
}

}  // namespace

extern "C" void app_main(void)
{
    auto cfg = M5.config();
    cfg.output_power = true;
    M5.begin(cfg);

    M5.Display.setFont(&fonts::efontJA_24);
    M5.Display.println("HelloWorld");
    ESP_LOGI(TAG, "HelloWorld");

    canvas.createSprite(320, 240);

#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_mount());
#endif

    if (camera_init() != ESP_OK) {
        while (true) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    while (true) {
        camera_capture_and_infer();
    }

    canvas.deleteSprite();
    delete detect;

#if CONFIG_HUMAN_FACE_DETECT_MODEL_IN_SDCARD
    ESP_ERROR_CHECK(bsp_sdcard_unmount());
#endif
}

#else

extern "C" void app_main(void) {}

#endif
