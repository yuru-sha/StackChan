/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "../default/default.h"
#include "../../avatar/avatar.h"
#include "../../avatar/elements/feature.h"
#include <lvgl.h>
#include <smooth_lvgl.hpp>
#include <memory>

namespace stackchan::avatar {

class RobotAvatar : public Avatar {
public:
    lv_color_t redColor       = lv_color_hex(0x000000);
    lv_color_t deepRedColor   = lv_color_hex(0x000000);
    lv_color_t blackColor     = lv_color_hex(0x000000);
    lv_color_t whiteColor     = lv_color_hex(0xFFFFFF);
    lv_color_t goldColor      = lv_color_hex(0xFFFFFF);
    lv_color_t secondaryColor = lv_color_hex(0x000000);

    void init(lv_obj_t* parent, const lv_font_t* font = &lv_font_montserrat_16);
    uitk::lvgl_cpp::Container* getPanel() const;

private:
    std::unique_ptr<uitk::lvgl_cpp::Container> _panel;
};

class RobotEyes : public Feature {
public:
    RobotEyes(lv_obj_t* parent, lv_color_t eyeColor, lv_color_t shineColor, lv_color_t lidColor, bool isLeftEye);
    ~RobotEyes();

    void setPosition(const uitk::Vector2i& position) override;
    void setWeight(int weight) override;
    void setRotation(int rotation) override;
    void setEmotion(const Emotion& emotion) override;
    void setVisible(bool visible) override;
    void setSize(int size) override;

private:
    void updateShape();

    bool _is_left_eye = false;
    lv_color_t _lid_color;
    int _eye_width  = 48;
    int _eye_height = 48;

    std::unique_ptr<uitk::lvgl_cpp::Container> _container;
    std::unique_ptr<uitk::lvgl_cpp::Container> _sclera;
    std::unique_ptr<uitk::lvgl_cpp::Container> _pupil;
    std::unique_ptr<uitk::lvgl_cpp::Container> _shine;
    std::unique_ptr<uitk::lvgl_cpp::Container> _lid;
};

class RobotMouth : public Feature {
public:
    RobotMouth(lv_obj_t* parent, lv_color_t mouthColor, lv_color_t muzzleColor, lv_color_t noseColor);
    ~RobotMouth();

    void setPosition(const uitk::Vector2i& position) override;
    void setWeight(int weight) override;
    void setRotation(int rotation) override;
    void setEmotion(const Emotion& emotion) override;
    void setVisible(bool visible) override;

private:
    std::unique_ptr<uitk::lvgl_cpp::Container> _container;
    std::unique_ptr<uitk::lvgl_cpp::Container> _mouth;
};

}  // namespace stackchan::avatar
