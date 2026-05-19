/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "robot.h"

using namespace stackchan::avatar;
using namespace uitk::lvgl_cpp;

void RobotAvatar::init(lv_obj_t* parent, const lv_font_t* font)
{
    _panel = std::make_unique<Container>(parent);
    _panel->align(LV_ALIGN_CENTER, 0, 0);
    _panel->setSize(320, 240);
    _panel->setRadius(0);
    _panel->setBorderWidth(0);
    _panel->setBgColor(redColor);
    _panel->removeFlag(LV_OBJ_FLAG_SCROLLABLE);
    _panel->setPadding(0, 0, 0, 0);

    _key_elements.leftEye = std::make_unique<RobotEyes>(_panel->get(), blackColor, whiteColor, blackColor, true);
    _key_elements.rightEye = std::make_unique<RobotEyes>(_panel->get(), blackColor, whiteColor, blackColor, false);
    _key_elements.mouth = std::make_unique<RobotMouth>(_panel->get(), whiteColor, blackColor, blackColor);
    _key_elements.speechBubble = std::make_unique<DefaultSpeechBubble>(_panel->get(), whiteColor, secondaryColor, font);
}

Container* RobotAvatar::getPanel() const
{
    if (_panel) {
        return _panel.get();
    }
    return nullptr;
}
