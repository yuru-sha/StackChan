/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "robot.h"

using namespace uitk;
using namespace uitk::lvgl_cpp;
using namespace stackchan::avatar;

static const Vector2i _eye_pos        = Vector2i(-70, -14);
static const Vector2i _pupil_min_offset = Vector2i(-8, -5);
static const Vector2i _pupil_max_offset = Vector2i(8, 5);
static const Vector2i _eye_size_limit = Vector2i(76, 76);

RobotEyes::RobotEyes(lv_obj_t* parent,
                     lv_color_t eyeColor,
                     lv_color_t shineColor,
                     lv_color_t lidColor,
                     bool isLeftEye)
{
    _is_left_eye = isLeftEye;
    _lid_color   = lidColor;

    _container = std::make_unique<Container>(parent);
    _container->setAlign(LV_ALIGN_CENTER);
    _container->setBorderWidth(0);
    _container->setBgOpa(0);
    _container->setPadding(0, 0, 0, 0);
    _container->setSize(76, 76);
    _container->setTransformPivot(38, 38);
    _container->addFlag(LV_OBJ_FLAG_CLICKABLE);
    _container->addFlag(LV_OBJ_FLAG_EVENT_BUBBLE);
    _container->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    _sclera = std::make_unique<Container>(_container->get());
    _sclera->setAlign(LV_ALIGN_CENTER);
    _sclera->setBorderWidth(0);
    _sclera->setBgColor(shineColor);
    _sclera->setRadius(LV_RADIUS_CIRCLE);
    _sclera->addFlag(LV_OBJ_FLAG_CLICKABLE);
    _sclera->addFlag(LV_OBJ_FLAG_EVENT_BUBBLE);
    _sclera->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    _pupil = std::make_unique<Container>(_container->get());
    _pupil->setAlign(LV_ALIGN_CENTER);
    _pupil->setBorderWidth(0);
    _pupil->setBgColor(eyeColor);
    _pupil->setRadius(LV_RADIUS_CIRCLE);
    _pupil->addFlag(LV_OBJ_FLAG_CLICKABLE);
    _pupil->addFlag(LV_OBJ_FLAG_EVENT_BUBBLE);
    _pupil->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    _shine = std::make_unique<Container>(_container->get());
    _shine->setAlign(LV_ALIGN_CENTER);
    _shine->setBorderWidth(0);
    _shine->setBgColor(shineColor);
    _shine->setRadius(LV_RADIUS_CIRCLE);
    _shine->setSize(0, 0);
    _shine->setHidden(true);
    _shine->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    _lid = std::make_unique<Container>(_container->get());
    _lid->setAlign(LV_ALIGN_CENTER);
    _lid->setBorderWidth(0);
    _lid->setBgColor(_lid_color);
    _lid->setRadius(8);
    _lid->setSize(0, 0);
    _lid->setHidden(true);
    _lid->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    setSize(0);
    setWeight(100);
    setPosition(_position);
    setRotation(0);
}

RobotEyes::~RobotEyes()
{
    _lid.reset();
    _shine.reset();
    _pupil.reset();
    _sclera.reset();
    _container.reset();
}

void RobotEyes::setPosition(const Vector2i& position)
{
    Element::setPosition(position);

    auto pos_x = _is_left_eye ? _eye_pos.x : -_eye_pos.x;
    _container->setPos(pos_x, _eye_pos.y);

    auto pupil_x = map_range(_position.x, -100, 100, _pupil_min_offset.x, _pupil_max_offset.x);
    auto pupil_y = map_range(_position.y, -100, 100, _pupil_min_offset.y, _pupil_max_offset.y);
    _pupil->setPos(pupil_x, pupil_y);
}

void RobotEyes::setWeight(int weight)
{
    Feature::setWeight(weight);
    updateShape();
}

void RobotEyes::setRotation(int rotation)
{
    Element::setRotation(rotation);
    _container->setRotation(rotation);
}

void RobotEyes::setEmotion(const Emotion& emotion)
{
    if (getIgnoreEmotion()) {
        return;
    }

    auto apply_style = [this](int weight, int rotation) {
        setWeight(weight);
        if (_is_left_eye) {
            setRotation(rotation);
        } else {
            setRotation(3600 - rotation);
        }
    };

    switch (emotion) {
        case Emotion::Neutral:
            apply_style(100, 0);
            break;
        case Emotion::Happy:
            apply_style(34, 120);
            break;
        case Emotion::Angry:
            apply_style(58, 260);
            break;
        case Emotion::Sad:
            apply_style(52, 3350);
            break;
        case Emotion::Doubt:
            apply_style(_is_left_eye ? 78 : 45, _is_left_eye ? 80 : 3500);
            break;
        case Emotion::Sleepy:
            apply_style(24, 0);
            break;
        default:
            break;
    }
}

void RobotEyes::setVisible(bool visible)
{
    Element::setVisible(visible);
    _container->setHidden(!visible);
}

void RobotEyes::setSize(int size)
{
    Feature::setSize(size);
    updateShape();
}

void RobotEyes::updateShape()
{
    _eye_width  = map_range(_size, -100, 100, _eye_size_limit.x, _eye_size_limit.y);
    _eye_height = _eye_size_limit.y;

    _sclera->setSize(_eye_width, _eye_height);
    _sclera->setRadius(LV_RADIUS_CIRCLE);

    const auto pupil_width  = 58;
    const auto pupil_height = map_range(_weight, 0, 100, 8, 58);
    _pupil->setSize(pupil_width, pupil_height);
    _pupil->setRadius(LV_RADIUS_CIRCLE);

    _shine->setHidden(true);
    _lid->setHidden(true);
}
