/*
 * SPDX-FileCopyrightText: 2026 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#include "robot.h"

using namespace uitk;
using namespace uitk::lvgl_cpp;
using namespace stackchan::avatar;

static const Vector2i _mouth_pos        = Vector2i(0, 68);
static const Vector2i _mouth_min_offset = Vector2i(-8, -6);
static const Vector2i _mouth_max_offset = Vector2i(8, 6);
static const Vector2i _mouth_min_size   = Vector2i(90, 8);
static const Vector2i _mouth_max_size   = Vector2i(66, 34);
static const int _mouth_min_radius      = 4;
static const int _mouth_max_radius      = 12;

RobotMouth::RobotMouth(lv_obj_t* parent, lv_color_t mouthColor, lv_color_t muzzleColor, lv_color_t noseColor)
{
    _container = std::make_unique<Container>(parent);
    _container->setAlign(LV_ALIGN_CENTER);
    _container->setBorderWidth(0);
    _container->setBgOpa(0);
    _container->setPadding(0, 0, 0, 0);
    _container->setSize(110, 55);
    _container->setTransformPivot(55, 27);
    _container->addFlag(LV_OBJ_FLAG_CLICKABLE);
    _container->addFlag(LV_OBJ_FLAG_EVENT_BUBBLE);
    _container->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    _mouth = std::make_unique<Container>(_container->get());
    _mouth->setAlign(LV_ALIGN_CENTER);
    _mouth->setBorderWidth(0);
    _mouth->setBgColor(mouthColor);
    _mouth->setBgOpa(LV_OPA_COVER);
    _mouth->setSize(90, 10);
    _mouth->setRadius(5);
    _mouth->addFlag(LV_OBJ_FLAG_CLICKABLE);
    _mouth->addFlag(LV_OBJ_FLAG_EVENT_BUBBLE);
    _mouth->removeFlag(LV_OBJ_FLAG_SCROLLABLE);

    setPosition(_position);
    setWeight(0);
    setRotation(0);
}

RobotMouth::~RobotMouth()
{
    _mouth.reset();
    _container.reset();
}

void RobotMouth::setPosition(const Vector2i& position)
{
    Element::setPosition(position);

    auto pos_x = _mouth_pos.x + map_range(_position.x, -100, 100, _mouth_min_offset.x, _mouth_max_offset.x);
    auto pos_y = _mouth_pos.y + map_range(_position.y, -100, 100, _mouth_min_offset.y, _mouth_max_offset.y);

    _container->setPos(pos_x, pos_y);
}

void RobotMouth::setWeight(int weight)
{
    Feature::setWeight(weight);

    auto size_x = map_range(_weight, 0, 100, _mouth_min_size.x, _mouth_max_size.x);
    auto size_y = map_range(_weight, 0, 100, _mouth_min_size.y, _mouth_max_size.y);
    auto radius = map_range(_weight, 0, 100, _mouth_min_radius, _mouth_max_radius);

    _mouth->setSize(size_x, size_y);
    _mouth->setRadius(radius);
}

void RobotMouth::setRotation(int rotation)
{
    Element::setRotation(rotation);

    _container->setRotation(rotation);
}

void RobotMouth::setEmotion(const Emotion& emotion)
{
    if (getIgnoreEmotion()) {
        return;
    }

    switch (emotion) {
        case Emotion::Neutral:
            setWeight(0);
            setRotation(0);
            break;
        case Emotion::Happy:
            setWeight(28);
            setRotation(0);
            break;
        case Emotion::Angry:
            setWeight(10);
            setRotation(40);
            break;
        case Emotion::Sad:
            setWeight(6);
            setRotation(3560);
            break;
        case Emotion::Doubt:
            setWeight(16);
            setRotation(80);
            break;
        case Emotion::Sleepy:
            setWeight(8);
            setRotation(0);
            break;
        default:
            break;
    }
}

void RobotMouth::setVisible(bool visible)
{
    Element::setVisible(visible);
    _container->setHidden(!visible);
}
