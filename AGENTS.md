# Repository Guidelines

## Project Structure & Module Organization

This repository contains StackChan firmware, mobile app, backend server, and remote controller code.

- `firmware/`: ESP-IDF firmware for the CoreS3 device. Main code is in `firmware/main/`, app modules in `firmware/main/apps/`, and assets in `firmware/main/assets/`.
- `remote/code/`: ESP-IDF remote controller firmware for ESP-NOW, joystick, and LVGL UI.
- `app/`: Flutter app. Dart code is in `app/lib/`, tests in `app/test/`, and bundled media/models in `app/assets/`.
- `server/`: Go backend. API definitions are in `server/api/`, implementation code in `server/internal/`, config in `server/manifest/`, and database setup in `server/check_list/`.

## Build, Test, and Development Commands

- `cd firmware && python3 ./fetch_repos.py`: fetch firmware dependencies.
- `cd firmware && idf.py build`: build the device firmware with ESP-IDF v5.5.4.
- `cd firmware && idf.py flash`: flash firmware to a connected device.
- `cd remote/code && idf.py build`: build the ESP-NOW remote firmware.
- `cd app && flutter pub get`: install app dependencies.
- `cd app && flutter analyze`: run Dart analysis using `flutter_lints`.
- `cd app && flutter test`: run app tests.
- `cd server && go mod download`: install Go dependencies.
- `cd server && go test ./...`: run server tests.
- `cd server && go run main.go` or `make run`: start the backend.

## Coding Style & Naming Conventions

Use the existing style in each subproject. Format Dart with `dart format .`; use lower_snake_case files and UpperCamelCase classes. Format Go with `gofmt`; keep package names short and lowercase. For ESP-IDF C/C++, follow nearby style and keep headers beside implementations.

## Testing Guidelines

Place Flutter tests under `app/test/` with `_test.dart` suffix. Place Go tests beside package code with `_test.go` suffix. Firmware changes should pass `idf.py build`; include hardware test notes for sensors, motors, BLE, ESP-NOW, camera, or audio.

## Commit & Pull Request Guidelines

Recent history uses short subjects such as `update firmware v1.4.1` and `add MIT License`, plus merge commits from feature branches. Keep commits focused by subproject.

Pull requests should include a summary, affected area (`firmware`, `remote`, `app`, or `server`), commands run, linked issues, and screenshots or videos for app UI changes. Note config changes, migrations, hardware used, and tests not run.

## Security & Configuration Tips

Do not commit private keys, production JWT secrets, API tokens, keystores, or local database credentials. Keep `app/android/key.properties` and JKS files local. Review `server/manifest/config/config.yaml` and app endpoint constants before sharing builds.
