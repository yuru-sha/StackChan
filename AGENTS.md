# Repository Guidelines

## Project Overview

StackChan is an open-source AI desktop robot built around M5Stack CoreS3. This repository contains the device firmware, a Flutter mobile app, an ESP-NOW remote, and a Go backend. The root README notes that repository contents may lag released firmware and apps; verify active manifests and component READMEs before relying on version claims.

## Architecture & Data Flow

- **Firmware (`firmware/`)**: ESP-IDF application. StackChan-specific components model the robot and its motion/avatar behavior, while the HAL provides board, display, network, BLE, WebSocket and service interfaces. Apps consume HAL events and update robot state. BLE/WebSocket callbacks generally signal or queue incoming data for app-level processing.
- **Mobile app (`app/`)**: Flutter UI and state communicate with the robot over BLE and with the backend over HTTP/WebSocket. `lib/util/native_bridge.dart` defines platform channels implemented in `android/` and `ios/`.
- **Backend (`server/`)**: GoFrame server. API definitions route through controllers to services, DAOs and models; middleware and boot packages handle shared request/startup behavior. WebSocket routes relay device/app messages.
- **Remote (`remote/`)**: Separate ESP-IDF firmware; use its own README and toolchain instructions.

Firmware build selection matters: `firmware/main/CMakeLists.txt` has an ESP32-S3 branch that registers a camera/face-detection example in `main.cpp` and returns early. The alternate source collection pulls in the StackChan components. Check the active target before assuming a source directory is included.

## Key Directories

- `firmware/main/stackchan/`: robot model, motion, avatar skins/decorators, modifiers, animation, JSON and utilities.
- `firmware/main/hal/`: hardware/service boundary, board adaptation, drivers and helpers.
- `firmware/main/apps/`: device UI apps and workers (launcher, setup, avatar, dance, AI agent, etc.).
- `app/lib/`: Flutter app; `view/` for screens/widgets, `model/` for data, `network/` for backend transport, and `util/` for BLE, native bridge and shared utilities.
- `server/api/`: request/response contracts; `server/internal/controller/`, `service/`, `dao/`, and `model/`: backend layers. `server/manifest/` contains deployment/configuration assets.
- `remote/code/`: remote-controller firmware.

## Development Commands

Run commands from the relevant component directory; there is no shared root build.

```bash
# Firmware dependency setup/build (firmware/); ESP-IDF 5.5.4
python3 ./fetch_repos.py
idf.py build
idf.py flash

# Flutter (app/)
flutter pub get
flutter analyze
flutter test
flutter run -d ios       # or: flutter run -d android

# Go backend (server/)
go mod download
go run main.go
go build -o stackchan-server main.go
make build               # GoFrame CLI build
```

`server/README.MD` also mentions `make run`, but no such target is defined in the checked-in Makefile/includes; use `go run main.go`. Run `python3 ./fetch_repos.py` from `firmware/` to fetch/update the pinned firmware dependencies before building. Remote instructions are in `remote/code/README.md` and use a different ESP-IDF version.

## Code Conventions & Common Patterns

- Keep changes scoped to the selected component and follow its local style and README; firmware CMake explicitly selects sources, so a new file may need registration or may be excluded by the selected target.
- Firmware apps use lifecycle callbacks such as `onOpen()`, `onRunning()`, and `onClose()`. Use the existing HAL/robot access patterns (`GetHAL()`, `GetStackChan()`); protect LVGL operations with the established `LvglLockGuard` pattern.
- Flutter uses Dart/Flutter conventions: `camelCase` members, `PascalCase` types, and `snake_case` JSON keys. State is managed through GetX (`app/lib/app_state.dart`); inspect existing controllers and reactive state before adding another mechanism.
- Backend code follows GoFrame's API/controller/service/DAO/model separation. Keep transport handlers thin and place domain operations in services.
- Prefer existing component patterns and configuration. Check target, platform-channel and protocol contracts when changing cross-component behavior.

## Important Files

- `README.md`: project purpose and component overview.
- `firmware/CMakeLists.txt`, `firmware/main/CMakeLists.txt`, `firmware/sdkconfig.defaults`, `firmware/main/Kconfig.projbuild`: firmware build/target configuration.
- `firmware/repos.json`, `firmware/patches/`: pinned external sources and local integration patches.
- `app/pubspec.yaml`, `app/analysis_options.yaml`: Flutter dependencies and analyzer rules.
- `server/go.mod`, `server/Makefile`, `server/hack/`: Go module and build tooling.
- `server/manifest/`: server runtime and deployment configuration.

## Runtime/Tooling Preferences

- Firmware: ESP-IDF 5.5.4 and its `idf.py`/CMake toolchain; dependency components are managed through ESP-IDF configuration and pinned repository metadata.
- App: Flutter/Dart with Pub. Use `app/pubspec.yaml` as the authority for SDK constraints (Dart `^3.11.5`), rather than older README minimum-version text.
- Server: Go modules and GoFrame. Use `server/go.mod` as the Go version authority (Go 1.26.2); README prerequisite text may be stale.
- Remote: separate ESP-IDF project; see `remote/code/README.md` for its version and platform-specific steps.

## Testing & QA

- Flutter tests use `flutter_test` under `app/test/`; run `flutter test` from `app/`. `flutter test --coverage` is available, and `flutter analyze` runs the configured Flutter lints.
- Go tests use the standard `testing` package; run `go test ./...` from `server/`. The current test set is sparse, so do not treat a passing run as broad behavioral coverage.
- No firmware application test suite or dedicated lint command is documented. For firmware changes, at minimum build the active target with `idf.py build`; for hardware-dependent behavior, verify on the applicable board when available.
- For fixes, identify a consumer-visible behavior and exercise the affected path; update relevant component documentation when commands or interfaces change.
