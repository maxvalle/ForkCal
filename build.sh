#!/bin/bash
set -e

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Darwin) PLATFORM="macos" ;;
  Linux)  PLATFORM="linux" ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

echo "Building for $PLATFORM..."

# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build
if [ "$PLATFORM" = "macos" ]; then
  python -m nuitka \
    --standalone \
    --macos-create-app-bundle \
    --enable-plugin=pyside6 \
    --follow-imports \
    --include-package=PySide6 \
    --macos-app-icon=macOS/icon.icns \
    --macos-app-name=ForkCal \
    --output-dir=dist \
    forkcal.py

  # 4. Patch Info.plist with mic usage description
  /usr/libexec/PlistBuddy -c \
    "Add :NSMicrophoneUsageDescription string 'This app requires microphone access for audio input'" \
    dist/ForkCal.app/Contents/Info.plist

  # 5. Sign with entitlements (ad-hoc)
  codesign --deep --force --sign - \
    --entitlements macOS/entitlements.plist \
    dist/ForkCal.app

else
  python -m nuitka \
    --onefile \
    --enable-plugin=pyside6 \
    --follow-imports \
    --include-package=PySide6 \
    --output-filename=ForkCal \
    --output-dir=dist \
    forkcal.py
fi

# 6. Deactivate virtual environment
deactivate

# 7. Clean up
rm -rf venv
rm -rf __pycache__

# 8. Done
if [ "$PLATFORM" = "macos" ]; then
  echo "Done! Check dist/ForkCal.app"
else
  echo "Done! Check dist/ForkCal"
fi