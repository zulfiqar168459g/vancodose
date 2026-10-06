#!/bin/bash
# VancoDose launcher for macOS (also works on Linux: run ./Start\ VancoDose.command).
# Double-click it. It installs anything missing (Python, packages, model) and opens VancoDose.
cd "$(dirname "$0")" || exit 1
source "./scripts/common.sh"
printf '\033]0;VancoDose\007'
set_icon "$ROOT/Start VancoDose.command"
set_icon "$ROOT/Publish to GitHub.command"

if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
  fail "This launcher is for macOS and Linux. On Windows, double-click 'Start VancoDose.bat'."
  pause_and_exit 1
fi
ensure_python || pause_and_exit 1
"$PY" "$ROOT/scripts/launcher.py" "$@"
code=$?
[[ $code -eq 0 || $code -eq 130 ]] || pause_and_exit "$code"
exit 0
