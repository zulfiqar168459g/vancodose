#!/bin/bash
# Publishes this project to YOUR PERSONAL GitHub account (asks you to confirm the account first).
cd "$(dirname "$0")" || exit 1
source "./scripts/common.sh"
printf '\033]0;Publish VancoDose to GitHub\007'
ensure_python || pause_and_exit 1
"$PY" "$ROOT/scripts/publish_github.py"
code=$?
pause_and_exit "$code"
