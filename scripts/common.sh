# Shared helpers for the macOS/Linux launchers (sourced, not run directly).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OS="$(uname -s)"; ARCH="$(uname -m)"
PY_SPEC="3.12.10"
VIOLET=$'\033[38;2;132;75;255m'; RED=$'\033[31m'; AMBER=$'\033[33m'; OFF=$'\033[0m'

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "${AMBER}$*${OFF}"; }
fail() { printf '\n%s\n' "${RED}✗ $*${OFF}"; }
pause_and_exit() { printf '\n'; read -r -p "Press Return to close this window. " _ 2>/dev/null </dev/tty || true; exit "${1:-1}"; }

set_icon() {  # give a launcher file the VancoDose icon in Finder (macOS, best effort)
  [[ "$OS" == "Darwin" && -f "$ROOT/assets/vancodose.png" ]] || return 0
  osascript -l JavaScript -e "ObjC.import('AppKit');
    var img = \$.NSImage.alloc.initWithContentsOfFile('$ROOT/assets/vancodose.png');
    \$.NSWorkspace.sharedWorkspace.setIconForFileOptions(img, '$1', 0);" >/dev/null 2>&1 || true
}

python_ok() {  # Python 3.12-3.14, 64-bit
  "$1" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32 else 1)' >/dev/null 2>&1
}

find_python() {
  local c p
  for c in python3.12 python3.13 python3.14 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.13 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 python3; do
    p="$(command -v "$c" 2>/dev/null)" || continue
    # /usr/bin/python3 on a Mac without developer tools only opens an install dialog: skip it
    if [[ "$OS" == "Darwin" && "$p" == "/usr/bin/python3" ]] && ! xcode-select -p >/dev/null 2>&1; then continue; fi
    if python_ok "$p"; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

install_python() {
  if [[ "$OS" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      say "Installing Python 3.12 with Homebrew…"
      brew install python@3.12 && return 0
      warn "Homebrew could not install Python; trying the official installer instead."
    fi
    local tmp url="https://www.python.org/ftp/python/${PY_SPEC}/python-${PY_SPEC}-macos11.pkg"
    tmp="$(mktemp -d)"
    say "Downloading Python ${PY_SPEC} from python.org…"
    curl -fL --retry 3 --progress-bar -o "$tmp/python.pkg" "$url" || { fail "The download failed. Check your internet connection."; return 1; }
    if ! pkgutil --check-signature "$tmp/python.pkg" 2>/dev/null | grep -q "Python Software Foundation"; then
      fail "The downloaded installer is not signed by the Python Software Foundation; it was not installed."; return 1
    fi
    say "macOS will now ask for your password to install Python (this happens only once)."
    osascript -e "do shell script \"installer -pkg '$tmp/python.pkg' -target /\" with administrator privileges" >/dev/null \
      || { fail "Python installation was cancelled or failed."; return 1; }
    rm -rf "$tmp"
    return 0
  fi
  fail "Python 3.12 or newer is required."
  say "Install it with your package manager, for example:"
  say "  Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
  say "  Fedora:         sudo dnf install python3.12"
  return 1
}

ensure_python() {  # sets $PY
  PY="$(find_python)" && return 0
  warn "Python 3.12 or newer was not found on this computer. Installing it now…"
  install_python || return 1
  hash -r
  PY="$(find_python)" && return 0
  fail "Python was installed but could not be found. Please restart the computer and run this launcher again."
  return 1
}
