#!/usr/bin/env python3
"""VancoDose launcher: checks the computer, installs what is missing and starts the app.

Standard library only, so it runs before anything is installed. The platform launchers
(`Start VancoDose.command` on macOS/Linux, `Start VancoDose.bat` on Windows) first make sure
a suitable Python (3.12-3.14) exists, then run this file with it.

Options:  --check  verify and install only    --no-browser    --port N    --reinstall
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
LOGS = ROOT / "logs"
LOG_FILE = LOGS / "launcher.log"
SERVER_LOG = LOGS / "server.log"
PY_MIN, PY_MAX = (3, 12), (3, 14)
IS_WIN = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
MARKER = VENV / ".vancodose-requirements"
REQUIRED_FILES = ["vanco/__init__.py", "app/server.py", "app/static/index.html", "requirements.txt",
                  "data/patient_splits.csv",
                  "data/raw/vancomycin-precision-dosing-cohort-400-patients/vanco_sample_patients.parquet",
                  "data/raw/vancomycin-precision-dosing-cohort-400-patients/vanco_sample_dosing_regimens.parquet",
                  "data/raw/vancomycin-precision-dosing-cohort-400-patients/vanco_sample_pk_profiles.parquet"]
IMPORT_CHECK = "import numpy, pandas, pyarrow, scipy, sklearn, joblib, matplotlib, fastapi, uvicorn"
MODEL_CHECK = (
    "import json, sys, sklearn\n"
    "from vanco import config as C\n"
    "from vanco.predict import VancoPredictor\n"
    "VancoPredictor.load()\n"
    "v = json.loads(C.MODEL_CARD.read_text()).get('sklearn_version') if C.MODEL_CARD.exists() else None\n"
    "sys.exit(0 if v in (None, sklearn.__version__) else 4)\n"
)

for _stream in (sys.stdout, sys.stderr):  # never crash on ✓ or the logo when output is redirected (old code pages)
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
if IS_WIN:
    os.system("")  # enable ANSI colours in the Windows console
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
C_VIOLET, C_GREEN, C_AMBER, C_RED, C_DIM, C_OFF = (
    ("\033[38;2;132;75;255m", "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m") if USE_COLOR else ("",) * 6)


class LaunchError(Exception):
    """A problem the user can act on: message plus suggested fixes."""

    def __init__(self, message: str, fixes: list[str] | None = None, code: int = 1):
        super().__init__(message)
        self.fixes = fixes or []
        self.code = code


# ----------------------------------------------------------------------------- output
_ANSI = re.compile(r"\033\[[0-9;]*m")


def log(msg: str) -> None:
    LOGS.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {_ANSI.sub('', msg)}\n")


def say(msg: str = "", color: str = "") -> None:
    print(f"{color}{msg}{C_OFF if color else ''}", flush=True)
    log(msg)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n{C_VIOLET}[{n}/{total}]{C_OFF} {msg}")


def ok(msg: str) -> None:
    say(f"  {C_GREEN}✓{C_OFF} {msg}")


def run(cmd: list, *, capture=True, env_extra: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1",
           "PYTHONUTF8": "1", **(env_extra or {})}
    log("$ " + " ".join(str(c) for c in cmd))
    res = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, text=True, timeout=timeout,
                         stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    if capture and res.stdout:
        log(res.stdout[-8000:])
    return res


def banner() -> None:
    print(f"""
{C_VIOLET}   ██╗   ██╗{C_OFF}   VancoDose
{C_VIOLET}   ╚██╗ ██╔╝{C_OFF}   Vancomycin precision dosing
{C_VIOLET}    ╚████╔╝ {C_OFF}   {C_DIM}Research prototype on synthetic data. Not for clinical use.{C_OFF}
{C_VIOLET}     ╚═══╝  {C_OFF}""")
    log(f"===== launcher start ({platform.platform()}, Python {platform.python_version()}, {sys.executable})")


# ----------------------------------------------------------------------------- checks
def check_system() -> None:
    v = sys.version_info[:2]
    ok(f"{platform.system()} {platform.release()} ({platform.machine()}), Python {platform.python_version()}")
    if not PY_MIN <= v <= PY_MAX:
        raise LaunchError(f"Python {v[0]}.{v[1]} is not supported; VancoDose needs Python 3.12 to 3.14.",
                          ["Run the VancoDose launcher again; it installs Python 3.12 automatically.",
                           "Or install Python 3.12 from https://www.python.org/downloads/ and retry."], code=3)
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing:
        raise LaunchError("Some project files are missing: " + ", ".join(missing),
                          ["Download the complete project again (including the data folder) and retry."])
    ok("Project files present")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 1.0 and not VENV_PY.exists():
        raise LaunchError(f"Only {free_gb:.1f} GB of disk space is free; installation needs about 1 GB.",
                          ["Free some disk space and run the launcher again."])
    ok(f"{free_gb:.0f} GB disk space free")


def venv_python_ok() -> bool:
    if not VENV_PY.exists():
        return False
    try:
        res = subprocess.run([str(VENV_PY), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if res.returncode != 0:
        return False
    major, minor = map(int, res.stdout.strip().split("."))
    return PY_MIN <= (major, minor) <= PY_MAX


def ensure_venv(reinstall: bool) -> None:
    if reinstall or not venv_python_ok():
        if VENV.exists():
            say("  Rebuilding the Python environment as requested." if reinstall
                else "  The existing Python environment is outdated or damaged; rebuilding it.", C_AMBER)
            shutil.rmtree(VENV, ignore_errors=True)
        say("  Creating a private Python environment for VancoDose (one time)…")
        try:
            venv.EnvBuilder(with_pip=True, clear=True, upgrade_deps=False).create(VENV)
        except Exception as e:  # noqa: BLE001
            fixes = ["On Ubuntu/Debian run: sudo apt install python3-venv  (then retry)"] if sys.platform.startswith("linux") else []
            raise LaunchError(f"Could not create the Python environment: {e}", fixes + [
                "Make sure the project folder is not read-only (for example inside a zip file or a protected folder)."])
        if not venv_python_ok():
            raise LaunchError("The new Python environment does not start.", ["Delete the .venv folder and run the launcher again."])
    ok("Python environment ready (.venv)")


def requirements_hash() -> str:
    h = hashlib.sha256((ROOT / "requirements.txt").read_bytes())
    h.update(platform.python_version().encode())
    return h.hexdigest()


def network_ok(host="pypi.org", port=443, timeout=6) -> bool:
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        return True  # cannot test through a proxy cheaply; let pip try
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pip_install(req: str) -> bool:
    say(f"  Installing packages from {req} (first run takes a few minutes)…")
    res = run([VENV_PY, "-m", "pip", "install", "--prefer-binary", "--progress-bar", "off", "-r", ROOT / req],
              capture=True, timeout=3600)
    return res.returncode == 0


def ensure_packages(reinstall: bool) -> None:
    want = requirements_hash()
    have = MARKER.read_text().strip() if MARKER.exists() else ""
    if have == want and not reinstall and run([VENV_PY, "-c", IMPORT_CHECK]).returncode == 0:
        ok("All packages already installed")
        return
    if not network_ok():
        raise LaunchError("No internet connection to the Python package index (pypi.org).",
                          ["Connect to the internet (or your organisation's VPN) and run the launcher again.",
                           "Behind a corporate proxy? Set HTTPS_PROXY before starting, or ask IT to allow pypi.org."])
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"], timeout=600)
    if not pip_install("requirements.txt"):
        say("  Exact versions could not be installed on this computer; trying compatible versions instead.", C_AMBER)
        if not pip_install("requirements-compat.txt"):
            raise LaunchError("Package installation failed.",
                              ["Check your internet connection and run the launcher again.",
                               f"Technical details are in {LOG_FILE.relative_to(ROOT)}."])
    res = run([VENV_PY, "-c", IMPORT_CHECK])
    if res.returncode != 0:
        raise LaunchError("Packages were installed but cannot be loaded.",
                          ["Run the launcher with --reinstall to rebuild the environment.",
                           f"Technical details are in {LOG_FILE.relative_to(ROOT)}."])
    MARKER.write_text(want)
    ok("Packages installed")


def ensure_model() -> None:
    res = run([VENV_PY, "-c", MODEL_CHECK])
    if res.returncode == 0:
        ok("Prediction model ready")
        return
    say("  Preparing the prediction model for this computer (about 30 seconds)…")
    res = run([VENV_PY, "-m", "vanco.train"], timeout=1800)
    if res.returncode != 0 or run([VENV_PY, "-c", MODEL_CHECK]).returncode != 0:
        raise LaunchError("The prediction model could not be prepared.",
                          [f"Technical details are in {LOG_FILE.relative_to(ROOT)}.",
                           "Run the launcher with --reinstall to rebuild the environment."])
    ok("Prediction model trained and saved")


# ----------------------------------------------------------------------------- server
def health(port: int, timeout=1.5) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start_server(preferred: int, open_browser: bool) -> int:
    for p in range(preferred, preferred + 11):
        h = health(p, timeout=0.5)
        if h and h.get("app") == "VancoDose":
            url = f"http://127.0.0.1:{p}/"
            say(f"\n  VancoDose is already running at {url}", C_GREEN)
            if open_browser:
                webbrowser.open(url)
            return 0
    port = next((p for p in range(preferred, preferred + 11) if port_free(p)), None)
    if port is None:
        raise LaunchError(f"Ports {preferred}–{preferred + 10} are all in use by other programs.",
                          ["Close other local web servers, or start with --port 8500."])
    if port != preferred:
        say(f"  Port {preferred} is busy; using port {port} instead.", C_AMBER)
    LOGS.mkdir(exist_ok=True)
    out = SERVER_LOG.open("w", encoding="utf-8")
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"}
    proc = subprocess.Popen([str(VENV_PY), "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(120):
        if proc.poll() is not None:
            out.close()
            tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-1500:]
            log(tail)
            raise LaunchError("The VancoDose server stopped while starting.\n" + tail,
                              [f"Full details are in {SERVER_LOG.relative_to(ROOT)}.", "Run the launcher with --reinstall."])
        if health(port):
            break
        time.sleep(0.5)
    else:
        proc.terminate()
        raise LaunchError("The VancoDose server did not respond within 60 seconds.",
                          [f"Details are in {SERVER_LOG.relative_to(ROOT)}.", "Restart the computer and try again."])
    say(f"\n  {C_GREEN}VancoDose is running at {url}{C_OFF}")
    say("  Keep this window open while you use VancoDose. Close it, or press Ctrl+C, to stop.")
    if open_browser:
        webbrowser.open(url)

    def _stop(*_):  # closing the window or a system stop behaves like Ctrl+C
        raise KeyboardInterrupt
    for sig in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), _stop)
            except (ValueError, OSError):
                pass
    try:
        return proc.wait()
    except KeyboardInterrupt:
        say("\n  Stopping VancoDose…")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        say("  Stopped. You can close this window.")
        return 0
    finally:
        out.close()


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Start VancoDose")
    ap.add_argument("--check", action="store_true", help="check and install only; do not start the app")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reinstall", action="store_true", help="rebuild the Python environment from scratch")
    a = ap.parse_args(argv)
    banner()
    try:
        step(1, 4, "Checking this computer")
        check_system()
        step(2, 4, "Preparing Python")
        ensure_venv(a.reinstall)
        ensure_packages(a.reinstall)
        step(3, 4, "Preparing the prediction model")
        ensure_model()
        if a.check:
            say("\n  Everything is installed and ready.", C_GREEN)
            return 0
        step(4, 4, "Starting VancoDose")
        return start_server(a.port, not a.no_browser)
    except LaunchError as e:
        say(f"\n  {C_RED}✗ {e}{C_OFF}")
        for f in e.fixes:
            say(f"    • {f}")
        say(f"\n  {C_DIM}Log: {LOG_FILE}{C_OFF}")
        return e.code
    except KeyboardInterrupt:
        say("\n  Cancelled.")
        return 130
    except Exception as e:  # noqa: BLE001 - last-resort message instead of a stack trace
        log(repr(e))
        import traceback
        log(traceback.format_exc())
        say(f"\n  {C_RED}✗ Unexpected problem: {e}{C_OFF}")
        say(f"    • Details are in {LOG_FILE}. Run the launcher again, or with --reinstall.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
