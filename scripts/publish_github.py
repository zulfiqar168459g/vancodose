#!/usr/bin/env python3
"""Publish VancoDose to a PERSONAL GitHub account.

Safe by design:
* Uses its own GitHub sign-in stored in .tools/gh-config (never your existing GitHub CLI login),
  so a business account that is already signed in on this computer cannot be used by accident.
* Shows the signed-in account and asks you to confirm it before creating anything.
* Commits with your GitHub "noreply" address, so no work e-mail ends up in the history.
* Never force-pushes.

Run through "Publish to GitHub.command" (macOS) or:  python scripts/publish_github.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / ".tools"
GH_CONFIG = TOOLS / "gh-config"
DEFAULT_NAME = "vancodose"
DESCRIPTION = "VancoDose: explainable vancomycin exposure prediction and precision dosing (research prototype, synthetic data)"
VISIBILITY = "public"
TRAILER = ""
ENV = {**os.environ, "GH_CONFIG_DIR": str(GH_CONFIG), "GIT_TERMINAL_PROMPT": "0"}
for _k in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GH_PROMPT_DISABLED", "GH_HOST"):
    ENV.pop(_k, None)


def say(msg=""):
    print(msg, flush=True)


def stop(msg, hint=""):
    say(f"\n✗ {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def ask(question, default=""):
    try:
        ans = input(f"{question}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def yes(question) -> bool:
    return ask(f"{question} (y/n)", "n").lower().startswith("y")


def run(cmd, check=True, capture=True, env=None, **kw):
    res = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env or ENV, text=True,
                         stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None, **kw)
    if check and res.returncode != 0:
        stop(f"Command failed: {' '.join(str(c) for c in cmd[:3])}…", (res.stderr or res.stdout or "").strip()[-600:])
    return res


# ----------------------------------------------------------------------------- tools
def ensure_git() -> str:
    git = shutil.which("git")
    if git and subprocess.run([git, "--version"], capture_output=True).returncode == 0:
        return git
    if sys.platform == "darwin":
        subprocess.run(["xcode-select", "--install"])
        stop("Git is not installed yet. macOS is now offering to install the Command Line Tools.",
             "Click Install, wait for it to finish, then run this publisher again.")
    if os.name == "nt":
        stop("Git is not installed.", "Run:  winget install --id Git.Git -e   (or download from https://git-scm.com), then retry.")
    stop("Git is not installed.", "Install git with your package manager, then retry.")


def gh_asset_name(version: str) -> str:
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "amd64"
    if sys.platform == "darwin":
        return f"gh_{version}_macOS_{arch}.zip"
    if os.name == "nt":
        return f"gh_{version}_windows_{arch}.zip"
    return f"gh_{version}_linux_{arch}.tar.gz"


def ensure_gh() -> Path:
    """GitHub CLI: use the copy in .tools, else download the official release (checksum verified)."""
    exe = "gh.exe" if os.name == "nt" else "gh"
    found = list((TOOLS / "gh").rglob(exe)) if (TOOLS / "gh").exists() else []
    if found:
        return found[0]
    system = shutil.which("gh")
    if system:
        return Path(system)
    say("Downloading the GitHub command-line tool (one time)…")
    with urllib.request.urlopen("https://api.github.com/repos/cli/cli/releases/latest", timeout=30) as r:
        rel = json.load(r)
    version = rel["tag_name"].lstrip("v")
    name = gh_asset_name(version)
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    if name not in assets:
        stop(f"No GitHub CLI download for this computer ({name}).", "Install it from https://cli.github.com and retry.")
    data = urllib.request.urlopen(assets[name], timeout=120).read()
    sums = urllib.request.urlopen(assets[f"gh_{version}_checksums.txt"], timeout=30).read().decode()
    expected = next((l.split()[0] for l in sums.splitlines() if l.endswith(name)), None)
    if hashlib.sha256(data).hexdigest() != expected:
        stop("The GitHub CLI download failed its checksum; nothing was installed.")
    dest = TOOLS / "gh"
    dest.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(dest, filter="data")
    path = next(dest.rglob(exe))
    path.chmod(0o755)
    return path


# ----------------------------------------------------------------------------- account
def signed_in_user(gh) -> dict | None:
    res = run([gh, "api", "user"], check=False)
    return json.loads(res.stdout) if res.returncode == 0 else None


def sign_in(gh) -> dict:
    while True:
        user = signed_in_user(gh)
        if not user:
            say("\nA browser window will open to sign in to GitHub.")
            say("➜ Sign in with your PERSONAL GitHub account, not the business one.")
            say("  If the browser is already signed in to the business account, click your avatar → Sign out first,")
            say("  or use a private/incognito window and paste the code shown below.\n")
            # Any "set up git" step gh offers writes to an isolated file, never your global git settings.
            isolated = {**ENV, "GIT_CONFIG_GLOBAL": str(TOOLS / "gitconfig-isolated")}
            res = run([gh, "auth", "login", "--hostname", "github.com", "--git-protocol", "https", "--web",
                       "--scopes", "repo"], check=False, capture=False, env=isolated)
            if res.returncode != 0:
                stop("GitHub sign-in did not complete.", "Run the publisher again when you are ready.")
            user = signed_in_user(gh)
            if not user:
                stop("Could not read the signed-in GitHub account.")
        say(f"\nSigned in to GitHub as: {user['login']}" + (f" ({user['name']})" if user.get("name") else ""))
        say(f"  Profile: {user['html_url']}")
        if yes("Is this your PERSONAL account?"):
            return user
        run([gh, "auth", "logout", "--hostname", "github.com"], check=False, input="Y\n")
        say("Signed out. Let's sign in with the personal account instead.")


# ----------------------------------------------------------------------------- repository
def prepare_repo(git, user) -> None:
    if not (ROOT / ".git").exists():
        run([git, "init", "-b", "main"])
    run([git, "config", "--local", "user.name", user.get("name") or user["login"]])
    run([git, "config", "--local", "user.email", f"{user['id']}+{user['login']}@users.noreply.github.com"])
    run([git, "config", "--local", "core.autocrlf", "false"])
    run([git, "add", "-A"])
    run([git, "update-index", "--chmod=+x", "Start VancoDose.command", "Publish to GitHub.command"], check=False)
    files = run([git, "ls-files", "-s"]).stdout.splitlines()
    sizes = []
    for line in run([git, "ls-files"]).stdout.splitlines():
        p = ROOT / line
        if p.is_file():
            sizes.append((p.stat().st_size, line))
    total = sum(s for s, _ in sizes) / 1e6
    big = [(s, f) for s, f in sizes if s > 50e6]
    say(f"\nReady to publish {len(files)} files ({total:.1f} MB):")
    tops = sorted({f.split('/')[0] for _, f in sizes})
    say("  " + ", ".join(tops))
    for s, f in sorted(sizes, reverse=True)[:5]:
        say(f"  largest: {f} ({s / 1e6:.1f} MB)")
    if big:
        stop("Some files are larger than GitHub allows (100 MB) or recommends (50 MB): " + ", ".join(f for _, f in big))


def commit(git) -> None:
    if run([git, "diff", "--cached", "--quiet"], check=False).returncode == 0 and \
            run([git, "rev-parse", "--verify", "HEAD"], check=False).returncode == 0:
        say("No new changes to commit.")
        return
    first = run([git, "rev-parse", "--verify", "HEAD"], check=False).returncode != 0
    msg = ("VancoDose: vancomycin exposure prediction and precision dosing" if first else "Update VancoDose") + TRAILER
    run([git, "commit", "-q", "-m", msg])


def main():
    say("VancoDose → GitHub publisher\n")
    git = ensure_git()
    gh = ensure_gh()
    user = sign_in(gh)
    name = ask("Repository name", DEFAULT_NAME)
    full = f"{user['login']}/{name}"
    exists = run([gh, "repo", "view", full, "--json", "url"], check=False).returncode == 0
    prepare_repo(git, user)
    say(f"\nDestination: https://github.com/{full}  ({'existing repository' if exists else f'new {VISIBILITY} repository'})")
    if not exists:
        say(f"Visibility: {VISIBILITY.upper()} — anyone on the internet will be able to see these files.")
    if not yes("Publish now?"):
        stop("Nothing was published.")
    commit(git)
    if not exists:
        run([gh, "repo", "create", full, f"--{VISIBILITY}", "--description", DESCRIPTION])
    remote = f"https://github.com/{full}.git"
    if run([git, "remote", "get-url", "origin"], check=False).returncode == 0:
        run([git, "remote", "set-url", "origin", remote])
    else:
        run([git, "remote", "add", "origin", remote])
    # Repository-local credential helper: uses this publisher's own sign-in, nothing global changes.
    helper = f'!GH_CONFIG_DIR="{GH_CONFIG.as_posix()}" "{gh.as_posix()}" auth git-credential'
    run([git, "config", "--local", "--replace-all", "credential.https://github.com.helper", ""])
    run([git, "config", "--local", "--add", "credential.https://github.com.helper", helper])
    say("Uploading…")
    res = run([git, "push", "-u", "origin", "main"], check=False)
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        if "rejected" in err or "fetch first" in err:
            stop("GitHub refused the upload because the repository already has different history.",
                 "Choose a new repository name, or merge the online changes first. Nothing was overwritten.")
        stop("The upload failed.", err[-600:])
    url = f"https://github.com/{full}"
    say(f"\n✓ Published: {url}")
    say(f"\nGitHub is now testing the launchers on fresh Windows, macOS and Linux machines (about 15 minutes).")
    say(f"Results: {url}/actions   A green tick on the repository page means every check passed.")
    webbrowser.open(url + "/actions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop("Cancelled. Nothing more was published.")
    except urllib.error.URLError as e:  # type: ignore[attr-defined]
        stop(f"Network problem: {e}", "Check your internet connection and try again.")
