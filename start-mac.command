#!/bin/bash
# Start from Finder or run: bash start-mac.command
set -u
umask 077

SCRIPT_PATH=${BASH_SOURCE[0]}
case "$SCRIPT_PATH" in
  */*) SCRIPT_PARENT=${SCRIPT_PATH%/*} ;;
  *) SCRIPT_PARENT=. ;;
esac
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_PARENT" && pwd -P) || exit 1
cd -- "$PROJECT_DIR" || exit 1

fail() {
  printf '\n%s\n' "$1" >&2
  exit 1
}
python_help() {
  printf '\nPython 3.10 or newer is required. Install Python 3 from:\nhttps://www.python.org/downloads/macos/\nThen open a new Terminal and run this launcher again.\n' >&2
  exit 1
}

printf '\nFasolati Cube — local experiment lab\n'
if [ ! -f sim/cube/lab_server.py ] || [ ! -f web/cube-lab.html ]; then
  fail 'The Cube package is incomplete. Extract the entire ZIP, then run start-mac.command inside that folder.'
fi
command -v python3 >/dev/null 2>&1 || python_help
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 || python_help

VENV_DIR="$PROJECT_DIR/.venv-cube"
VENV_PY="$VENV_DIR/bin/python"
if [ ! -d "$VENV_DIR" ]; then
  printf 'Preparing a private Python environment in this project…\n'
  python3 -m venv "$VENV_DIR" || fail 'Could not create the Python environment. Install Python from python.org, then retry. If .venv-cube was left incomplete, remove only that folder before retrying.'
fi
[ -x "$VENV_PY" ] || fail 'The .venv-cube environment is incomplete. Remove only .venv-cube from this project, then run this launcher again.'
"$VENV_PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 || fail 'The .venv-cube environment uses an older or unavailable Python. Remove only .venv-cube from this project, then run this launcher again.'

if ! "$VENV_PY" -c 'import numpy; raise SystemExit(0 if numpy.lib.NumpyVersion(numpy.__version__) >= "1.26.0" else 1)' >/dev/null 2>&1; then
  printf 'Installing NumPy in the project environment (internet required)…\n'
  "$VENV_PY" -m pip install --disable-pip-version-check 'numpy>=1.26' || fail 'NumPy installation failed. Check your internet connection and rerun this launcher; no system Python packages were changed.'
  "$VENV_PY" -c 'import numpy; raise SystemExit(0 if numpy.lib.NumpyVersion(numpy.__version__) >= "1.26.0" else 1)' >/dev/null 2>&1 || fail 'NumPy could not be loaded after installation. Check the installation error above.'
fi

printf '\nOpen http://127.0.0.1:8765/journey after the server starts.\n'
printf 'Enter your Meta API key at the hidden prompt. It stays in this process.\n'
printf 'Keep this Terminal open. Press Control-C to stop the lab.\n\n'
if "$VENV_PY" -m sim.cube.lab_server --meta; then
  exit 0
else
  SERVER_STATUS=$?
  if [ "$SERVER_STATUS" -eq 130 ]; then
    printf '\nCube Lab stopped.\n'
  else
    printf '\nCube Lab could not stay running. Check the error above.\nIf port 8765 is already in use, an earlier Cube Lab may still be running.\n' >&2
  fi
  exit "$SERVER_STATUS"
fi
