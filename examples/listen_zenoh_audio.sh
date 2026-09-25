#!/usr/bin/env bash
# Listen to microphone audio on the local Zenoh bus (Raspberry Pi friendly).
#
# Usage:
#   ./examples/listen_zenoh_audio.sh
#   ./examples/listen_zenoh_audio.sh --play
#
# Optional env overrides:
#   CYBERWAVE_TWIN_UUID   (auto-detected from cyberwave-driver-* container if unset)
#   ZENOH_CONNECT         (optional; leave unset for P2P — matches edge drivers)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="$ROOT/.venv/bin/python"
VENV_PIP="$ROOT/.venv/bin/pip"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creating virtualenv at $ROOT/.venv ..."
  python3 -m venv .venv
fi

if ! "$VENV_PYTHON" -c "import zenoh" 2>/dev/null; then
  echo "Installing cyberwave[zenoh] into .venv (do not use system pip) ..."
  "$VENV_PIP" install -q -U pip
  "$VENV_PIP" install -q -e '.[zenoh]'
fi

if [[ -z "${CYBERWAVE_TWIN_UUID:-}" || "${CYBERWAVE_TWIN_UUID}" == *"your"* ]]; then
  DRIVER="$(docker ps --format '{{.Names}}' 2>/dev/null | grep '^cyberwave-driver-' | head -1 || true)"
  if [[ -n "$DRIVER" ]]; then
    CYBERWAVE_TWIN_UUID="$(docker inspect "$DRIVER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
      | sed -n 's/^CYBERWAVE_TWIN_UUID=//p' | head -1)"
    export CYBERWAVE_TWIN_UUID
    echo "Using twin UUID from $DRIVER: $CYBERWAVE_TWIN_UUID"
  fi
fi

if [[ -z "${CYBERWAVE_TWIN_UUID:-}" ]]; then
  echo "error: set CYBERWAVE_TWIN_UUID or start a cyberwave-driver-* container" >&2
  exit 1
fi

export CYBERWAVE_DATA_BACKEND="${CYBERWAVE_DATA_BACKEND:-zenoh}"

# Default sensor id matches twins whose microphone sensor is named "audio".
exec "$VENV_PYTHON" -u examples/listen_zenoh_audio.py --sensor audio "$@"
