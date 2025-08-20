#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # Should be project root
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/.venv"
BACKEND_PATH="$PROJECT_ROOT/cyberwave-backend"
SDK_PATH="$PROJECT_ROOT/cyberwave-sdk-python"
SDK_TEST_SCRIPT="$SDK_PATH/test_sdk_frictionless.py"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8000"
HEALTH_CHECK_URL="http://localhost:$SERVER_PORT/health"
STARTUP_TIMEOUT=15 # Increased slightly just in case

# --- Activate Virtual Environment ---
echo "Activating virtual environment at $VENV_PATH..."
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please ensure it's created in the project root ($PROJECT_ROOT) using Python 3.11+."
    exit 1
fi
echo "Python executable: $(which python)"
PYTHON_CMD="$VENV_PATH/bin/python" # Use direct path

# --- Cleanup function ---
cleanup() {
  echo "Cleaning up..."
  if [ ! -z "$SERVER_PID" ]; then
    echo "Stopping background server (PID: $SERVER_PID)..."
    # Kill the process started by python -m uvicorn
    # Use kill directly on the PID captured
    kill $SERVER_PID || echo "Server process $SERVER_PID not found or already stopped."
    wait $SERVER_PID 2>/dev/null # Wait briefly to allow cleanup
  fi
}

# Set trap to run cleanup function on exit or interrupt
trap cleanup EXIT SIGINT SIGTERM

# --- Start Backend Server in Background ---
echo "Starting CyberWave Backend server in background from $BACKEND_PATH..."
cd "$BACKEND_PATH" # Need to be in backend dir for src imports
"$PYTHON_CMD" -m uvicorn src.main:app --host "$SERVER_HOST" --port "$SERVER_PORT" & 
SERVER_PID=$!
cd "$PROJECT_ROOT" # Go back to project root
echo "Server started with PID: $SERVER_PID"

# --- Wait for Server Startup ---
echo "Waiting for server to become healthy at $HEALTH_CHECK_URL (max ${STARTUP_TIMEOUT}s)..."
SECONDS=0
while true; do
  if curl -sf -o /dev/null "$HEALTH_CHECK_URL"; then
    echo "Server is healthy!"
    break
  fi
  if [ "$SECONDS" -ge "$STARTUP_TIMEOUT" ]; then
    echo "ERROR: Server did not start within $STARTUP_TIMEOUT seconds."
    exit 1 # Cleanup will run via trap
  fi
  sleep 1
  SECONDS=$((SECONDS + 1))
done

# --- Run SDK Test Script ---
echo "Running SDK test script: $SDK_TEST_SCRIPT..."
if [ ! -f "$SDK_TEST_SCRIPT" ]; then
    echo "ERROR: SDK test script not found at $SDK_TEST_SCRIPT"
    exit 1
fi

# Ensure SDK package is importable (if not installed, being in root helps)
"$PYTHON_CMD" "$SDK_TEST_SCRIPT" "$@" # Pass any script args
TEST_EXIT_CODE=$?

# --- Exit ---
echo "SDK Test script finished with exit code: $TEST_EXIT_CODE"
# Cleanup will be handled by the trap
exit $TEST_EXIT_CODE 