#!/bin/bash

# Script to run tests with a fresh backend instance
# Kills any existing backend process, starts a new one, and runs tests in parallel

echo "======== CyberWave Test Runner ========"

# Kill any existing backend process
echo "Stopping any existing backend processes..."
# Find and kill processes using port 8000 and 8001
lsof -ti:8000,8001 | xargs kill -9 2>/dev/null || true

# Create log directory if it doesn't exist
mkdir -p logs

# Use a different port (8001) for the backend to avoid conflicts
BACKEND_PORT=8001
export CYBERWAVE_BACKEND_URL="http://localhost:$BACKEND_PORT/api/v1"

# Start the backend server in background, redirecting output to a log file
echo "Starting backend server on port $BACKEND_PORT..."
cd ../cyberwave-backend
# The backend has main.py in the src directory
python -m uvicorn src.main:app --reload --port $BACKEND_PORT > ../cyberwave-sdk-python/logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../cyberwave-sdk-python

# Give the backend some time to start
echo "Waiting for backend to initialize (5 seconds)..."
sleep 5

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "ERROR: Backend failed to start. Check logs/backend.log for details."
    cat logs/backend.log
    exit 1
fi

# Run the tests in parallel, outputting to a log file
echo "Running tests in parallel with CYBERWAVE_BACKEND_URL=$CYBERWAVE_BACKEND_URL..."
python -m pytest tests -v > logs/tests.log 2>&1 &
TEST_PID=$!

# Display backend logs in real-time in one terminal
echo "Showing backend logs (Ctrl+C to stop)..."
tail -f logs/backend.log &
TAIL_PID1=$!

# Display test logs in real-time in another terminal
echo "Showing test logs (Ctrl+C to stop)..."
tail -f logs/tests.log &
TAIL_PID2=$!

# Wait for the tests to complete
echo "Waiting for tests to complete..."
wait $TEST_PID
TEST_EXIT_CODE=$?

# Kill the log viewers and backend
echo "Tests completed. Cleaning up..."
kill $TAIL_PID1 $TAIL_PID2 2>/dev/null || true
kill $BACKEND_PID 2>/dev/null || true
lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true

# Report results
echo "======== Test Results ========"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Some tests failed. See logs/tests.log for details."
fi

echo "Backend log: logs/backend.log"
echo "Test log: logs/tests.log"

exit $TEST_EXIT_CODE 