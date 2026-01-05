#!/bin/bash

# Integration Test Runner for Cyberwave Python SDK
# This script helps run integration tests with proper setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Cyberwave Python SDK - Integration Test Runner${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Install dependencies (this installs the local package in editable mode)
echo -e "${BLUE}Installing dependencies...${NC}"
poetry install
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Check if backend URL is set
if [ -z "$CYBERWAVE_BASE_URL" ]; then
    echo -e "${YELLOW}⚠️  CYBERWAVE_BASE_URL not set, using default: http://localhost:8000${NC}"
    export CYBERWAVE_BASE_URL="http://localhost:8000"
fi

# Check if authentication is set
if [ -z "$CYBERWAVE_API_KEY" ] && [ -z "$CYBERWAVE_TOKEN" ]; then
    echo -e "${RED}❌ Error: Neither CYBERWAVE_API_KEY nor CYBERWAVE_TOKEN is set${NC}"
    echo ""
    echo "Please set one of the following environment variables:"
    echo "  export CYBERWAVE_API_KEY='your_api_key'"
    echo "  export CYBERWAVE_TOKEN='your_bearer_token'"
    echo ""
    echo "To get credentials:"
    echo "  1. Log in to your Cyberwave instance"
    echo "  2. Navigate to Settings → API Keys"
    echo "  3. Create and copy your API key"
    exit 1
fi

echo -e "${GREEN}✓${NC} Backend URL: ${BLUE}$CYBERWAVE_BASE_URL${NC}"
if [ -n "$CYBERWAVE_API_KEY" ]; then
    echo -e "${GREEN}✓${NC} Authentication: ${BLUE}API Key (set)${NC}"
else
    echo -e "${GREEN}✓${NC} Authentication: ${BLUE}Bearer Token (set)${NC}"
fi
echo ""

# Check if backend is running
echo -e "${BLUE}Checking backend connectivity...${NC}"
if curl -s -f -o /dev/null "$CYBERWAVE_BASE_URL/health" || curl -s -f -o /dev/null "$CYBERWAVE_BASE_URL/api/health" || curl -s -f -o /dev/null "$CYBERWAVE_BASE_URL"; then
    echo -e "${GREEN}✓${NC} Backend is reachable at $CYBERWAVE_BASE_URL"
else
    echo -e "${YELLOW}⚠️  Warning: Could not reach backend at $CYBERWAVE_BASE_URL${NC}"
    echo "   Make sure the backend is running:"
    echo "     cd cyberwave-backend"
    echo "     docker-compose -f local.yml up"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Error: Poetry is not installed${NC}"
    echo "Install poetry: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo -e "${GREEN}✓${NC} Poetry is installed"
echo ""

# Install dependencies if needed
echo -e "${BLUE}Checking dependencies...${NC}"
if [ ! -d ".venv" ] && [ ! -f "poetry.lock" ]; then
    echo "Installing dependencies..."
    poetry install
else
    echo -e "${GREEN}✓${NC} Dependencies already installed"
fi
echo ""

# Parse command line arguments
TEST_ARGS="tests/test_integration.py"
PYTEST_ARGS="-v -s"

if [ "$1" == "quick" ]; then
    echo -e "${BLUE}Running quick tests (read-only)...${NC}"
    TEST_ARGS="tests/test_integration.py::TestIntegrationReadOperations"
elif [ "$1" == "workflow" ]; then
    echo -e "${BLUE}Running complete workflow test...${NC}"
    TEST_ARGS="tests/test_integration.py::TestIntegrationWorkflow"
elif [ "$1" == "errors" ]; then
    echo -e "${BLUE}Running error handling tests...${NC}"
    TEST_ARGS="tests/test_integration.py::TestIntegrationErrorHandling"
elif [ "$1" == "context" ]; then
    echo -e "${BLUE}Running context manager test...${NC}"
    TEST_ARGS="tests/test_integration.py::TestIntegrationWithContextManager"
elif [ "$1" == "all" ] || [ -z "$1" ]; then
    echo -e "${BLUE}Running all integration tests...${NC}"
else
    echo -e "${YELLOW}Unknown test type: $1${NC}"
    echo "Usage: $0 [quick|workflow|errors|context|all]"
    echo "  quick    - Run read-only tests (fastest)"
    echo "  workflow - Run complete user workflow test"
    echo "  errors   - Run error handling tests"
    echo "  context  - Run context manager test"
    echo "  all      - Run all tests (default)"
    exit 1
fi
echo ""

# Run the tests
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Running Tests${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Run with poetry
poetry run pytest $TEST_ARGS $PYTEST_ARGS

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${BLUE}================================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed (exit code: $EXIT_CODE)${NC}"
fi
echo -e "${BLUE}================================================================${NC}"

exit $EXIT_CODE

