#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    print_error "twine is not installed. Please install it using: pip install twine"
    exit 1
fi

# Check if build is installed
if ! command -v python -m build &> /dev/null; then
    print_error "build is not installed. Please install it using: pip install build"
    exit 1
fi

# Clean previous builds
print_message "Cleaning previous builds..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/

# Build the package
print_message "Building package..."
python -m build

# Check if build was successful
if [ ! -d "dist" ]; then
    print_error "Build failed. dist directory not found."
    exit 1
fi

# Verify the package
print_message "Verifying package..."
twine check dist/*

# Ask for confirmation before uploading
read -p "Do you want to upload to PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Upload cancelled by user."
    exit 0
fi

# Upload to PyPI
print_message "Uploading to PyPI..."
twine upload dist/*

print_message "Package successfully published to PyPI!" 