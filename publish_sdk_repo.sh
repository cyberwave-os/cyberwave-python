#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SDK_REPO_NAME="cyberwave-python"
SDK_USERNAME="cyberwave-os"
SDK_REPO_URL="git@github.com:${SDK_USERNAME}/${SDK_REPO_NAME}.git"  # GitHub repository URL
TEMP_DIR="/tmp/${SDK_REPO_NAME}"

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

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git."
    exit 1
fi

# Create temporary directory
print_message "Creating temporary directory..."
rm -rf ${TEMP_DIR}
mkdir -p ${TEMP_DIR}

# Copy SDK files
print_message "Copying SDK files..."
cp -r cyberwave-python/* ${TEMP_DIR}/
cp pyproject.toml ${TEMP_DIR}/
cp setup.py ${TEMP_DIR}/
cp setup.cfg ${TEMP_DIR}/
cp README.md ${TEMP_DIR}/

# Initialize git repository
print_message "Initializing git repository..."
cd ${TEMP_DIR}
git init

# Create .gitignore
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
env/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/
EOL

# Add files to git
print_message "Adding files to git..."
git add .
git commit -m "Initial SDK release"

# Rename master branch to main
print_message "Renaming master branch to main..."
git branch -M main

# Check if remote exists
if git ls-remote --exit-code ${SDK_REPO_URL} &> /dev/null; then
    print_message "Remote repository exists, pushing changes..."
    git remote add origin ${SDK_REPO_URL}
    git push -u origin main
else
    print_warning "Remote repository does not exist or is empty. Creating initial commit..."
    git remote add origin ${SDK_REPO_URL}
    git push -u origin main
fi

# Clean up
print_message "Cleaning up..."
cd -
rm -rf ${TEMP_DIR}

print_message "SDK repository setup complete!"
print_message "Next steps:"
print_message "1. Run ./publish_sdk.sh to publish to PyPI"
print_message "2. Create a GitHub release for version tracking" 