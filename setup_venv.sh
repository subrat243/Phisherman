#!/bin/bash

# Phisherman Virtual Environment Setup Script
# This script creates an isolated Python environment to avoid dependency conflicts

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements-core.txt"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Phisherman Virtual Environment Setup            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if Python 3 is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Found Python ${PYTHON_VERSION}"

# Check Python version (require 3.8+)
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    print_error "Python 3.8 or higher is required. You have Python ${PYTHON_VERSION}"
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at: ${VENV_DIR}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        print_success "Removed old virtual environment"
    else
        print_status "Using existing virtual environment"
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip, setuptools, and wheel
print_status "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Package managers upgraded"

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "Requirements file not found: ${REQUIREMENTS_FILE}"
    print_warning "Using fallback: requirements.txt"
    REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"

    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "No requirements file found!"
        exit 1
    fi
fi

# Install dependencies
print_status "Installing dependencies from $(basename $REQUIREMENTS_FILE)..."
print_warning "This may take a few minutes..."
echo ""

pip install -r "$REQUIREMENTS_FILE"

echo ""
print_success "Dependencies installed successfully"

# Download NLTK data
print_status "Downloading NLTK data..."
python3 << 'PYTHON_SCRIPT'
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded")
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")
PYTHON_SCRIPT
print_success "NLTK data configured"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/cache"
print_success "Project directories created"

# Test installation
print_status "Testing installation..."
python3 << 'PYTHON_SCRIPT'
import sys
errors = []

packages = [
    'numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm',
    'fastapi', 'uvicorn', 'requests', 'beautifulsoup4',
    'tldextract', 'cryptography'
]

for package in packages:
    try:
        __import__(package)
    except ImportError as e:
        errors.append(f"{package}: {e}")

if errors:
    print("Errors found:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("All core packages imported successfully")
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Create activation helper script
ACTIVATE_SCRIPT="${PROJECT_DIR}/activate.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Quick activation script for Phisherman virtual environment

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROJECT_DIR}/venv/bin/activate"

echo "Phisherman virtual environment activated"
echo "Python: $(which python3)"
echo ""
echo "Available commands:"
echo "  python3 detect.py -u <URL>        - Analyze a URL"
echo "  python3 api/main.py               - Start API server"
echo "  python3 train_quick_ml.py         - Train ML model"
echo "  deactivate                        - Exit virtual environment"
echo ""
EOF
chmod +x "$ACTIVATE_SCRIPT"
print_success "Created activation helper: ${ACTIVATE_SCRIPT}"

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Setup Complete Successfully!                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Virtual Environment:${NC} ${VENV_DIR}"
echo -e "${BLUE}Python Version:${NC} ${PYTHON_VERSION}"
echo ""
echo -e "${YELLOW}To use Phisherman:${NC}"
echo ""
echo "  1. Activate the virtual environment:"
echo -e "     ${GREEN}source venv/bin/activate${NC}"
echo "     or"
echo -e "     ${GREEN}source activate.sh${NC}"
echo ""
echo "  2. Run commands:"
echo -e "     ${GREEN}python3 detect.py -u https://example.com${NC}"
echo -e "     ${GREEN}python3 api/main.py${NC}"
echo ""
echo "  3. Deactivate when done:"
echo -e "     ${GREEN}deactivate${NC}"
echo ""
echo -e "${YELLOW}Quick Test:${NC}"
echo -e "  ${GREEN}source venv/bin/activate && python3 detect.py -u https://google.com${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  - README.md"
echo "  - DEPENDENCY_RESOLUTION.md"
echo "  - INSTALLATION.md"
echo ""

# Deactivate for now
deactivate

print_success "Setup script completed!"
echo ""
echo -e "${YELLOW}Note:${NC} Remember to activate the virtual environment before using Phisherman"
echo -e "      Run: ${GREEN}source venv/bin/activate${NC}"
