#!/bin/bash

# Setup Script for Auto Phishing Detection Tool
# This script sets up the environment and installs all dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Banner
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🛡️  AUTO PHISHING DETECTION TOOL - SETUP 🛡️          ║
║                                                           ║
║           AI-Powered Phishing Protection System          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check Python version
print_header "Checking System Requirements"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

# Check pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed"
    echo "Please install pip3"
    exit 1
fi
print_success "pip3 found"

# Create virtual environment
print_header "Creating Virtual Environment"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        print_success "Virtual environment recreated"
    else
        print_info "Using existing virtual environment"
    fi
else
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_header "Upgrading pip"
pip install --upgrade pip setuptools wheel
print_success "pip upgraded"

# Install dependencies
print_header "Installing Python Dependencies"
print_info "This may take several minutes..."

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "All dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Download NLTK data
print_header "Downloading NLTK Data"
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: Could not download NLTK data: {e}')
"
print_success "NLTK data downloaded"

# Create necessary directories
print_header "Creating Project Directories"

directories=(
    "data/raw"
    "data/processed"
    "models"
    "logs"
    "results"
    "test_emails"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    else
        print_info "Directory already exists: $dir"
    fi
done

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep

# Create .env file if it doesn't exist
print_header "Creating Configuration Files"

if [ ! -f ".env" ]; then
    cat > .env << 'EOL'
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=models/best_model.pkl
SCALER_PATH=models/best_model_scaler.pkl
FEATURES_PATH=models/best_model_features.json

# Detection Thresholds
RISK_THRESHOLD=60
BLOCK_THRESHOLD=80

# Cache Settings
CACHE_DURATION=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/phishing_detector.log
EOL
    print_success ".env file created"
else
    print_warning ".env file already exists (skipping)"
fi

# Test imports
print_header "Testing Installation"

python3 << 'EOF'
import sys

def test_imports():
    packages = [
        'numpy',
        'pandas',
        'sklearn',
        'xgboost',
        'lightgbm',
        'fastapi',
        'requests',
        'beautifulsoup4',
        'tldextract',
    ]

    failed = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            failed.append(package)

    if failed:
        print(f"Failed to import: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All core packages imported successfully!")

test_imports()
EOF

if [ $? -eq 0 ]; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    exit 1
fi

# Create sample URLs file for testing
print_header "Creating Sample Files"

cat > urls_sample.txt << 'EOL'
https://www.google.com
https://www.microsoft.com
https://www.github.com
EOL
print_success "Created urls_sample.txt"

# Display completion message
print_header "Setup Complete!"

echo -e "${GREEN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║              ✅ Installation Successful! ✅                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "\n${BLUE}Next Steps:${NC}\n"
echo "1. Activate the virtual environment:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}\n"

echo "2. Collect data and train models:"
echo -e "   ${YELLOW}python train.py --collect-data${NC}\n"

echo "3. Start the API server:"
echo -e "   ${YELLOW}python api/main.py${NC}\n"

echo "4. Use the CLI tool:"
echo -e "   ${YELLOW}python detect.py -u https://example.com${NC}\n"

echo "5. Or try interactive mode:"
echo -e "   ${YELLOW}python detect.py -i${NC}\n"

echo -e "${BLUE}Additional Resources:${NC}\n"
echo "• Documentation: README.md"
echo "• API Docs: http://localhost:8000/docs (after starting API)"
echo "• Browser Extension: browser_extension/ directory"

echo -e "\n${GREEN}Happy Phishing Detection! 🛡️${NC}\n"

# Deactivate virtual environment
deactivate

echo -e "${YELLOW}Note: Virtual environment has been deactivated.${NC}"
echo -e "${YELLOW}Remember to activate it before running the tools:${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}\n"
