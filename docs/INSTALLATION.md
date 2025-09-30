# 🚀 Installation Guide

Complete installation instructions for the Auto Phishing Detection Tool.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **4GB RAM minimum** (8GB recommended for training)
- **Internet connection** (for package installation and feature extraction)

Check your Python version:
```bash
python3 --version
```

---

## 🎯 Quick Installation

### Method 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Auto-phishing-detect-tool.git
cd Auto-phishing-detect-tool

# Run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:
- ✅ Create virtual environment (optional)
- ✅ Install all dependencies with `--break-system-packages`
- ✅ Download required data
- ✅ Create necessary directories
- ✅ Configure the environment

---

### Method 2: Manual Installation

#### Step 1: Install Core Dependencies

```bash
# Install core ML libraries
pip install --break-system-packages numpy pandas scikit-learn joblib

# Install web/API frameworks
pip install --break-system-packages fastapi uvicorn requests beautifulsoup4

# Install feature extraction libraries
pip install --break-system-packages tldextract python-whois dnspython validators

# Install utilities
pip install --break-system-packages python-dotenv pyyaml nltk colorama tqdm loguru

# Install security libraries
pip install --break-system-packages cryptography pyOpenSSL certifi aiohttp lxml

# Install email analysis
pip install --break-system-packages email-validator
```

#### Step 2: Install Optional ML Frameworks

**⚠️ Warning**: These packages require significant memory during installation.

```bash
# Try installing XGBoost and LightGBM (optional for ML training)
pip install --break-system-packages xgboost lightgbm
```

**If installation fails or gets killed**: Skip this step. You can still use rule-based detection without these libraries Or Install dependencies one by one manually.

#### Step 3: Download NLTK Data

```bash
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

#### Step 4: Create Required Directories

```bash
mkdir -p data/raw data/processed models logs
```

---

## 🐛 Common Installation Issues

### Issue 1: `externally-managed-environment` Error

**Error:**
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution:**
Always use the `--break-system-packages` flag:

```bash
pip install --break-system-packages package-name
```

Or use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Issue 2: Installation Gets "Killed"

**Error:**
```
Killed
```

**Cause:** System running out of memory during package compilation.

**Solutions:**

**Option A: Install in Smaller Batches**
```bash
# Install one at a time
pip install --break-system-packages numpy
pip install --break-system-packages pandas
pip install --break-system-packages scikit-learn
# etc.
```

**Option B: Skip Heavy Packages**
```bash
# Skip xgboost and lightgbm
# The tool works with rule-based detection without these
```

**Option C: Increase Swap Space** (Linux)
```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Option D: Use Pre-built Wheels**
```bash
pip install --break-system-packages --only-binary=:all: xgboost
```

---

### Issue 3: `ModuleNotFoundError` After Installation

**Error:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**

**If you need the module:**
```bash
pip install --break-system-packages xgboost
```

**If you don't need ML training:**
```bash
# Just use rule-based detection (works without xgboost/lightgbm)
python3 detect.py -u https://example.com
```

---

### Issue 4: `hashlib-sha3` Not Found

**Error:**
```
ERROR: No matching distribution found for hashlib-sha3
```

**Solution:**
This package is not needed (built into Python 3.6+). It has been removed from the requirements. If you see this error, update your requirements.txt:

```bash
# Remove the line with hashlib-sha3 from requirements.txt
# Or use the updated requirements.txt from the repository
```

---

## 🔧 Installation Verification

### Test 1: Check Imports

```bash
python3 -c "
import numpy
import pandas
import sklearn
print('✓ Core libraries OK')
"
```

### Test 2: Test Feature Extraction

```bash
python3 -c "
from src.feature_extractor import URLFeatureExtractor
extractor = URLFeatureExtractor()
features = extractor.extract_all_features('https://google.com')
print(f'✓ Extracted {len(features)} features')
"
```

### Test 3: Test Detection

```bash
python3 detect.py -u https://google.com
```

Expected output:
```
======================================================================
🛡️  Phisherman - AI-POWERED PHISHING DETECTION TOOL
======================================================================
...
✅ SAFE
Risk Score: 0.0/100
...
```

### Test 4: Test API (Optional)

```bash
# Start API in one terminal
python3 api/main.py

# Test in another terminal
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "timestamp": "...",
  "model_loaded": false,
  "version": "1.0.0"
}
```

---

## 🎯 What Can I Use Without Full Installation?

### Without ANY ML Libraries (xgboost, lightgbm)

✅ **Rule-based detection** - Works immediately!
```bash
python3 detect.py -u https://example.com
```

✅ **CLI tool** - All features except training
```bash
python3 detect.py -i  # Interactive mode
python3 detect.py -f urls.txt  # Batch processing
```

✅ **API server** - Uses rule-based detection
```bash
python3 api/main.py
```

✅ **Browser extension** - Full functionality with rule-based detection

❌ **Model training** - Requires xgboost and/or lightgbm
```bash
# This will fail without ML libraries
python3 train.py --collect-data
```

---

## 📦 Minimal Installation (Rule-Based Detection Only)

If you only want rule-based detection without ML training:

```bash
# Absolute minimum dependencies
pip install --break-system-packages \
    numpy pandas \
    requests beautifulsoup4 \
    tldextract python-whois dnspython \
    fastapi uvicorn \
    colorama tqdm

# Test it works
python3 detect.py -u https://example.com
```

---

## 🌟 Full Installation (With ML Training)

For complete functionality including ML model training:

```bash
# Install everything
pip install --break-system-packages -r requirements.txt

# Or if that fails, install in order:
pip install --break-system-packages numpy pandas scikit-learn
pip install --break-system-packages fastapi uvicorn requests beautifulsoup4
pip install --break-system-packages tldextract python-whois dnspython validators
pip install --break-system-packages nltk colorama tqdm loguru
pip install --break-system-packages lxml cryptography pyOpenSSL certifi
pip install --break-system-packages email-validator python-dotenv pyyaml
pip install --break-system-packages aiohttp joblib matplotlib seaborn
pip install --break-system-packages xgboost lightgbm
```

---

## 🔄 Installing Updates

```bash
# Update the repository
git pull origin main

# Reinstall dependencies (in case of updates)
pip install --break-system-packages -r requirements.txt

# Or just update specific packages
pip install --break-system-packages --upgrade package-name
```

---

## 🐳 Docker Installation (Alternative)

If you have Docker installed:

```bash
# Build the image
docker build -t phishing-detector .

# Run the container
docker run -p 8000:8000 phishing-detector

# Or use docker-compose
docker-compose up
```

**Note:** Docker files are not included by default. You'll need to create:
- `Dockerfile`
- `docker-compose.yml`

---

## 🍎 Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Install development headers (if compiling packages)
sudo apt-get install python3-dev build-essential

# Continue with pip installation
pip install --break-system-packages -r requirements.txt
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Continue with pip installation
pip3 install --break-system-packages -r requirements.txt
```

### Windows

```bash
# Install Python from python.org (3.8+)
# Make sure to check "Add Python to PATH"

# Open Command Prompt or PowerShell
pip install -r requirements.txt

# Note: --break-system-packages flag not needed on Windows
```

---

## 🧹 Uninstallation

### Remove Packages Only

```bash
pip uninstall -y -r requirements.txt
```

### Remove Everything

```bash
# Remove virtual environment (if used)
rm -rf venv/

# Remove cached files
rm -rf src/__pycache__/
rm -rf api/__pycache__/
rm -rf tests/__pycache__/
rm -rf .pytest_cache/

# Remove models
rm -rf models/*.pkl

# Remove data
rm -rf data/raw/* data/processed/*

# Remove logs
rm -rf logs/*.log
```

---

## ✅ Post-Installation Checklist

- [ ] Python 3.8+ installed and working
- [ ] Core dependencies installed (numpy, pandas, scikit-learn)
- [ ] Web/API dependencies installed (fastapi, uvicorn, requests)
- [ ] Feature extraction libraries installed (tldextract, dnspython)
- [ ] CLI tool works: `python3 detect.py -u https://google.com`
- [ ] API starts: `python3 api/main.py`
- [ ] Browser extension loads (optional)
- [ ] ML libraries installed (optional, for training)

---

## 🆘 Still Having Issues?

1. **Check Troubleshooting Guide**: See `TROUBLESHOOTING.md`
2. **Check Python Version**: Must be 3.8 or higher
3. **Check Available Memory**: At least 2GB free RAM
4. **Check Disk Space**: At least 2GB free space
5. **Try Minimal Installation**: Install only core packages
6. **Open an Issue**: Provide error messages and system info

---

## 📚 Next Steps

After successful installation:

1. **Quick Test**: `python3 detect.py -u https://example.com`
2. **Read Quick Start**: See `QUICKSTART.md`
3. **Read Documentation**: See `README.md`
4. **Train Models** (optional): `python3 train.py --collect-data`
5. **Start API**: `python3 api/main.py`
6. **Install Browser Extension**: Load from `browser_extension/` folder

---

**Installation complete! Start protecting against phishing now! 🛡️**