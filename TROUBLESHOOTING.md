# 🔧 Troubleshooting Guide

Common issues and their solutions for the Auto Phishing Detection Tool.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Module/Import Errors](#moduleimport-errors)
- [Training Issues](#training-issues)
- [API Issues](#api-issues)
- [Browser Extension Issues](#browser-extension-issues)
- [Performance Issues](#performance-issues)
- [SSL/Certificate Errors](#sslcertificate-errors)
- [Memory Issues](#memory-issues)

---

## Installation Issues

### Issue: `externally-managed-environment` Error

**Symptom:**
```
error: externally-managed-environment
```

**Solution:**
Use the `--break-system-packages` flag:

```bash
pip install -r requirements.txt --break-system-packages
```

Or use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Issue: Installation Gets "Killed"

**Symptom:**
```
Killed
```

**Cause:** System running out of memory during package installation.

**Solution:** Install packages in smaller batches:

```bash
# Install core dependencies
pip install --break-system-packages numpy pandas scikit-learn joblib

# Install web/API frameworks
pip install --break-system-packages fastapi uvicorn requests beautifulsoup4

# Install feature extraction
pip install --break-system-packages tldextract python-whois dnspython

# Install utilities
pip install --break-system-packages nltk colorama tqdm loguru

# Install ML frameworks (optional, may fail on low memory)
pip install --break-system-packages xgboost lightgbm
```

**Alternative:** Use rule-based detection without ML libraries:
```bash
python detect.py -u https://example.com
# Works without xgboost/lightgbm
```

---

### Issue: `hashlib-sha3` Package Not Found

**Symptom:**
```
ERROR: No matching distribution found for hashlib-sha3
```

**Solution:**
This package has been removed from requirements.txt as it's built into Python 3.6+. Update your requirements.txt or ignore this error.

---

## Module/Import Errors

### Issue: `ModuleNotFoundError: No module named 'xgboost'`

**Symptom:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution 1:** Install the missing module:
```bash
pip install --break-system-packages xgboost
```

**Solution 2:** Use rule-based detection (no training):
```bash
python detect.py -u https://example.com
# Works without ML libraries
```

**Solution 3:** Train with only scikit-learn models:
```bash
python train.py --collect-data --models random_forest
```

---

### Issue: `ImportError: cannot import name 'XXX'`

**Solution:**
Ensure all dependencies are installed:
```bash
pip install --break-system-packages -r requirements.txt
```

Check Python version (requires 3.8+):
```bash
python --version
```

---

## Training Issues

### Issue: Training Script Fails with Memory Error

**Symptom:**
```
MemoryError or Killed
```

**Solution:**

1. **Reduce dataset size:**
```python
# Edit src/data_collector.py
# Limit number of URLs collected
```

2. **Use lightweight model:**
```bash
python train.py --collect-data --models random_forest
```

3. **Skip training entirely:**
```bash
# Use rule-based detection
python detect.py -u https://example.com
```

---

### Issue: `No module named 'xgboost'` During Training

**Solution:**
Install ML libraries or skip them:

```bash
# Try installing
pip install --break-system-packages xgboost lightgbm

# Or train without them
python train.py --collect-data --models random_forest logistic_regression
```

---

### Issue: Training Takes Too Long

**Solution:**

1. **Reduce dataset size**
2. **Disable hyperparameter tuning:**
```bash
python train.py --collect-data --models random_forest
# Don't use --tune flag
```

3. **Use faster models:**
```bash
python train.py --collect-data --models logistic_regression
```

---

## API Issues

### Issue: API Won't Start

**Symptom:**
```
Address already in use
```

**Solution:**

Check if port 8000 is in use:
```bash
lsof -i :8000
# Kill process using port
kill -9 <PID>
```

Or use a different port:
```bash
python api/main.py --port 8080
# Or
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

---

### Issue: API Returns 500 Internal Server Error

**Solution:**

1. **Check logs:**
```bash
# Terminal output shows errors
```

2. **Verify model files exist:**
```bash
ls -la models/
# Should show best_model.pkl, scaler, features
```

3. **Test without model:**
```bash
# API uses rule-based detection if model not found
```

4. **Check dependencies:**
```bash
pip list | grep -E "(fastapi|uvicorn|pydantic)"
```

---

### Issue: CORS Errors in Browser

**Solution:**

The API already has CORS enabled. If still seeing errors:

1. **Edit `api/main.py`:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Restart API server**

---

## Browser Extension Issues

### Issue: Extension Not Loading

**Solution:**

1. **Check Chrome extensions page:**
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `browser_extension` folder

2. **Check for errors:**
   - Click "Errors" button on extension card
   - Fix any JavaScript errors

3. **Verify manifest.json:**
   - Ensure it's valid JSON
   - Check all file paths exist

---

### Issue: Extension Shows "API Connection Failed"

**Solution:**

1. **Start the API server:**
```bash
python api/main.py
```

2. **Check API URL in extension settings:**
   - Click extension icon → Settings
   - Verify URL is `http://localhost:8000/api/v1`

3. **Test API manually:**
```bash
curl http://localhost:8000/health
```

---

### Issue: Extension Not Checking URLs

**Solution:**

1. **Check if protection is enabled:**
   - Click extension icon
   - Ensure toggle is ON

2. **Check permissions:**
   - Extension needs `<all_urls>` permission
   - Review and accept permissions

3. **Check browser console:**
   - F12 → Console tab
   - Look for error messages

---

## Performance Issues

### Issue: Slow URL Analysis (>10 seconds)

**Symptom:**
```
⏱️  Analysis completed in 10.269 seconds
```

**Causes:**
- Network timeouts (WHOIS, DNS, SSL checks)
- Slow internet connection
- Remote servers not responding

**Solution:**

1. **Reduce timeout:**
```python
# Edit src/feature_extractor.py
self.timeout = 2  # Reduce from 5 to 2 seconds
```

2. **Skip network-heavy features:**
```python
# Comment out WHOIS checks in extract_all_features()
# features.update(self._extract_whois_features(url))
```

3. **Use cached results:**
```python
# API automatically caches results for 1 hour
```

---

### Issue: High Memory Usage

**Solution:**

1. **Reduce batch size:**
```bash
python detect.py -f urls.txt
# Process fewer URLs at once
```

2. **Use lighter models:**
```bash
python train.py --models random_forest
# Avoid ensemble models
```

3. **Clear cache:**
```python
# Browser extension cache grows over time
# Click extension → Settings → Clear Cache
```

---

## SSL/Certificate Errors

### Issue: "Invalid or missing SSL certificate" for Legitimate Sites

**Symptom:**
```
⚠️  Warnings (1):
  1. ⚠️ Invalid or missing SSL certificate
```

**Cause:** 
- Network issues
- Firewall blocking SSL connections
- Certificate validation timeout

**Solution:**

This has been fixed in the latest version. The tool now:
- Treats SSL connection failures as "uncertain" (0.5) not "invalid" (0.0)
- Only warns about SSL if definitely invalid
- Doesn't heavily penalize temporary SSL issues

Update your code:
```bash
git pull origin main
# Or manually update feature_extractor.py and phishing_detector.py
```

---

### Issue: SSL Timeout Errors

**Solution:**

1. **Increase timeout:**
```python
# Edit src/feature_extractor.py
self.timeout = 10  # Increase timeout
```

2. **Skip SSL checks:**
```python
# Comment out SSL feature extraction
# features.update(self._extract_ssl_features(url))
```

---

## Memory Issues

### Issue: "MemoryError" During Training

**Solution:**

1. **Use smaller dataset:**
```python
# Edit train.py
df = df.sample(1000)  # Use only 1000 samples
```

2. **Train one model at a time:**
```bash
python train.py --models random_forest
python train.py --models xgboost
```

3. **Increase swap space:**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Issue: System Freezes During pip install

**Solution:**

1. **Install packages one at a time:**
```bash
pip install --break-system-packages numpy
pip install --break-system-packages pandas
# etc.
```

2. **Use pre-built wheels:**
```bash
pip install --break-system-packages --only-binary=:all: xgboost
```

3. **Skip large packages:**
```bash
# Don't install xgboost, lightgbm if not needed
# Use rule-based detection instead
```

---

## General Debugging

### Enable Debug Mode

**For CLI:**
```bash
python detect.py -u https://example.com -v
```

**For API:**
```python
# Edit api/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**For Browser Extension:**
```
1. Open Chrome DevTools (F12)
2. Go to Console tab
3. Check for error messages
```

---

### Check System Requirements

```bash
# Python version (needs 3.8+)
python --version

# Available memory
free -h

# Disk space
df -h

# Installed packages
pip list
```

---

### Reset Everything

If all else fails, start fresh:

```bash
# Remove virtual environment
rm -rf venv/

# Remove cached files
rm -rf src/__pycache__/
rm -rf api/__pycache__/
rm -rf tests/__pycache__/

# Remove models
rm -rf models/*.pkl

# Reinstall
./setup.sh
```

---

## Getting Help

If you're still stuck:

1. **Check Documentation:**
   - README.md
   - QUICKSTART.md
   - API docs: http://localhost:8000/docs

2. **Check Logs:**
   - Terminal output
   - Browser console (F12)
   - API logs

3. **Test Components Individually:**
   ```bash
   # Test feature extraction
   python -c "from src.feature_extractor import URLFeatureExtractor; e = URLFeatureExtractor(); print(len(e.extract_all_features('https://google.com')))"
   
   # Test detector
   python -c "from src.phishing_detector import PhishingDetector; d = PhishingDetector(); print(d.predict_url('https://google.com'))"
   ```

4. **Open an Issue:**
   - Provide error message
   - Include Python version
   - Describe what you tried
   - Include relevant logs

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Installation fails | Use `--break-system-packages` flag |
| Module not found | `pip install --break-system-packages <module>` |
| Training fails | Use rule-based detection: `python detect.py -u URL` |
| API won't start | Change port: `uvicorn api.main:app --port 8080` |
| Extension not working | Start API: `python api/main.py` |
| Slow analysis | Reduce timeout in `feature_extractor.py` |
| Memory error | Install packages one at a time |
| SSL warnings | Update to latest code (fixed) |

---

**Need more help?** Check the full documentation in README.md or open an issue on GitHub.