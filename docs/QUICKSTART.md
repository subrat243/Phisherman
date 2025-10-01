# 🚀 Quick Start Guide

Get up and running with the Auto Phishing Detection Tool in 5 minutes!

## ⚡ Fast Track Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Phisherman.git
cd Phisherman

# Run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download required data
- ✅ Create necessary directories
- ✅ Set up configuration files

**Note**: If installation fails with memory issues, install packages manually:

```bash
# Install in smaller batches to avoid memory issues
pip install --break-system-packages numpy pandas scikit-learn joblib
pip install --break-system-packages fastapi uvicorn requests beautifulsoup4
pip install --break-system-packages tldextract python-whois dnspython
pip install --break-system-packages nltk colorama tqdm loguru
pip install --break-system-packages lxml cryptography pyOpenSSL

# Optional ML frameworks (requires more memory)
pip install --break-system-packages xgboost lightgbm
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

Or if using the system Python:

```bash
# No virtual environment needed if installed with --break-system-packages
python3 detect.py -u https://example.com
```

### 3. Choose Your Path

#### 🎯 Option A: Quick Test (No Training Required)

Test the tool immediately using rule-based detection:

```bash
# Test a single URL
python detect.py -u https://google.com

# Interactive mode
python detect.py -i

# Batch analysis
echo "https://google.com" > test_urls.txt
echo "http://suspicious-site.tk" >> test_urls.txt
python detect.py -f test_urls.txt
```

#### 🧠 Option B: Full ML-Powered Detection

Train ML models for enhanced accuracy (requires xgboost and lightgbm):

```bash
# First, ensure ML libraries are installed
pip install --break-system-packages xgboost lightgbm

# Collect data and train models (takes 5-10 minutes)
python train.py --collect-data --models random_forest xgboost ensemble

# Use the trained model
python detect.py -u https://example.com -m models/best_model.pkl
```

**Note**: If you can't install xgboost/lightgbm due to memory constraints, use Option A (rule-based detection) which works without training.

#### 🌐 Option C: API Server

Start the REST API for integration:

```bash
# Start the API server
python api/main.py

# In another terminal, test it
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}'
```

Access API documentation at: http://localhost:8000/docs

#### 🔌 Option D: Browser Extension

Install the Chrome/Edge extension:

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `browser_extension` folder
5. Done! The shield icon will appear in your toolbar

**Note:** Make sure the API is running (Option C) for the extension to work with ML models.

## 📖 Common Usage Examples

### CLI Tool

```bash
# Analyze URL with verbose output
python detect.py -u https://paypal-verify.com -v

# Check extracted features only
python detect.py -u https://example.com --check-features

# Batch analysis with JSON output
python detect.py -f urls.txt -o results.json

# Interactive mode for continuous checking
python detect.py -i
```

### Python API

```python
from src.phishing_detector import PhishingDetector

# Initialize detector
detector = PhishingDetector()

# Analyze URL
result = detector.predict_url('https://suspicious-site.com')
print(f"Is Phishing: {result['is_phishing']}")
print(f"Risk Score: {result['risk_score']:.1f}/100")
print(f"Warnings: {len(result['warnings'])}")

# With trained model
detector = PhishingDetector(
    model_path='models/best_model.pkl',
    scaler_path='models/best_model_scaler.pkl',
    feature_names_path='models/best_model_features.json'
)

result = detector.predict_url('https://example.com')
```

### REST API

```bash
# Check single URL
curl -X POST "http://localhost:8000/api/v1/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Batch check
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://google.com", "http://phishing-site.tk"]}'

# Comprehensive analysis
curl -X POST "http://localhost:8000/api/v1/analyze/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://suspicious-site.com"}'

# Get statistics
curl "http://localhost:8000/api/v1/stats"
```

## 🎓 Training Your Own Model

### Quick Training

```bash
# Train with default settings
python train.py --collect-data
```

### Advanced Training

```bash
# Train specific models with tuning
python train.py \
  --collect-data \
  --models random_forest xgboost lightgbm ensemble \
  --tune \
  --tune-model xgboost \
  --cross-validate \
  --cv-folds 5
```

### Using Your Own Data

```bash
# Prepare your CSV with columns: url, label (0=safe, 1=phishing)
# Then extract features
python -c "
from src.data_collector import PhishingDataCollector
import pandas as pd

collector = PhishingDataCollector()
df = pd.read_csv('my_urls.csv')
processed = collector.prepare_training_data(df)
processed.to_csv('data/processed/my_training_data.csv', index=False)
"

# Train on your data
python train.py --data-file data/processed/my_training_data.csv
```

## 🔧 Troubleshooting

### Issue: Module Not Found

```bash
# Make sure virtual environment is activated (if using venv)
source venv/bin/activate

# Reinstall dependencies with --break-system-packages
pip install -r requirements.txt --break-system-packages

# Or install specific missing module
pip install --break-system-packages <module-name>
```

### Issue: API Won't Start

```bash
# Check if port 8000 is available
lsof -i :8000

# Use different port
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### Issue: Model Training Fails

```bash
# Check data directory
ls -la data/processed/

# Ensure ML libraries are installed
pip install --break-system-packages xgboost lightgbm

# Try with smaller dataset first
python train.py --collect-data --models random_forest

# If still fails, use rule-based detection (no training needed)
python detect.py -u https://example.com
```

### Issue: Browser Extension Not Working

1. Make sure API is running: `python api/main.py`
2. Check extension settings - click extension icon → Settings
3. Verify API URL is set to: `http://localhost:8000/api/v1`
4. Check browser console for errors (F12 → Console)

## 📊 Understanding Results

### Risk Score Scale

- **0-20**: Safe ✅
- **20-40**: Low Risk ⚠️
- **40-60**: Medium Risk ⚠️
- **60-80**: High Risk 🚨
- **80-100**: Critical Risk 🚨

### Classification Levels

- **SAFE**: Website appears legitimate
- **LOW**: Minor suspicious indicators
- **MEDIUM**: Multiple suspicious patterns
- **HIGH**: Strong phishing indicators
- **CRITICAL**: Definite phishing attempt

## 🎯 Next Steps

1. **Explore Features**: Try different URLs and see what features are extracted
   ```bash
   python detect.py -u https://example.com --check-features
   ```

2. **Train Better Models**: Add more training data to improve accuracy
   ```bash
   python train.py --collect-data --tune --cross-validate
   ```

3. **Integrate with Your App**: Use the REST API in your applications
   ```bash
   # Start API
   python api/main.py

   # Test integration
   curl http://localhost:8000/docs
   ```

4. **Deploy Browser Extension**: Install on all browsers for real-time protection

5. **Monitor Performance**: Check API statistics and model metrics
   ```bash
   curl http://localhost:8000/api/v1/stats
   curl http://localhost:8000/api/v1/model/info
   ```

## 📚 More Resources

- **Full Documentation**: [README.md](README.md)
- **API Documentation**: http://localhost:8000/docs (when API is running)
- **Training Guide**: See [train.py --help](train.py)
- **Configuration**: Edit `.env` file for custom settings

## 💡 Tips

1. **Performance**: Use `--models ensemble` for best accuracy
2. **Speed**: Use `random_forest` for fastest predictions
3. **Memory**: If training fails, reduce dataset size or use `lightgbm`
4. **Installation**: Use `--break-system-packages` flag for pip on some systems
5. **No Training**: Rule-based detection works immediately without ML models
6. **API**: Enable CORS in production for web integrations
7. **Security**: Use HTTPS in production deployments

## ⚠️ Important Notes

- Rule-based detection works immediately but ML models are more accurate
- Use `pip install --break-system-packages` on systems with externally-managed Python
- Training requires xgboost and lightgbm libraries (optional)
- Training requires at least 1000 samples for good results
- Browser extension needs API running for ML predictions
- Always verify suspicious detections manually
- Keep training data updated with new phishing patterns
- If memory is limited, use rule-based detection without training

## 🆘 Need Help?

- Check [README.md](README.md) for detailed documentation
- Run `python detect.py --help` for CLI options
- Run `python train.py --help` for training options
- Open an issue on GitHub for bugs/features

---

**Ready to protect against phishing? Start detecting now! 🛡️**
