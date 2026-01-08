# Datasets Directory

This directory is for storing phishing detection datasets.

## Supported Formats

- **CSV** - Comma-separated values
- **JSON** - JavaScript Object Notation
- **TXT** - Plain text (one URL per line)

## Usage

### Import Dataset
```bash
python train.py --dataset datasets/your_dataset.csv
```

### Expected Format

**CSV:**
```csv
url,label
https://google.com,0
http://phishing-site.com,1
```

**JSON:**
```json
[
  {"url": "https://google.com", "label": 0},
  {"url": "http://phishing-site.com", "label": 1}
]
```

## Labels

- `0` = Legitimate
- `1` = Phishing

## Free Data Sources

- **PhishTank**: http://phishtank.org
- **OpenPhish**: https://openphish.com
- **URLhaus**: https://urlhaus.abuse.ch

## Auto-Collection

Instead of manual download, use:
```bash
python train.py --collect-data --source all --max-samples 1000
```
