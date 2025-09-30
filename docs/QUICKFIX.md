# 🚀 Phisherman Quick Fix Guide

## Issue: Dependency Conflicts with System Tools

**Error:** `pip's dependency resolver does not currently take into account all the packages that are installed`

### ✅ Solution: Use Virtual Environment

```bash
# Navigate to project
cd /home/kaizen/Arsenal/Projects/Phisherman

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements-core.txt

# Verify it works
python3 detect.py -u https://google.com
```

### 🎯 One-Line Quick Start

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements-core.txt
```

### 🤖 Automated Setup

```bash
./setup_venv.sh
```

---

## Why This Happens

Your system has security tools with strict version requirements:
- `theharvester` needs specific versions of aiohttp, requests, etc.
- `mitmproxy` needs specific versions of tornado, pyperclip, etc.
- `faradaysec` needs specific versions of Flask, celery, etc.

These conflict with Phisherman's requirements.

**Virtual environments solve this by isolating dependencies.**

---

## Quick Commands Reference

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Check You're in Virtual Environment
```bash
which python3
# Should show: /home/kaizen/Arsenal/Projects/Phisherman/venv/bin/python3
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Full Workflow Example
```bash
cd /home/kaizen/Arsenal/Projects/Phisherman
source venv/bin/activate
python3 detect.py -u https://suspicious-site.com
python3 api/main.py  # Start API server
deactivate  # When done
```

---

## Alternative Solutions

### Option 1: Docker (Best for Production)
```bash
docker build -t phisherman .
docker run -p 8000:8000 phisherman
```

### Option 2: Conda
```bash
conda create -n phisherman python=3.11
conda activate phisherman
pip install -r requirements-core.txt
```

### Option 3: pipx (CLI Only)
```bash
pipx install -e /home/kaizen/Arsenal/Projects/Phisherman
```

---

## Troubleshooting

### "Still getting conflicts after creating venv"

**Solution:** Make sure venv is activated
```bash
# Check if activated
echo $VIRTUAL_ENV
# Should show: /home/kaizen/Arsenal/Projects/Phisherman/venv

# If empty, activate it
source venv/bin/activate
```

### "Command not found: pip"

**Solution:** Use python3 -m pip
```bash
python3 -m pip install -r requirements-core.txt
```

### "Cannot create venv"

**Solution:** Install venv module
```bash
sudo apt-get install python3-venv  # Debian/Ubuntu
sudo yum install python3-venv      # RHEL/CentOS
```

### "Still using system Python"

**Solution:** Recreate venv cleanly
```bash
rm -rf venv
python3 -m venv venv --clear
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
```

---

## File Structure After Setup

```
Phisherman/
├── venv/                    # Virtual environment (isolated)
│   ├── bin/
│   │   ├── activate        # Activation script
│   │   ├── python3         # Isolated Python
│   │   └── pip             # Isolated pip
│   └── lib/                # Isolated packages
├── requirements-core.txt   # Lightweight dependencies
├── setup_venv.sh          # Automated setup
├── activate.sh            # Quick activation helper
└── ...
```

---

## Requirements Files Explained

| File | Size | Use Case |
|------|------|----------|
| `requirements-core.txt` | ~500 MB | **Recommended** - Core ML, fast install |
| `requirements.txt` | ~3 GB | Full features with deep learning |

**For most users:** Use `requirements-core.txt`

---

## Quick Health Check

Run this after setup to verify everything works:

```bash
source venv/bin/activate

# Test 1: Check packages
python3 -c "import sklearn, xgboost, lightgbm, fastapi; print('✓ All packages OK')"

# Test 2: Run detector
python3 detect.py -u https://google.com

# Test 3: Start API (Ctrl+C to stop)
python3 api/main.py
```

---

## Common Mistakes

❌ **Wrong:** Installing without venv
```bash
pip install -r requirements-core.txt  # BAD - conflicts with system
```

✅ **Correct:** Using virtual environment
```bash
source venv/bin/activate
pip install -r requirements-core.txt  # GOOD - isolated
```

---

❌ **Wrong:** Forgetting to activate
```bash
python3 detect.py -u https://example.com  # Uses system Python
```

✅ **Correct:** Activate first
```bash
source venv/bin/activate
python3 detect.py -u https://example.com  # Uses venv Python
```

---

## Integration with Other Tools

### Using Phisherman Alongside System Tools

```bash
# Use system tool (mitmproxy, theharvester, etc.)
mitmproxy  # Uses system Python

# Switch to Phisherman
cd /home/kaizen/Arsenal/Projects/Phisherman
source venv/bin/activate
python3 detect.py -u https://example.com

# Switch back
deactivate

# Use system tools again
theharvester -d example.com
```

Virtual environments let you use both without conflicts!

---

## Automation Scripts

### Create Alias for Quick Access

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias phisherman='cd /home/kaizen/Arsenal/Projects/Phisherman && source venv/bin/activate'
alias phish-detect='cd /home/kaizen/Arsenal/Projects/Phisherman && source venv/bin/activate && python3 detect.py'
```

Then use:
```bash
phisherman  # Activates environment
phish-detect -u https://example.com  # Quick detection
```

---

## Status Indicators

Your shell should show when venv is active:

```bash
# Inactive
user@host:~/Projects/Phisherman$

# Active
(venv) user@host:~/Projects/Phisherman$
       ↑
       └── This shows venv is active
```

---

## Need More Help?

1. **Full Guide:** [DEPENDENCY_RESOLUTION.md](DEPENDENCY_RESOLUTION.md)
2. **Installation:** [INSTALLATION.md](INSTALLATION.md)
3. **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
4. **Docker Setup:** [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml)

---

## TL;DR

```bash
# Setup (once)
./setup_venv.sh

# Use (every time)
source venv/bin/activate
python3 detect.py -u <URL>
deactivate
```

**That's it! No more conflicts.** 🎉
