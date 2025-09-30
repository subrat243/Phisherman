# Dependency Conflict Resolution Guide

This guide helps you resolve dependency conflicts between Phisherman and system-installed security tools.

## Problem

You may encounter dependency conflicts if your system has security tools like:
- mitmproxy
- theharvester
- faradaysec
- sslyze
- certipy-ad

These tools have strict version requirements that can conflict with Phisherman's dependencies.

## Solution 1: Virtual Environment (Recommended)

The best solution is to use a Python virtual environment to isolate Phisherman's dependencies.

### Step 1: Create Virtual Environment

```bash
cd /home/kaizen/Arsenal/Projects/Phisherman

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 2: Install Core Requirements

```bash
# Install only core requirements (lightweight)
pip install -r requirements-core.txt
```

### Step 3: Verify Installation

```bash
# Test the detector
python3 detect.py -u https://google.com

# Start the API
python3 api/main.py
```

### Step 4: Deactivate When Done

```bash
deactivate
```

## Solution 2: Use pipx (For CLI Tool Only)

If you only want to use Phisherman as a CLI tool:

```bash
# Install pipx if not already installed
pip install --user pipx
python3 -m pipx ensurepath

# Install Phisherman in isolated environment
pipx install -e /home/kaizen/Arsenal/Projects/Phisherman
```

## Solution 3: Docker Container (Production)

For production deployment, use Docker to completely isolate dependencies:

### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-core.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-core.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Run API
CMD ["python3", "api/main.py"]
```

### Build and Run

```bash
# Build image
docker build -t phisherman:latest .

# Run container
docker run -p 8000:8000 phisherman:latest
```

## Solution 4: Conda Environment

Use Conda for better dependency management:

```bash
# Create conda environment
conda create -n phisherman python=3.11

# Activate it
conda activate phisherman

# Install dependencies
pip install -r requirements-core.txt

# When done
conda deactivate
```

## Understanding the Conflicts

### Common Conflicting Packages

| Package | Phisherman Needs | System Tools Need | Resolution |
|---------|------------------|-------------------|------------|
| aiohttp | >=3.8.5 | ==3.12.14 (theharvester) | Use venv |
| beautifulsoup4 | >=4.12.0 | ==4.13.4 (theharvester) | Use venv |
| certifi | >=2023.5.7 | ==2025.7.14 (theharvester) | Use venv |
| requests | >=2.31.0 | ==2.32.4 (theharvester) | Use venv |
| lxml | >=4.9.0 | ==6.0.0 (theharvester) | Use venv |
| fastapi | >=0.100.0 | ==0.116.1 (theharvester) | Use venv |

### Why Virtual Environments Work

Virtual environments create isolated Python installations where:
- Each project has its own dependency versions
- System tools remain unaffected
- No global package conflicts
- Easy to replicate across systems

## Quick Start Commands

### Option A: Virtual Environment (Fastest)

```bash
# One-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt

# Every time you use Phisherman
source venv/bin/activate
python3 detect.py -u https://example.com
deactivate
```

### Option B: Docker (Most Isolated)

```bash
# One-time build
docker build -t phisherman .

# Every time you use it
docker run -p 8000:8000 phisherman
```

## Troubleshooting

### Issue: "Cannot activate virtual environment"

**Solution:**
```bash
# Make sure you're in the project directory
cd /home/kaizen/Arsenal/Projects/Phisherman

# Try with full path
source /home/kaizen/Arsenal/Projects/Phisherman/venv/bin/activate
```

### Issue: "pip: command not found inside venv"

**Solution:**
```bash
# Reinstall pip in venv
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
```

### Issue: "Still getting conflicts"

**Solution:**
```bash
# Remove and recreate venv
rm -rf venv
python3 -m venv venv --clear
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
```

### Issue: "System packages still interfere"

**Solution:**
```bash
# Create venv without system packages
python3 -m venv venv --without-pip
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python3
pip install -r requirements-core.txt
```

## Requirements Files Explained

### requirements.txt
- Full dependencies with all optional features
- Includes TensorFlow, transformers, computer vision
- Use for development or full ML capabilities
- ~2-3 GB disk space

### requirements-core.txt
- Minimal dependencies for core functionality
- Traditional ML only (no deep learning)
- Lightweight and fast to install
- ~500 MB disk space
- **Recommended for most users**

## Best Practices

1. **Always use virtual environments** for Python projects
2. **Activate venv before running** any Phisherman commands
3. **Keep system tools separate** from project dependencies
4. **Use Docker** for production deployments
5. **Document your environment** in deployment notes

## Environment Management Cheat Sheet

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Check active environment
which python3
# Should show: .../venv/bin/python3

# List installed packages
pip list

# Freeze current versions
pip freeze > requirements-frozen.txt

# Deactivate
deactivate
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Phisherman

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Create virtual environment
        run: python3 -m venv venv

      - name: Install dependencies
        run: |
          source venv/bin/activate
          pip install -r requirements-core.txt
          pip install pytest

      - name: Run tests
        run: |
          source venv/bin/activate
          pytest tests/
```

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Docker Documentation](https://docs.docker.com/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/)
- [pipx Documentation](https://pypa.github.io/pipx/)

## Summary

**TL;DR:** Use a virtual environment to avoid conflicts:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
python3 detect.py -u https://example.com
```

This isolates Phisherman's dependencies from your system tools and solves all conflict issues.
