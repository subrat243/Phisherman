#!/usr/bin/env python3
"""
Phisherman Setup Script
Automated installation and configuration
"""

import sys
import subprocess
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step, total, text):
    """Print step progress"""
    print(f"[{step}/{total}] {text}...")

def run_command(cmd, description):
    """Run command and handle errors"""
    try:
        subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
        print(f"  ✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed")
        print(f"    Error: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        print(f"    Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("  ✗ requirements.txt not found")
        return False
    
    cmd = f"{sys.executable} -m pip install -r requirements.txt --quiet"
    return run_command(cmd, "Installing dependencies")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("  Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("  ✓ NLTK data downloaded")
        return True
    except ImportError:
        print("  ⚠ NLTK not installed, skipping data download")
        return True
    except Exception as e:
        print(f"  ⚠ NLTK data download failed: {e}")
        return True  # Non-critical

def verify_installation():
    """Verify core modules can be imported"""
    modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('requests', 'Requests'),
        ('fastapi', 'FastAPI'),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} not found")
            all_ok = False
    
    return all_ok

def create_directories():
    """Create necessary directories"""
    dirs = [
        'models',
        'models/checkpoints',
        'data/raw',
        'data/raw/phishtank',
        'data/raw/openphish',
        'data/raw/tranco',
        'data/raw/urlhaus',
        'data/processed',
        'data/datasets',
        'data/feedback',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("  ✓ Created directories")
    return True

def run_basic_test():
    """Run basic functionality test"""
    try:
        # Test URL parsing
        from urllib.parse import urlparse
        test_url = "https://example.com"
        parsed = urlparse(test_url)
        
        if parsed.scheme and parsed.netloc:
            print("  ✓ Basic URL parsing works")
            return True
        else:
            print("  ✗ URL parsing test failed")
            return False
    except Exception as e:
        print(f"  ✗ Basic test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 70)
    print("  Setup Complete! 🎉")
    print("=" * 70)
    print("\nNext Steps:\n")
    print("1. Test URL detection:")
    print('   python detect.py --url "https://google.com"\n')
    print("2. Start API server:")
    print("   python api/main.py\n")
    print("3. Load browser extension:")
    print("   - Open chrome://extensions/")
    print("   - Enable Developer mode")
    print("   - Load unpacked → select browser_extension/\n")
    print("4. Train ML models (optional):")
    print("   python train.py --collect-data\n")
    print("5. View API docs:")
    print("   http://localhost:8000/docs\n")
    print("=" * 70 + "\n")

def main():
    """Main setup process"""
    print_header("Phisherman Setup")
    
    total_steps = 7
    current_step = 0
    
    # Step 1: Check Python version
    current_step += 1
    print_step(current_step, total_steps, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing dependencies")
    if not install_dependencies():
        print("\n⚠ Dependency installation failed. Try manually:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Download NLTK data
    current_step += 1
    print_step(current_step, total_steps, "Downloading NLTK data")
    download_nltk_data()
    
    # Step 4: Create directories
    current_step += 1
    print_step(current_step, total_steps, "Creating directories")
    create_directories()
    
    # Step 5: Verify installation
    current_step += 1
    print_step(current_step, total_steps, "Verifying installation")
    if not verify_installation():
        print("\n⚠ Some modules are missing. Installation may be incomplete.")
    
    # Step 6: Run basic test
    current_step += 1
    print_step(current_step, total_steps, "Running basic tests")
    run_basic_test()
    
    # Step 7: Complete
    current_step += 1
    print_step(current_step, total_steps, "Finalizing setup")
    print("  ✓ Setup complete")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSetup failed with error: {e}")
        sys.exit(1)
