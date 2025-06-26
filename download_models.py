#!/usr/bin/env python3
"""
Model download script for InsightFace Face Matching Application
Downloads and sets up the buffalo_l model for face recognition
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import hashlib
import shutil

# InsightFace model configuration
INSIGHTFACE_MODELS = {
    'buffalo_l': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        'sha256': None,  # Add if available
        'description': 'High accuracy face recognition model (recommended)',
        'size_mb': 326
    },
    'buffalo_m': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip', 
        'sha256': None,
        'description': 'Medium accuracy model (faster inference)',
        'size_mb': 85
    },
    'buffalo_s': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip',
        'sha256': None, 
        'description': 'Small model (fastest inference)',
        'size_mb': 31
    }
}

# Additional ONNX models (optional)
ONNX_MODELS = {
    'retinaface': {
        'url': 'https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/models/retinaface-R50.onnx',
        'description': 'Alternative face detection model',
        'filename': 'retinaface-R50.onnx',
        'size_mb': 109
    }
}

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def download_with_progress(url, destination_path, expected_size_mb=None):
    """Download a file with progress indication"""
    print(f"Downloading: {os.path.basename(destination_path)}")
    print(f"From: {url}")
    
    if expected_size_mb:
        print(f"Expected size: ~{expected_size_mb}MB")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded // (1024*1024)}MB)", end='', flush=True)
        
        print(f"\nDownload completed: {destination_path}")
        print(f"File size: {os.path.getsize(destination_path) // (1024*1024)}MB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file to destination"""
    print(f"Extracting: {os.path.basename(zip_path)}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Extracted to: {extract_to}")
        
        # List extracted contents
        extracted_files = []
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), extract_to)
                extracted_files.append(rel_path)
        
        print(f"Extracted files: {len(extracted_files)}")
        for file in extracted_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(extracted_files) > 5:
            print(f"  ... and {len(extracted_files) - 5} more files")
            
        return True
        
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def setup_insightface_models(models_dir="/app/models"):
    """Download and setup InsightFace models"""
    models_dir = Path(models_dir)
    insightface_dir = models_dir / "insightface"
    insightface_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Setting up InsightFace Models ===")
    print(f"Models directory: {models_dir}")
    
    # Download buffalo_l (primary model used in your code)
    model_name = 'buffalo_l'
    model_config = INSIGHTFACE_MODELS[model_name]
    
    print(f"\nDownloading {model_name} model...")
    print(f"Description: {model_config['description']}")
    
    zip_path = insightface_dir / f"{model_name}.zip"
    extract_path = insightface_dir / model_name
    
    # Download the model
    if download_with_progress(model_config['url'], str(zip_path), model_config['size_mb']):
        
        # Extract the model
        if extract_zip(str(zip_path), str(extract_path)):
            # Remove zip file after successful extraction
            os.remove(zip_path)
            print(f"Removed zip file: {zip_path}")
            
            # Verify model files exist
            model_files = list(extract_path.rglob("*.onnx"))
            if model_files:
                print(f"✓ Model files found: {len(model_files)} ONNX files")
                for model_file in model_files:
                    rel_path = model_file.relative_to(extract_path)
                    print(f"  - {rel_path}")
            else:
                print("⚠ Warning: No ONNX files found in extracted model")
            
            return True
    
    return False

def setup_insightface_cache(cache_dir="/home/appuser/.insightface"):
    """Setup InsightFace cache directory structure"""
    cache_dir = Path(cache_dir)
    models_cache = cache_dir / "models"
    models_cache.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Setting up InsightFace Cache ===")
    print(f"Cache directory: {cache_dir}")
    
    # Create symbolic links if models exist in /app/models
    app_models = Path("/app/models/insightface")
    if app_models.exists():
        for model_dir in app_models.iterdir():
            if model_dir.is_dir():
                cache_link = models_cache / model_dir.name
                if not cache_link.exists():
                    try:
                        cache_link.symlink_to(model_dir, target_is_directory=True)
                        print(f"Created symlink: {cache_link} -> {model_dir}")
                    except Exception as e:
                        print(f"Could not create symlink for {model_dir.name}: {e}")
                        # Copy instead of symlink
                        shutil.copytree(model_dir, cache_link)
                        print(f"Copied instead: {model_dir} -> {cache_link}")

def verify_installation():
    """Verify that models are properly installed"""
    print("\n=== Verifying Installation ===")
    
    # Check for buffalo_l model (required by your code)
    buffalo_l_path = Path("/app/models/insightface/buffalo_l")
    if buffalo_l_path.exists():
        onnx_files = list(buffalo_l_path.rglob("*.onnx"))
        if onnx_files:
            print("✓ buffalo_l model installed correctly")
            print(f"Model files: {len(onnx_files)}")
            return True
        else:
            print("✗ buffalo_l model directory exists but no ONNX files found")
    else:
        print("✗ buffalo_l model not found")
    
    return False

def main():
    """Main function to download and setup all models"""
    models_directory = sys.argv[1] if len(sys.argv) > 1 else "/app/models"
    
    print("InsightFace Model Downloader")
    print("=" * 40)
    print(f"Target directory: {models_directory}")
    
    success = True
    
    # Setup InsightFace models
    if not setup_insightface_models(models_directory):
        print("✗ Failed to setup InsightFace models")
        success = False
    
    # Setup cache directory
    try:
        setup_insightface_cache()
    except Exception as e:
        print(f"Warning: Could not setup cache directory: {e}")
    
    # Verify installation
    if not verify_installation():
        print("✗ Model verification failed")
        success = False
    
    if success:
        print("\n✓ All models downloaded and setup successfully!")
        print("\nYour face matching application is ready to use.")
        print("Models available:")
        print("  - buffalo_l: High accuracy face recognition")
        
        # Show disk usage
        models_path = Path(models_directory)
        if models_path.exists():
            total_size = sum(f.stat().st_size for f in models_path.rglob('*') if f.is_file())
            print(f"\nTotal disk usage: {total_size // (1024*1024)}MB")
    else:
        print("\n✗ Model setup failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)