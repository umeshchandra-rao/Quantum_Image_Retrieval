"""
Quantum Image Retrieval System Configuration
Supports both Azure Cosmos DB and Azure Blob Storage
"""

import os

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not available, using environment variables only")
except Exception as e:
    print(f"Could not load .env file: {e}")

# === Azure Cosmos DB Configuration - India Region ===
COSMOS_ENDPOINT = os.getenv('COSMOS_ENDPOINT', 'https://quantum-cosmos-india.documents.azure.com:443/')
COSMOS_KEY = os.getenv('COSMOS_KEY')  # Must be set via environment variable
COSMOS_DATABASE = os.getenv('COSMOS_DATABASE', 'quantum-images-india')
COSMOS_CONTAINER = os.getenv('COSMOS_CONTAINER', 'feature-vectors-india')

# === Azure Blob Storage Configuration ===
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING', '')
AZURE_STORAGE_CONTAINER = os.getenv('AZURE_STORAGE_CONTAINER', 'quantum-images')

# === Local Storage Configuration ===
UPLOAD_FOLDER = 'uploads'
DATASET_PATH = os.path.join('data', 'sample_images')
PROFESSIONAL_IMAGES_PATH = os.path.join('data', 'professional_images')

# === Model Configuration ===
MODEL_WEIGHTS_PATH = 'consistent_resnet50_8d.pth'
FEATURE_DIMENSION = 8

# === Application Configuration ===
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 8000))

# === Flask Configuration ===
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 8000
FLASK_DEBUG = False

# === Quantum Algorithm Configuration ===
N_ENCODING_QUBITS = 3
N_AUXILIARY_QUBITS = 7
USE_QUANTUM_INSPIRED = True  # Use quantum-inspired mode for performance

# === Confidence Thresholds ===
HIGH_CONFIDENCE_THRESHOLD = 0.88
GOOD_CONFIDENCE_THRESHOLD = 0.84

def validate_config():
    """Validate that all required configuration is present"""
    errors = []
    
    # Check Cosmos DB configuration
    if not COSMOS_ENDPOINT:
        errors.append("COSMOS_ENDPOINT not configured")
    
    if not COSMOS_KEY:
        errors.append("COSMOS_KEY environment variable is required")
    
    if not COSMOS_DATABASE:
        errors.append("COSMOS_DATABASE not configured")
    
    if not COSMOS_CONTAINER:
        errors.append("COSMOS_CONTAINER not configured")
    
    # Check Azure Blob Storage configuration (only warn, don't fail)
    if not AZURE_STORAGE_CONNECTION_STRING:
        print("Warning: AZURE_STORAGE_CONNECTION_STRING not configured - blob upload will be disabled")
    
    # Check model file exists
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        errors.append(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    else:
        print("Configuration validated")

def print_config():
    """Print current configuration (without sensitive keys)"""
    print("\n📋 Current Configuration:")
    print(f"🌐 Cosmos Endpoint: {COSMOS_ENDPOINT}")
    print(f"🗄️  Database: {COSMOS_DATABASE}")
    print(f"📦 Container: {COSMOS_CONTAINER}")
    print(f"🔑 Cosmos Key Length: {len(COSMOS_KEY)} characters" if COSMOS_KEY else "🔑 Cosmos Key: Not configured")
    print(f"Blob Storage: {'Configured' if AZURE_STORAGE_CONNECTION_STRING else 'Not configured'}")
    print(f"🧠 Model Weights: {MODEL_WEIGHTS_PATH}")
    print(f"📁 Upload Folder: {UPLOAD_FOLDER}")
    print(f"High Confidence: {HIGH_CONFIDENCE_THRESHOLD}")
    print(f"Good Confidence: {GOOD_CONFIDENCE_THRESHOLD}")

def validate_uploaded_file(file):
    """Validate uploaded file for security"""
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        raise ValueError(f"Invalid file extension: {file_ext}")
    
    # Check file size (16MB limit)
    if hasattr(file, 'content_length') and file.content_length > 16 * 1024 * 1024:
        raise ValueError("File too large (max 16MB)")
    
    # Check MIME type if python-magic is available
    try:
        import magic
        file_data = file.read(1024)  # Read first 1KB for MIME detection
        file.seek(0)  # Reset file pointer
        
        mime_type = magic.from_buffer(file_data, mime=True)
        allowed_mimes = {'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'}
        
        if mime_type not in allowed_mimes:
            raise ValueError(f"Invalid file type: {mime_type}")
    except ImportError:
        # python-magic not available, skip MIME check
        print("Warning: python-magic not available, skipping MIME type validation")
    except Exception as e:
        # Other errors in MIME detection
        print(f"Warning: MIME type detection failed: {e}")
    
    return True
