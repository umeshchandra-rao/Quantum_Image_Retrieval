"""
Enhanced Web Application for Quantum Image Retrieval

This web application provides a user interface for uploading images
and finding similar images using the enhanced quantum image retrieval system.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from flask_cors import CORS
import os
import sys
import numpy as np
from PIL import Image
import base64
import io
import json
import time
from datetime import datetime
import logging

# Azure imports for blob storage
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Exception Classes
class QuantumImageRetrievalError(Exception):
    """Base exception for Quantum Image Retrieval System"""
    pass

class FeatureExtractionError(QuantumImageRetrievalError):
    """Raised when feature extraction fails"""
    pass

class DatabaseConnectionError(QuantumImageRetrievalError):
    """Raised when database connection fails"""
    pass

class ImageValidationError(QuantumImageRetrievalError):
    """Raised when image validation fails"""
    pass

class QuantumAlgorithmError(QuantumImageRetrievalError):
    """Raised when quantum algorithm fails"""
    pass

def make_json_safe(obj):
    """Convert object to JSON-safe format"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    else:
        return obj

# Add src to path
sys.path.append('src')

# Import unified feature extractor (replaces inline extraction and old extractor)
from unified_feature_extractor import UnifiedFeatureExtractor

# Quantum algorithm import - ENABLED for true quantum processing
try:
    from src.quantum.ae_qip_algorithm import AEQIPAlgorithm
    QUANTUM_AVAILABLE = True
except ImportError as e:
    QUANTUM_AVAILABLE = False
    AEQIPAlgorithm = None

# from src.cloud.cloud_quantum_retrieval import CloudQuantumRetrieval  # Not needed for unified system

# Import configuration
import config

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
feature_extractor = None
quantum_algorithm = None
cloud_retrieval = None
image_features_cache = {}

def initialize_components():
    """Initialize the system components with optimized resource management"""
    global feature_extractor, quantum_algorithm, cloud_retrieval
    
    # Validate configuration first
    config.validate_config()
    
    # Initialize unified feature extractor (singleton pattern ensures single instance)
    try:
        feature_extractor = UnifiedFeatureExtractor(
            cosmos_endpoint=config.COSMOS_ENDPOINT,
            cosmos_key=config.COSMOS_KEY,
            database_name=config.COSMOS_DATABASE,
            container_name=config.COSMOS_CONTAINER
        )
    except ValueError as e:
        logger.error(f"Feature extractor initialization failed: {e}")
        raise
    
    # Initialize quantum algorithm with quantum-inspired mode for performance
    if QUANTUM_AVAILABLE:
        quantum_algorithm = AEQIPAlgorithm(
            n_encoding_qubits=config.N_ENCODING_QUBITS,
            n_auxiliary_qubits=config.N_AUXILIARY_QUBITS,
            use_true_quantum=not config.USE_QUANTUM_INSPIRED  # Use quantum-inspired by default
        )
    else:
        quantum_algorithm = None
        logger.warning("Quantum algorithm not available")
    
    return True

def get_cloud_retrieval():
    """Get cloud retrieval instance, initializing if needed - NOT USED WITH UNIFIED SYSTEM"""
    global cloud_retrieval
    if cloud_retrieval is None:
        # Using unified system instead - no longer needed
        raise RuntimeError("CloudQuantumRetrieval is disabled - using UnifiedFeatureExtractor")
    return cloud_retrieval

def compute_similarity(features1, features2):
    """Compute quantum similarity between two feature vectors using 8D quantum algorithm"""
    global quantum_algorithm
    
    if quantum_algorithm is None and QUANTUM_AVAILABLE:
        initialize_components()
    
    try:
        if quantum_algorithm is not None:
            # Use your 8D quantum algorithm
            similarity = quantum_algorithm.calculate_similarity(features1, features2)
            return similarity
        else:
            # Fallback to classical cosine similarity if quantum not available
            import numpy as np
            features1 = np.array(features1)
            features2 = np.array(features2)
            cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            return max(0.0, cosine_sim)
    except Exception as e:
        return 0.0

def get_all_images():
    """Get all available images from both sample and professional directories"""
    images = []
    
    # Get sample images
    sample_path = config.DATASET_PATH
    if os.path.exists(sample_path):
        for file in os.listdir(sample_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append({
                    'path': os.path.join(sample_path, file),
                    'name': file,
                    'type': 'sample'
                })
    
    # Get professional images
    professional_path = config.PROFESSIONAL_IMAGES_PATH
    if os.path.exists(professional_path):
        for category in os.listdir(professional_path):
            category_path = os.path.join(professional_path, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append({
                            'path': os.path.join(category_path, file),
                            'name': file,
                            'type': category,
                            'category': category
                        })
    
    return images

@app.route('/')
def index():
    """Main page"""
    return render_template('minimal_upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and similarity search using cloud infrastructure"""
    
    # Validate request
    if 'file' not in request.files:
        logger.warning("Upload request missing file")
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload request with empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Validate uploaded file
        config.validate_uploaded_file(file)
        
    except ValueError as e:
        logger.error(f"File validation failed: {e}")
        return jsonify({"error": f"Invalid file: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        return jsonify({"error": "File validation failed"}), 500
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        return _process_uploaded_image(filepath, filename)
        
    except ImageValidationError as e:
        logger.error(f"Image validation error: {e}")
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400
    except FeatureExtractionError as e:
        logger.error(f"Feature extraction error: {e}")
        return jsonify({"error": "Feature extraction failed"}), 500
    except DatabaseConnectionError as e:
        logger.error(f"Database connection error: {e}")
        return jsonify({"error": "Database connection failed"}), 503
    except QuantumAlgorithmError as e:
        logger.error(f"Quantum algorithm error: {e}")
        return jsonify({"error": "Similarity calculation failed"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in upload: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

def _process_uploaded_image(filepath, filename):
    """Process uploaded image and return similarity search results"""
    try:
        # Read and validate image
        image = Image.open(filepath).convert('RGB')
        
        # Extract features using unified feature extractor
        query_features = feature_extractor.extract_features_from_image(image)
        
        if query_features is None:
            raise FeatureExtractionError("Feature extraction returned None")
        
        # Perform similarity search
        results, processing_time, search_method = _perform_similarity_search(query_features)
        
        # Build response
        return _build_search_response(results, filename, processing_time, search_method)
        
    except Exception as e:
        logger.error(f"Error processing image {filepath}: {e}")
        # Clean up uploaded file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

def _perform_similarity_search(query_features):
    """Perform similarity search and return filtered results"""
    start_time = time.time()
    
    try:
        # Get database stats from unified extractor
        stats = feature_extractor.get_database_stats()
        
        # Search for similar images using unified extractor (only top 5 results)
        search_results = feature_extractor.search_similar_images(
            query_features=query_features,
            top_k=5
        )
        
        # Filter and process results
        filtered_results = _filter_search_results(search_results)
        
        processing_time = time.time() - start_time
        search_method = f"India Region Cosmos DB ({stats.get('region', 'unknown')})"
        
        return filtered_results, processing_time, search_method
        
    except Exception as e:
        logger.error(f"Database search failed: {e}")
        raise DatabaseConnectionError(f"Search failed: {e}")

def _filter_search_results(search_results):
    """Filter search results based on confidence thresholds"""
    if not search_results:
        return []
    
    # Enhanced similarity filtering with graduated confidence thresholds
    similarities = [result['similarity'] for result in search_results]
    
    # Count matches by confidence level
    high_confidence = [r for r in search_results if r['similarity'] >= config.HIGH_CONFIDENCE_THRESHOLD]
    good_confidence = [r for r in search_results if config.GOOD_CONFIDENCE_THRESHOLD <= r['similarity'] < config.HIGH_CONFIDENCE_THRESHOLD]
    
    # Accept results if we have high or good confidence matches
    has_meaningful_matches = len(high_confidence) > 0 or len(good_confidence) > 0
    
    if not has_meaningful_matches:
        return []
    
    # Keep high and good confidence results
    filtered_results = high_confidence + good_confidence
    
    # Convert to standardized format
    results = []
    for result in filtered_results:
        # Only include results with similarity >= good confidence threshold
        if result['similarity'] < config.GOOD_CONFIDENCE_THRESHOLD:
            continue
        
        result_data = _format_search_result(result)
        results.append(result_data)
    
    return results

def _format_search_result(result):
    """Format a single search result for API response"""
    image_id = result['image_id']
    
    # Use a single unified image serving route for ALL images
    image_url = f"/serve_unified_image/{image_id}"
    
    # Detect category for metadata only
    if image_id.startswith('healthcare_'):
        category = 'healthcare'
    elif image_id.startswith('satellite_'):
        category = 'satellite'
    elif image_id.startswith('surveillance_'):
        category = 'surveillance'
    else:
        category = 'unknown'
    
    # Generate appropriate filename based on category
    if category == 'satellite':
        filename = f"{image_id}.jpg"  # Satellite images use .jpg
    elif category == 'surveillance':
        filename = f"{image_id}.jpg"  # Surveillance images typically use .jpg
    else:
        filename = f"{image_id}.jpeg"  # Healthcare images use .jpeg
    
    # Determine confidence level for each result
    confidence_level = "high" if result['similarity'] >= config.HIGH_CONFIDENCE_THRESHOLD else "good"
    is_exact_match = result['similarity'] > 0.95  # Very high threshold for "exact"
    
    return {
        'id': image_id,
        'image_id': image_id,
        'image_url': image_url,
        'similarity': result['similarity'],
        'similarity_score': result['similarity'],
        'confidence_level': confidence_level,
        'type': category,
        'is_exact_match': is_exact_match,
        'metadata': {
            'filename': filename,
            'category': category,
            'size': 'unknown',
            'timestamp': datetime.now().isoformat()
        }
    }

def _build_search_response(results, filename, processing_time, search_method):
    """Build final API response for search results"""
    if len(results) == 0:
        response_data = {
            'status': 'no_matches',
            'message': "No similar images found",
            'results': [],
            'similar_images': [],
            'upload_path': f"/uploads/{filename}",
            'processing_time': processing_time,
            'search_method': search_method,
            'total_images_searched': 0  # Will be updated with actual stats
        }
    else:
        # Calculate confidence distribution
        high_conf_count = len([r for r in results if r['confidence_level'] == 'high'])
        good_conf_count = len([r for r in results if r['confidence_level'] == 'good'])
        
        response_data = {
            'status': 'success',
            'message': f"Found {len(results)} similar images ({high_conf_count} high-confidence, {good_conf_count} good-confidence) in {processing_time:.3f} seconds",
            'results': results,  # minimal_upload.html expects 'results'
            'similar_images': results,  # compatibility for frontend
            'upload_path': f"/uploads/{filename}",
            'processing_time': processing_time,
            'search_method': search_method,
            'total_images_searched': 0,  # Will be updated with actual stats
            'confidence_info': {
                'threshold_used': config.GOOD_CONFIDENCE_THRESHOLD,
                'confidence_levels': {
                    'high': f'≥{int(config.HIGH_CONFIDENCE_THRESHOLD * 100)}%',
                    'good': f'{int(config.GOOD_CONFIDENCE_THRESHOLD * 100)}-{int(config.HIGH_CONFIDENCE_THRESHOLD * 100)}%'
                },
                'highest_similarity': max(result['similarity'] for result in results) if results else 0.0,
                'confidence_distribution': {
                    'high_confidence': high_conf_count,
                    'good_confidence': good_conf_count
                }
            }
        }
    
    return jsonify(make_json_safe(response_data))

@app.route('/health')
def health_check():
    """Health check endpoint for React frontend"""
    return jsonify({"status": "healthy", "message": "Quantum Image Search API is running"})

@app.route('/manifest.json')
def manifest():
    """Return empty manifest for React frontend"""
    return jsonify({"name": "Quantum Image Search", "short_name": "QIS", "start_url": "/"})

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from sample directory"""
    return send_from_directory(config.DATASET_PATH, filename)

@app.route('/professional/<category>/<filename>')
def serve_professional_image(category, filename):
    """Serve images from professional directory"""
    category_path = os.path.join(config.PROFESSIONAL_IMAGES_PATH, category)
    return send_from_directory(category_path, filename)

@app.route('/serve_unified_image/<path:filename>')
def serve_unified_image(filename):
    """Unified image serving route for all categories from Azure Blob Storage"""
    
    def get_blob_image_data(image_id):
        """Retrieve image data from Azure Blob Storage"""
        try:
            # Initialize blob client if not already done
            if not config.AZURE_STORAGE_CONNECTION_STRING:
                logger.error("Azure Blob Storage not configured")
                return None, "Azure Blob Storage not configured"
            
            blob_service_client = BlobServiceClient.from_connection_string(
                config.AZURE_STORAGE_CONNECTION_STRING
            )
            
            # Determine category and container from image_id
            if image_id.startswith('healthcare_'):
                container_name = 'quantum-images-healthcare'
                base_id = image_id[len('healthcare_'):]
                possible_extensions = ['.jpeg', '.jpg', '.png']
            elif image_id.startswith('satellite_'):
                container_name = 'quantum-images-satellite'
                base_id = image_id[len('satellite_'):]
                possible_extensions = ['.jpg', '.jpeg', '.png']
            elif image_id.startswith('surveillance_'):
                container_name = 'quantum-images-surveillance'
                base_id = image_id[len('surveillance_'):]
                possible_extensions = ['.jpg', '.jpeg', '.png']
            else:
                logger.error(f"Unknown image category for ID: {image_id}")
                return None, f"Unknown image category for ID: {image_id}"
            
            # Try different blob name patterns based on how uploaders store files
            blob_name_patterns = [
                # Pattern 1: base_id + extension (e.g., "IM-0001-0001.jpeg")
                *[base_id + ext for ext in possible_extensions],
                # Pattern 2: category/filename (e.g., "healthcare/IM-0001-0001.jpeg")
                *[f"{image_id.split('_')[0]}/{base_id}" + ext for ext in possible_extensions],
                # Pattern 3: full image_id as blob name (e.g., "healthcare_IM-0001-0001.jpeg")
                *[image_id + ext for ext in possible_extensions],
            ]
            
            # Try different blob name patterns to find the blob
            for blob_name in blob_name_patterns:
                try:
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name, 
                        blob=blob_name
                    )
                    
                    # Check if blob exists and download
                    if blob_client.exists():
                        blob_data = blob_client.download_blob().readall()
                        # Return the extension from the blob name for MIME type
                        ext = '.' + blob_name.split('.')[-1] if '.' in blob_name else '.jpg'
                        return blob_data, ext
                        
                except Exception as e:
                    continue
            
            return None, f"Image not found in any container: {image_id}"
            
        except Exception as e:
            logger.error(f"Error retrieving image from blob storage: {e}")
            return None, f"Blob storage error: {e}"
    
    try:
        # Get image data from Azure Blob Storage
        image_data, ext_or_error = get_blob_image_data(filename)
        
        if image_data is None:
            # Fallback to local file system as backup
            return serve_local_image_fallback(filename)
        
        # Determine MIME type based on extension
        if ext_or_error.lower() in ['.jpg', '.jpeg']:
            mimetype = 'image/jpeg'
        elif ext_or_error.lower() == '.png':
            mimetype = 'image/png'
        else:
            mimetype = 'image/jpeg'  # Default
        
        # Return image data directly
        return Response(
            image_data,
            mimetype=mimetype,
            headers={
                'Cache-Control': 'public, max-age=3600',  # Cache for 1 hour
                'Content-Length': str(len(image_data))
            }
        )
        
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({"error": f"Failed to serve image: {str(e)}"}), 500

def serve_local_image_fallback(filename):
    """Fallback to serve images from local directories if blob storage fails"""
    def find_image_file(image_id):
        """Find the actual file for a given image_id in local storage"""        
        # Determine category and base filename
        if image_id.startswith('healthcare_'):
            base_id = image_id[len('healthcare_'):]
            folder = os.path.join("data", "professional_images", "healthcare")
            extensions = ['.jpeg', '.jpg', '.png']
        elif image_id.startswith('satellite_'):
            base_id = image_id[len('satellite_'):]
            folder = os.path.join("data", "professional_images", "satellite")
            extensions = ['.jpg', '.jpeg', '.png']
        elif image_id.startswith('surveillance_'):
            base_id = image_id[len('surveillance_'):]
            folder = os.path.join("data", "professional_images", "surveillance")
            # For surveillance: base_id + .jpg works perfectly
            filename = base_id + '.jpg'
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                return folder, filename
            # Try other extensions if .jpg fails
            for ext in ['.jpeg', '.png']:
                filename = base_id + ext
                full_path = os.path.join(folder, filename)
                if os.path.exists(full_path):
                    return folder, filename
            return None, None
        else:
            return None, None
        
        if not os.path.exists(folder):
            return None, None
        
        # For healthcare and satellite - try extensions
        for ext in extensions:
            filename = base_id + ext
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                return folder, filename
        
        return None, None
    
    try:
        # Find the actual file using proven logic
        folder, actual_filename = find_image_file(filename)
        
        if folder and actual_filename:
            return send_from_directory(folder, actual_filename)
        else:
            logger.error(f"Image not found in local storage either: {filename}")
            return jsonify({"error": f"Image {filename} not found in blob storage or local storage"}), 404
            
    except Exception as e:
        logger.error(f"Error in local fallback for {filename}: {e}")
        return jsonify({"error": f"Image serving failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    app.run(debug=config.FLASK_DEBUG, host='0.0.0.0', port=config.FLASK_PORT)
