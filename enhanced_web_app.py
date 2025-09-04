"""
Enhanced Web Application for Quantum Image Retrieval

This web application provides a user interface for uploading images
and finding similar images using the enhanced quantum image retrieval system.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
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
    print("Quantum algorithm enabled - using AE-QIP algorithm for enhanced similarity calculation")
except ImportError as e:
    QUANTUM_AVAILABLE = False
    AEQIPAlgorithm = None
    print(f"Quantum algorithm disabled due to import error: {e}")
    print("   Falling back to classical similarity only")

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
    """Initialize the system components"""
    global feature_extractor, quantum_algorithm, cloud_retrieval
    
    # Validate configuration first
    config.validate_config()
    
    print(f"Configuration validated")
    print(f"Endpoint: {config.COSMOS_ENDPOINT}")
    print(f"Database: {config.COSMOS_DATABASE}")
    print(f"Container: {config.COSMOS_CONTAINER}")
    print(f"Cosmos Key Length: {len(config.COSMOS_KEY)} characters")
    
    # Initialize unified feature extractor for India region with explicit validation
    try:
        feature_extractor = UnifiedFeatureExtractor(
            cosmos_endpoint=config.COSMOS_ENDPOINT,
            cosmos_key=config.COSMOS_KEY,
            database_name=config.COSMOS_DATABASE,
            container_name=config.COSMOS_CONTAINER
        )
        print(f"UnifiedFeatureExtractor initialized successfully")
    except ValueError as e:
        print(f"Failed to initialize UnifiedFeatureExtractor: {e}")
        print(f"   Check your COSMOS_KEY configuration in the .env file")
        raise
    
    # Initialize quantum algorithm with quantum-inspired mode for performance
    if QUANTUM_AVAILABLE:
        quantum_algorithm = AEQIPAlgorithm(
            n_encoding_qubits=config.N_ENCODING_QUBITS,
            n_auxiliary_qubits=config.N_AUXILIARY_QUBITS,
            use_true_quantum=not config.USE_QUANTUM_INSPIRED  # Use quantum-inspired by default
        )
        print("System components initialized with ResNet-50 and AE-QIP Quantum Algorithm")
    else:
        quantum_algorithm = None
        print("System components initialized with ResNet-50 (Classical similarity only)")
    
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
        print(f"Error computing similarity: {e}")
        return 0.0

def get_all_images():
    """Get all available images from both sample and professional directories"""
    images = []
    
    # Get sample images
    sample_path = config.DATASET_PATH
    if os.path.exists(sample_path):
        for file in os.listdir(sample_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
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
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
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
    print("[UPLOAD DEBUG] Upload function called!")
    
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    print(f"[UPLOAD DEBUG] Processing file: {file.filename}")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image
        image = Image.open(filepath).convert('RGB')
        
        # Extract features using unified feature extractor
        print(f"Extracting features for uploaded image: {filepath}")
        query_features = feature_extractor.extract_features_from_image(image)
        
        if query_features is None:
            return jsonify({"error": "Feature extraction failed"}), 500
        
        # Convert features list to numpy array for similarity search
        import numpy as np
        query_features_np = np.array(query_features)
        
        # Use unified feature extractor for similarity search (India region)
        print("🇮🇳 Using India region Cosmos DB similarity search...")
        start_time = time.time()
        
        try:
            # Get database stats from unified extractor
            stats = feature_extractor.get_database_stats()
            print(f"📊 Connected to {stats.get('region')} Cosmos DB with {stats.get('total_items', 0)} images")
            
            # Search for similar images using unified extractor (only top 5 results)
            search_results = feature_extractor.search_similar_images(
                query_features=query_features,
                top_k=5
            )
            
            # Convert to expected format for frontend
            print(f"[MAIN DEBUG] Starting to process {len(search_results)} search results...")
            
            # Enhanced similarity filtering with graduated confidence thresholds
            if search_results:
                similarities = [result['similarity'] for result in search_results]
                max_similarity = max(similarities)
                avg_similarity = sum(similarities) / len(similarities)
                similarity_gap = max_similarity - avg_similarity if len(similarities) > 1 else 1.0
                
                print(f"📊 [CONFIDENCE DEBUG] Max similarity: {max_similarity:.3f}, Avg: {avg_similarity:.3f}, Gap: {similarity_gap:.3f}")
                
                # NEW GRADUATED CONFIDENCE THRESHOLDS (based on analysis)
                # High confidence: ≥88% (exact/near-exact matches)
                # Good confidence: ≥84% (strong matches) 
                # Medium confidence: ≥80% (potential matches)
                
                # Count matches by confidence level
                high_confidence = [r for r in search_results if r['similarity'] >= 0.88]
                good_confidence = [r for r in search_results if 0.84 <= r['similarity'] < 0.88] 
                medium_confidence = [r for r in search_results if 0.80 <= r['similarity'] < 0.84]
                
                print(f"[CONFIDENCE] High (≥88%): {len(high_confidence)}, Good (84-88%): {len(good_confidence)}, Medium (80-84%): {len(medium_confidence)}")
                
                # Accept results if we have high or good confidence matches
                has_meaningful_matches = len(high_confidence) > 0 or len(good_confidence) > 0
                
                if not has_meaningful_matches:
                    print(f"🚫 [CONFIDENCE] No meaningful matches found (max: {max_similarity:.3f} < 0.84)")
                    search_results = []  # Clear all results
                else:
                    # Keep high and good confidence results
                    search_results = high_confidence + good_confidence
                    print(f"[CONFIDENCE] Showing {len(search_results)} meaningful matches")
            
            results = []
            for result in search_results:
                # NEW LOWER THRESHOLD: Only include results with similarity >= 84% (0.84)
                if result['similarity'] < 0.84:
                    print(f"🚫 [DEBUG] Skipping low similarity result: {result['image_id']} ({result['similarity']:.3f})")
                    continue
                
                # Get image_id and create a unified serving URL
                image_id = result['image_id']
                print(f"[DEBUG] Processing search result: image_id = {image_id}")
                
                # Use a single unified image serving route for ALL images
                image_url = f"/serve_unified_image/{image_id}"
                print(f"[DEBUG] Generated unified URL: {image_url}")
                
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
                confidence_level = "high" if result['similarity'] >= 0.88 else "good"
                is_exact_match = result['similarity'] > 0.95  # Very high threshold for "exact"
                
                results.append({
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
                })
            
            processing_time = time.time() - start_time
            search_method = f"India Region Cosmos DB ({stats.get('region')})"
            
        except Exception as cloud_error:
            print(f"India region Cosmos DB search failed: {cloud_error}")
            return jsonify({"status": "error", "message": f"India region Cosmos DB search failed: {cloud_error}"}), 500
        
        # Return JSON response for frontend
        print(f"[FINAL DEBUG] Returning {len(results)} results to frontend")
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result['image_id']} -> {result['image_url']}")
        
        # Enhanced response with graduated confidence levels
        if len(results) == 0:
            response_data = {
                'status': 'no_matches',
                'message': "No similar images found",
                'results': [],
                'similar_images': [],
                'upload_path': f"/uploads/{filename}",
                'processing_time': processing_time,
                'search_method': search_method,
                'total_images_searched': stats.get('total_images', 0)
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
                'total_images_searched': stats.get('total_images', 0),
                'confidence_info': {
                    'threshold_used': 0.84,
                    'confidence_levels': {
                        'high': '≥88%',
                        'good': '84-88%'
                    },
                    'highest_similarity': max(result['similarity'] for result in results) if results else 0.0,
                    'confidence_distribution': {
                        'high_confidence': high_conf_count,
                        'good_confidence': good_conf_count
                    }
                }
            }
        
        return jsonify(make_json_safe(response_data))
    
    except Exception as e:
        return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500

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
    """Unified image serving route for all categories (healthcare, satellite, surveillance)"""
    def find_image_file(image_id):
        """Find the actual file for a given image_id - PROVEN WORKING LOGIC"""        
        # Determine category and base filename
        if image_id.startswith('healthcare_'):
            base_id = image_id[len('healthcare_'):]
            folder = os.path.join("data", "professional_images", "healthcare")
            extensions = ['.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp']
        elif image_id.startswith('satellite_'):
            base_id = image_id[len('satellite_'):]
            folder = os.path.join("data", "professional_images", "satellite")
            extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        elif image_id.startswith('surveillance_'):
            base_id = image_id[len('surveillance_'):]
            folder = os.path.join("data", "professional_images", "surveillance")
            # For surveillance: base_id + .jpg works perfectly (proven by testing)
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
    
    # Find the actual file using proven logic
    folder, actual_filename = find_image_file(filename)
    
    if folder and actual_filename:
        return send_from_directory(folder, actual_filename)
    else:
        return jsonify({"error": f"Image {filename} not found"}), 404

@app.route('/serve_healthcare_image/<path:filename>')
def serve_healthcare_image(filename):
    """Serve healthcare images from local healthcare folder with robust filename handling"""
    healthcare_path = os.path.join("data", "professional_images", "healthcare")
    
    print(f"🏥 [DEBUG] Original healthcare request for: {filename}")
    
    # Remove 'healthcare_' prefix if present (database IDs include this but files don't)
    if filename.startswith('healthcare_'):
        filename = filename[len('healthcare_'):]
        print(f"🏥 [DEBUG] After removing prefix: {filename}")
    
    # Remove any existing extension to get base filename
    base_filename = filename
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
        if base_filename.endswith(ext):
            base_filename = base_filename[:-len(ext)]
            break
    
    print(f"🏥 [DEBUG] Base filename: {base_filename}")
    print(f"🏥 [DEBUG] Looking in: {healthcare_path}")
    
    # Try exact match first (for most healthcare filenames)
    for ext in ['.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp']:
        actual_filename = base_filename + ext
        full_path = os.path.join(healthcare_path, actual_filename)
        print(f"🏥 [DEBUG] Trying exact: {actual_filename} -> {full_path}")
        if os.path.exists(full_path):
            print(f"[DEBUG] Found healthcare image: {actual_filename}")
            return send_from_directory(healthcare_path, actual_filename)
    
    # If exact match fails, try partial matching for complex IDs
    try:
        import os
        all_files = os.listdir(healthcare_path)
        print(f"🏥 [DEBUG] Trying partial matching among {len(all_files)} files...")
        
        # Look for files that start with the beginning of our base_filename
        for file in all_files:
            if file.startswith(base_filename[:10]) and file.endswith(('.jpeg', '.jpg', '.png', '.tiff', '.tif', '.bmp')):
                print(f"[DEBUG] Found healthcare image via partial match: {file}")
                return send_from_directory(healthcare_path, file)
        
        print(f"🏥 [DEBUG] No partial matches found for: {base_filename[:10]}...")
    except Exception as e:
        print(f"🏥 [DEBUG] Error during partial matching: {e}")
    
    print(f"[DEBUG] Healthcare file not found for: {base_filename} (tried exact and partial matching)")
    return jsonify({"error": f"Healthcare image {filename} not found"}), 404

@app.route('/serve_satellite_image/<path:filename>')
def serve_satellite_image(filename):
    """Serve satellite images from local satellite folder with robust filename handling"""
    satellite_path = os.path.join("data", "professional_images", "satellite")
    
    print(f"🛰️  [DEBUG] Original satellite request for: {filename}")
    
    # Remove 'satellite_' prefix if present (database IDs include this but files don't)
    if filename.startswith('satellite_'):
        filename = filename[len('satellite_'):]
        print(f"🛰️  [DEBUG] After removing prefix: {filename}")
    
    # Remove any existing extension to get base filename
    base_filename = filename
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        if base_filename.endswith(ext):
            base_filename = base_filename[:-len(ext)]
            break
    
    print(f"🛰️  [DEBUG] Base filename: {base_filename}")
    print(f"🛰️  [DEBUG] Looking in: {satellite_path}")
    
    # Try exact match first (for most satellite filenames)
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        actual_filename = base_filename + ext
        full_path = os.path.join(satellite_path, actual_filename)
        print(f"🛰️  [DEBUG] Trying exact: {actual_filename} -> {full_path}")
        if os.path.exists(full_path):
            print(f"[DEBUG] Found satellite image: {actual_filename}")
            return send_from_directory(satellite_path, actual_filename)
    
    # If exact match fails, try partial matching for complex IDs
    try:
        import os
        all_files = os.listdir(satellite_path)
        print(f"🛰️  [DEBUG] Trying partial matching among {len(all_files)} files...")
        
        # Look for files that start with the beginning of our base_filename
        for file in all_files:
            if file.startswith(base_filename[:15]) and file.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
                print(f"[DEBUG] Found satellite image via partial match: {file}")
                return send_from_directory(satellite_path, file)
        
        print(f"🛰️  [DEBUG] No partial matches found for: {base_filename[:15]}...")
    except Exception as e:
        print(f"🛰️  [DEBUG] Error during partial matching: {e}")
    
    print(f"[DEBUG] Satellite file not found for: {base_filename} (tried exact and partial matching)")
    return jsonify({"error": f"Satellite image {filename} not found"}), 404

@app.route('/serve_surveillance_image/<path:filename>')
def serve_surveillance_image(filename):
    """Serve surveillance images from local surveillance folder with robust filename handling"""
    surveillance_path = os.path.join("data", "professional_images", "surveillance")
    
    print(f"📹 [DEBUG] Original surveillance request for: {filename}")
    
    # Remove 'surveillance_' prefix if present (database IDs include this but files don't)
    if filename.startswith('surveillance_'):
        filename = filename[len('surveillance_'):]
        print(f"📹 [DEBUG] After removing prefix: {filename}")
    
    # Remove any existing extension to get base filename
    base_filename = filename
    for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
        if base_filename.endswith(ext):
            base_filename = base_filename[:-len(ext)]
            break
    
    print(f"📹 [DEBUG] Base filename: {base_filename}")
    print(f"📹 [DEBUG] Looking in: {surveillance_path}")
    
    # Try exact match first (for complex surveillance filenames)
    for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.avi']:
        actual_filename = base_filename + ext
        full_path = os.path.join(surveillance_path, actual_filename)
        print(f"📹 [DEBUG] Trying exact: {actual_filename} -> {full_path}")
        if os.path.exists(full_path):
            print(f"[DEBUG] Found surveillance image: {actual_filename}")
            return send_from_directory(surveillance_path, actual_filename)
    
    # If exact match fails, try partial matching for complex IDs
    try:
        import os
        all_files = os.listdir(surveillance_path)
        print(f"📹 [DEBUG] Trying partial matching among {len(all_files)} files...")
        
        # Look for files that start with the beginning of our base_filename
        # This handles cases where database IDs might be truncated
        for file in all_files:
            if file.startswith(base_filename[:20]) and file.endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
                print(f"[DEBUG] Found surveillance image via partial match: {file}")
                return send_from_directory(surveillance_path, file)
        
        print(f"📹 [DEBUG] No partial matches found for: {base_filename[:20]}...")
    except Exception as e:
        print(f"📹 [DEBUG] Error during partial matching: {e}")
    
    print(f"[DEBUG] Surveillance file not found for: {base_filename} (tried exact and partial matching)")
    return jsonify({"error": f"Surveillance image {filename} not found"}), 404

if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    print("Starting Enhanced Quantum Image Retrieval Web Interface...")
    print(f"📱 Open http://localhost:{config.FLASK_PORT} in your browser")
    
    app.run(debug=config.FLASK_DEBUG, host='0.0.0.0', port=config.FLASK_PORT)
