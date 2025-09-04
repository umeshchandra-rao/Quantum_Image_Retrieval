"""
Unified Feature Extraction Service for Quantum Image Retrieval - India Region

This module provides a single, authoritative feature extraction service that handles:
1. Database population (replaces migrate_blob_to_cosmos.py)
2. Runtime feature extraction for web uploads (replaces feature_extractor.py)
3. India region Cosmos DB integration

Key Features:
- Single ResNet-50 model initialization
- Consistent 8D feature extraction 
- India region Cosmos DB configuration
- Both batch and single image processing
- Database population and runtime extraction
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Azure imports
from azure.cosmos import CosmosClient, PartitionKey
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedFeatureExtractor:
    """
    Unified Feature Extraction Service for India Region
    
    This class provides consistent feature extraction for both:
    - Database population from blob storage
    - Runtime feature extraction for web uploads
    """
    
    VERSION = "1.0.0"
    FEATURE_DIM = 8
    REGION = "Central India"
    
    def __init__(self, cosmos_endpoint=None, cosmos_key=None, database_name=None, container_name=None):
        """
        Initialize unified feature extractor for India region
        
        Args:
            cosmos_endpoint: Cosmos DB endpoint (defaults to India region)
            cosmos_key: Cosmos DB key
            database_name: Database name (defaults to quantum-images-india)
            container_name: Container name (defaults to feature-vectors-india)
        """
        # India region configuration
        self.cosmos_endpoint = cosmos_endpoint or "https://quantum-cosmos-india.documents.azure.com:443/"
        # Improved key handling with multiple fallback options
        self.cosmos_key = cosmos_key or os.getenv('COSMOS_KEY') or os.getenv('COSMOS_DB_KEY')
        self.database_name = database_name or "quantum-images-india"
        self.container_name = container_name or "feature-vectors-india"
        
        # Validate cosmos key immediately
        if not self.cosmos_key or len(self.cosmos_key) < 60:  # Valid Azure Cosmos keys are ~88 chars
            logger.error(f"Invalid Cosmos DB key: length={len(self.cosmos_key) if self.cosmos_key else 0}")
            logger.error(f"   Expected: 88-character base64 key")
            logger.error(f"   Received: {repr(self.cosmos_key[:20] + '...' if self.cosmos_key and len(self.cosmos_key) > 20 else self.cosmos_key)}")
            raise ValueError(f"Invalid Cosmos DB key: expected ~88 characters, got {len(self.cosmos_key) if self.cosmos_key else 0}")
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self._setup_resnet_model()
        
        # Initialize Azure clients (lazy initialization)
        self.cosmos_client = None
        self.database = None
        self.container = None
        
        logger.info(f"UnifiedFeatureExtractor v{self.VERSION} initialized for {self.REGION}")
        logger.info(f"Feature dimension: {self.FEATURE_DIM}D")
        logger.info(f"Device: {self.device}")
    
    def _setup_resnet_model(self):
        """Initialize ResNet-50 model for consistent 8D feature extraction"""
        try:
            # Load pre-trained ResNet-50 (same as original migration script)
            self.model = models.resnet50(pretrained=True)
            
            # Replace final layer for 8D output
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.FEATURE_DIM)
            
            # Load consistent model weights for reproducible features
            model_weights_path = "consistent_resnet50_8d.pth"
            if os.path.exists(model_weights_path):
                self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
                logger.info("Loaded consistent model weights for reproducible features")
            else:
                logger.warning("Consistent model weights not found, using random initialization")
                # Set fixed seed for reproducible random initialization
                torch.manual_seed(42)
                torch.nn.init.xavier_uniform_(self.model.fc.weight)
                torch.nn.init.zeros_(self.model.fc.bias)
                logger.info("Applied fixed random initialization with seed=42")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Image preprocessing transforms (same as original)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"ResNet-50 model initialized for {self.FEATURE_DIM}D features")
            
        except Exception as e:
            logger.error(f"Failed to setup ResNet model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _initialize_cosmos_client(self):
        """Initialize Cosmos DB client (lazy initialization)"""
        if self.cosmos_client is None:
            try:
                if not self.cosmos_key:
                    raise ValueError("COSMOS_DB_KEY environment variable not set")
                
                self.cosmos_client = CosmosClient(self.cosmos_endpoint, self.cosmos_key)
                self.database = self.cosmos_client.create_database_if_not_exists(id=self.database_name)
                self.container = self.database.create_container_if_not_exists(
                    id=self.container_name,
                    partition_key=PartitionKey(path="/image_id")
                )
                
                logger.info(f"Cosmos DB client initialized for {self.REGION}")
                logger.info(f"   Database: {self.database_name}")
                logger.info(f"   Container: {self.container_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Cosmos DB: {str(e)}")
                raise RuntimeError(f"Cosmos DB initialization failed: {e}")
    
    def extract_features_from_image(self, image_input: Union[str, Path, Image.Image, bytes]) -> Optional[List[float]]:
        """
        Extract 8D feature vector from image input
        
        Args:
            image_input: Can be:
                - File path (str or Path)
                - PIL Image object
                - Image bytes
        
        Returns:
            List of 8 float values representing the feature vector
        """
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                # File path input
                image_path = Path(image_input)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path)
            elif isinstance(image_input, bytes):
                # Bytes input (from blob storage or upload)
                image = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, 'convert'):
                # PIL Image object
                image = image_input
            else:
                raise ValueError("image_input must be file path, PIL Image, or bytes")
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms and extract features
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.squeeze().cpu().numpy()  # Shape: (8,)
            
            # Return as list for JSON serialization (NO normalization)
            return features.tolist()
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def extract_batch_features(self, image_inputs: List[Union[str, Path, bytes]]) -> List[Optional[List[float]]]:
        """
        Extract features from multiple images
        
        Args:
            image_inputs: List of image inputs (paths, bytes, etc.)
        
        Returns:
            List of feature vectors (or None for failed extractions)
        """
        results = []
        for i, image_input in enumerate(image_inputs):
            logger.info(f"Processing image {i+1}/{len(image_inputs)}")
            features = self.extract_features_from_image(image_input)
            results.append(features)
        
        return results
    
    def store_features_in_cosmos(self, image_id: str, features: List[float], metadata: Dict[str, Any] = None) -> bool:
        """
        Store feature vector in Cosmos DB
        
        Args:
            image_id: Unique identifier for the image
            features: 8D feature vector as list
            metadata: Additional metadata for the image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._initialize_cosmos_client()
            
            # Prepare document for India region
            document = {
                "id": image_id,
                "image_id": image_id,  # Partition key
                "features": features,
                "feature_dimension": len(features),
                "region": self.REGION,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "extractor_version": self.VERSION
            }
            
            # Store in Cosmos DB
            self.container.upsert_item(document)
            logger.info(f"Stored features for {image_id} in {self.REGION}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features for {image_id}: {str(e)}")
            return False
    
    def get_features_from_cosmos(self, image_id: str) -> Optional[List[float]]:
        """
        Retrieve feature vector from Cosmos DB
        
        Args:
            image_id: Image identifier
        
        Returns:
            Feature vector as list or None if not found
        """
        try:
            self._initialize_cosmos_client()
            
            # Query by image_id
            query = "SELECT c.features FROM c WHERE c.image_id = @image_id"
            parameters = [{"name": "@image_id", "value": image_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning(f"Features not found for {image_id}")
                return None
            
            return items[0]['features']
            
        except Exception as e:
            logger.error(f"Failed to retrieve features for {image_id}: {str(e)}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for India region"""
        try:
            self._initialize_cosmos_client()
            
            # Count total documents
            query = "SELECT VALUE COUNT(1) FROM c"
            count_result = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            total_count = count_result[0] if count_result else 0
            
            return {
                "total_items": total_count,
                "total_images": total_count,
                "database_name": self.database_name,
                "container_name": self.container_name,
                "region": self.REGION,
                "cosmos_endpoint": self.cosmos_endpoint,
                "extractor_version": self.VERSION
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {"error": str(e)}
    
    def search_similar_images(self, query_features: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar images using quantum-inspired similarity
        
        Args:
            query_features: 8D feature vector to search for
            top_k: Number of similar images to return
        
        Returns:
            List of similar images with similarity scores
        """
        try:
            self._initialize_cosmos_client()
            
            # Import quantum algorithm
            from src.quantum.ae_qip_algorithm import AEQIPAlgorithm
            quantum_algo = AEQIPAlgorithm()
            
            # Query all feature vectors
            query = "SELECT c.id, c.image_id, c.features FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning("No feature vectors found in database")
                return []
            
            # Calculate quantum-inspired similarities
            similarities = []
            query_features_np = np.array(query_features)
            
            for item in items:
                stored_features = np.array(item['features'])
                
                # Use quantum-inspired similarity calculation
                similarity = quantum_algo.calculate_similarity(query_features_np, stored_features)
                similarities.append({
                    'image_id': item['image_id'],
                    'similarity': float(similarity),
                    'features': item['features']
                })
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similarities)} similar images, returning top {top_k}")
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []


def migrate_images_to_cosmos_india():
    """
    Migrate images from blob storage to India region Cosmos DB
    This replaces the functionality of migrate_blob_to_cosmos.py
    """
    print("🇮🇳 MIGRATING IMAGES TO COSMOS DB - INDIA REGION")
    print("=" * 55)
    
    try:
        # Initialize unified extractor
        extractor = UnifiedFeatureExtractor()
        
        # Get images from local healthcare directory
        healthcare_dir = Path("data/professional_images/healthcare")
        if not healthcare_dir.exists():
            raise FileNotFoundError(f"Healthcare images directory not found: {healthcare_dir}")
        
        image_files = list(healthcare_dir.glob("*.jpeg")) + list(healthcare_dir.glob("*.jpg"))
        print(f"📂 Found {len(image_files)} images to process")
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_file in image_files:
            try:
                print(f"Processing: {image_file.name}")
                
                # Extract features
                features = extractor.extract_features_from_image(image_file)
                if features is None:
                    print(f"Feature extraction failed for {image_file.name}")
                    failed += 1
                    continue
                
                # Store in Cosmos DB
                image_id = image_file.name
                metadata = {
                    "filename": image_file.name,
                    "category": "healthcare",
                    "source": "local_migration",
                    "file_size": image_file.stat().st_size
                }
                
                if extractor.store_features_in_cosmos(image_id, features, metadata):
                    print(f"{image_file.name} migrated successfully")
                    successful += 1
                else:
                    print(f"Failed to store {image_file.name}")
                    failed += 1
                    
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                failed += 1
        
        print(f"\n📊 MIGRATION COMPLETE")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
        
        # Show database stats
        stats = extractor.get_database_stats()
        print(f"\n🇮🇳 India Region Database Stats:")
        print(f"   Total images: {stats.get('total_images', 0)}")
        print(f"   Database: {stats.get('database_name')}")
        print(f"   Region: {stats.get('region')}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run migration if called directly
    migrate_images_to_cosmos_india()
