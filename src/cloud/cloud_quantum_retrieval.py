"""
Azure Cosmos DB Cloud-Only Quantum Image Retrieval System
Pure cloud implementation with standardized configuration
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import AzureCliCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
import logging
from dotenv import load_dotenv

# Import our standardized configuration
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudQuantumRetrieval:
    """Cloud-based quantum image retrieval using Azure Cosmos DB"""
    
    def __init__(self):
        """Initialize Azure Cosmos DB client with standardized configuration"""
        # Use standardized configuration
        self.cosmos_endpoint = config.COSMOS_ENDPOINT
        self.database_name = config.COSMOS_DATABASE
        self.container_name = config.COSMOS_CONTAINER
        
        # Get Cosmos DB key from configuration
        master_key = config.COSMOS_KEY
        
        if not self.cosmos_endpoint or not master_key:
            logger.error("Missing Azure Cosmos DB configuration")
            raise ValueError("Azure Cosmos DB credentials not found - check your .env file")
        
        logger.info(f"Using Cosmos DB endpoint: {self.cosmos_endpoint}")
        logger.info(f"Database: {self.database_name}")
        logger.info(f"Container: {self.container_name}")
        
        # Initialize Cosmos client
        self.cosmos_client = CosmosClient(
            url=self.cosmos_endpoint,
            credential=master_key
        )
        
        # Initialize database and container
        self._initialize_cosmos_resources()
    
    def _initialize_cosmos_resources(self):
        """Initialize Cosmos DB database and container"""
        try:
            # Create database if it doesn't exist (version 4.9.0)
            self.database = self.cosmos_client.create_database_if_not_exists(
                id=self.database_name
            )
            logger.info(f"Database '{self.database_name}' ready")
            
            # Create container if it doesn't exist (version 4.9.0)
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/image_id"),
                offer_throughput=400
            )
            logger.info(f"Container '{self.container_name}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB resources: {e}")
            raise RuntimeError(f"Cloud initialization failed: {e}")
    
    def store_feature_vector(self, image_id: str, features: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Store feature vector in Cosmos DB"""
        try:
            # Prepare document
            document = {
                "id": image_id,
                "image_id": image_id,  # Partition key
                "features": features.tolist(),
                "feature_dimension": len(features),
                "metadata": metadata or {},
                "timestamp": str(np.datetime64('now'))
            }
            
            # Store in Cosmos DB using version 4.9.0 API
            self.container.upsert_item(document)
            logger.info(f"Stored features for image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store features for {image_id}: {e}")
            raise RuntimeError(f"Cloud storage failed: {e}")
    
    def get_feature_vector(self, image_id: str) -> Optional[np.ndarray]:
        """Retrieve feature vector from Cosmos DB"""
        try:
            # Query for the specific image by image_id field (not document id)
            query = "SELECT c.features FROM c WHERE c.image_id = @image_id"
            parameters = [{"name": "@image_id", "value": image_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning(f"Features not found for image {image_id}")
                return None
            
            # Convert back to numpy array
            features = np.array(items[0]['features'])
            logger.info(f"Retrieved features for image {image_id}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve features for {image_id}: {e}")
            raise RuntimeError(f"Cloud retrieval failed: {e}")
    
    def search_similar_images(self, query_features: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar images using quantum-inspired similarity"""
        try:
            # Import quantum algorithm
            from src.quantum.ae_qip_algorithm import AEQIPAlgorithm
            quantum_algo = AEQIPAlgorithm()
            
            # Query all feature vectors from Cosmos DB using version 4.9.0 API
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
            
            for item in items:
                stored_features = np.array(item['features'])
                
                # Use quantum-inspired similarity calculation
                similarity = quantum_algo.calculate_similarity(query_features, stored_features)
                similarities.append((item['image_id'], float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            results = similarities[:top_k]
            logger.info(f"Found {len(results)} similar images")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar images: {e}")
            raise RuntimeError(f"Cloud search failed: {e}")
    
    def list_all_images(self) -> List[str]:
        """Get list of all stored image IDs"""
        try:
            query = "SELECT c.image_id FROM c"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            image_ids = [item['image_id'] for item in items]
            logger.info(f"Found {len(image_ids)} images in database")
            return image_ids
            
        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            raise RuntimeError(f"Cloud listing failed: {e}")
    
    def delete_image(self, image_id: str) -> bool:
        """Delete image features from Cosmos DB"""
        try:
            self.container.delete_item(
                item=image_id,
                partition_key=image_id
            )
            logger.info(f"Deleted features for image {image_id}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Image {image_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            raise RuntimeError(f"Cloud deletion failed: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Count total documents
            query = "SELECT VALUE COUNT(1) FROM c"
            count_result = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            total_count = count_result[0] if count_result else 0
            
            return {
                "total_items": total_count,
                "total_images": total_count,  # Alias for backward compatibility
                "database_name": self.database_name,
                "container_name": self.container_name,
                "cosmos_endpoint": self.cosmos_endpoint,
                "region": self.database_name  # Add region info
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise RuntimeError(f"Cloud stats failed: {e}")

def populate_cosmos_with_local_images():
    """Populate Cosmos DB with existing local images (one-time migration)"""
    print("Starting migration of local images to Cosmos DB...")
    
    try:
        # Initialize cloud retrieval
        cloud_retrieval = CloudQuantumRetrieval()
        
        # Import necessary modules
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from research.enhanced_quantum_similarity import EnhancedQuantumSimilarity
        
        # Initialize feature extractor
        extractor = EnhancedQuantumSimilarity()
        
        # Process images from data directory
        images_dir = "data/professional_images"
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return
        
        processed_count = 0
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, filename)
                
                try:
                    # Extract features
                    features = extractor.extract_features(image_path)
                    
                    # Store in Cosmos DB
                    cloud_retrieval.store_feature_vector(
                        image_id=filename,
                        features=features,
                        metadata={"source": "professional_images", "filename": filename}
                    )
                    
                    processed_count += 1
                    print(f"Processed {processed_count}: {filename}")
                    
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
        
        print(f"Migration complete! Processed {processed_count} images")
        
        # Show final stats
        stats = cloud_retrieval.get_database_stats()
        print(f"Database stats: {stats}")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    # Test the cloud retrieval system
    try:
        cloud_retrieval = CloudQuantumRetrieval()
        stats = cloud_retrieval.get_database_stats()
        print(f"Cloud system ready: {stats}")
        
        # Run migration if needed
        if stats['total_images'] == 0:
            populate_cosmos_with_local_images()
            
    except Exception as e:
        print(f"Cloud system test failed: {e}")
