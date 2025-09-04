"""
Satellite Images Uploader

This script uploads satellite images from data/professional_images/satellite/
to both Azure Blob Storage and Cosmos DB with feature extraction.

Usage:
    python satellite_uploader.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Azure imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmos import CosmosClient, PartitionKey
from azure.core.exceptions import ResourceExistsError, AzureError

# Add src to path for imports
sys.path.append('src')

# Import configuration and feature extractor
import config
from unified_feature_extractor import UnifiedFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('satellite_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SatelliteUploader:
    """Satellite Images Uploader for Azure Blob Storage and Cosmos DB"""
    
    def __init__(self):
        """Initialize the uploader with Azure clients"""
        self.category = "satellite"
        self.source_folder = "data/professional_images/satellite"
        
        # Validate configuration
        config.validate_config()
        
        # Initialize Azure clients
        self._init_blob_client()
        self._init_cosmos_client()
        
        # Initialize feature extractor
        self.feature_extractor = UnifiedFeatureExtractor(
            cosmos_endpoint=config.COSMOS_ENDPOINT,
            cosmos_key=config.COSMOS_KEY,
            database_name=config.COSMOS_DATABASE,
            container_name=config.COSMOS_CONTAINER
        )
        
        logger.info(f"SatelliteUploader initialized for category: {self.category}")
    
    def _init_blob_client(self):
        """Initialize Azure Blob Storage client"""
        try:
            if not config.AZURE_STORAGE_CONNECTION_STRING:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")
            
            self.blob_service_client = BlobServiceClient.from_connection_string(
                config.AZURE_STORAGE_CONNECTION_STRING
            )
            
            # Create container if it doesn't exist
            self.container_name = f"quantum-images-{self.category}"
            try:
                self.container_client = self.blob_service_client.create_container(
                    self.container_name
                )
                logger.info(f"✅ Created new container: {self.container_name}")
            except ResourceExistsError:
                self.container_client = self.blob_service_client.get_container_client(
                    self.container_name
                )
                logger.info(f"✅ Using existing container: {self.container_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Blob Storage: {e}")
            raise
    
    def _init_cosmos_client(self):
        """Initialize Azure Cosmos DB client"""
        try:
            self.cosmos_client = CosmosClient(
                config.COSMOS_ENDPOINT,
                config.COSMOS_KEY
            )
            
            # Get database and container
            self.database = self.cosmos_client.get_database_client(config.COSMOS_DATABASE)
            self.cosmos_container = self.database.get_container_client(config.COSMOS_CONTAINER)
            
            logger.info(f"✅ Connected to Cosmos DB: {config.COSMOS_DATABASE}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cosmos DB: {e}")
            raise
    
    def get_image_files(self) -> List[Path]:
        """Get list of image files in the satellite folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        source_path = Path(self.source_folder)
        if not source_path.exists():
            logger.warning(f"Source folder does not exist: {source_path}")
            return []
        
        for file_path in source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files in {self.source_folder}")
        return sorted(image_files)
    
    def upload_image_to_blob(self, image_path: Path) -> Optional[str]:
        """Upload single image to Azure Blob Storage"""
        try:
            blob_name = f"{self.category}/{image_path.name}"
            
            # Check if blob already exists
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            if blob_client.exists():
                logger.info(f"⚠️  Blob already exists: {blob_name}")
                return blob_client.url
            
            # Upload the image
            with open(image_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"✅ Uploaded to blob: {blob_name}")
            return blob_client.url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {image_path.name} to blob: {e}")
            return None
    
    def store_in_cosmos(self, image_path: Path, blob_url: str, features: List[float]) -> bool:
        """Store image metadata and features in Cosmos DB"""
        try:
            image_id = f"{self.category}_{image_path.stem}"
            
            # Prepare document
            document = {
                "id": image_id,
                "image_id": image_id,  # Partition key
                "category": self.category,
                "filename": image_path.name,
                "blob_url": blob_url,
                "features": features,
                "feature_dimension": len(features),
                "metadata": {
                    "original_path": str(image_path),
                    "file_size": image_path.stat().st_size,
                    "upload_timestamp": datetime.now().isoformat(),
                    "extractor_version": self.feature_extractor.VERSION,
                    "coordinates": "17.3850°N, 78.4867°E",  # Hyderabad coordinates
                    "satellite_type": "simulated",
                    "resolution": "high"
                },
                "timestamp": datetime.now().isoformat(),
                "region": "Central India"
            }
            
            # Store in Cosmos DB
            self.cosmos_container.upsert_item(document)
            logger.info(f"✅ Stored in Cosmos DB: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store {image_path.name} in Cosmos DB: {e}")
            return False
    
    def upload_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Upload a single image to both blob storage and cosmos db"""
        result = {
            "filename": image_path.name,
            "blob_success": False,
            "cosmos_success": False,
            "blob_url": None,
            "error": None
        }
        
        try:
            logger.info(f"Processing: {image_path.name}")
            
            # Extract features first
            features = self.feature_extractor.extract_features_from_image(image_path)
            if not features:
                result["error"] = "Feature extraction failed"
                return result
            
            # Upload to blob storage
            blob_url = self.upload_image_to_blob(image_path)
            if blob_url:
                result["blob_success"] = True
                result["blob_url"] = blob_url
                
                # Store in Cosmos DB
                if self.store_in_cosmos(image_path, blob_url, features):
                    result["cosmos_success"] = True
            else:
                result["error"] = "Blob upload failed"
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Error processing {image_path.name}: {e}")
        
        return result
    
    def upload_all_images(self) -> Dict[str, Any]:
        """Upload all images in the satellite folder"""
        logger.info("Starting bulk upload of all satellite images...")
        
        image_files = self.get_image_files()
        if not image_files:
            logger.warning("No image files found to upload")
            return {"total": 0, "success": 0, "failed": 0, "results": []}
        
        results = []
        success_count = 0
        
        for image_path in image_files:
            result = self.upload_single_image(image_path)
            results.append(result)
            
            if result["blob_success"] and result["cosmos_success"]:
                success_count += 1
        
        summary = {
            "total": len(image_files),
            "success": success_count,
            "failed": len(image_files) - success_count,
            "results": results
        }
        
        logger.info(f"✅ Upload complete: {success_count}/{len(image_files)} successful")
        return summary
    
    def upload_selected_images(self, selected_names: List[str]) -> Dict[str, Any]:
        """Upload only selected images by filename"""
        logger.info(f"Starting upload of selected satellite images: {selected_names}")
        
        image_files = self.get_image_files()
        selected_files = []
        
        # Find selected files
        for image_path in image_files:
            if image_path.name in selected_names:
                selected_files.append(image_path)
        
        if not selected_files:
            logger.warning(f"No matching files found for: {selected_names}")
            return {"total": 0, "success": 0, "failed": 0, "results": []}
        
        results = []
        success_count = 0
        
        for image_path in selected_files:
            result = self.upload_single_image(image_path)
            results.append(result)
            
            if result["blob_success"] and result["cosmos_success"]:
                success_count += 1
        
        summary = {
            "total": len(selected_files),
            "success": success_count,
            "failed": len(selected_files) - success_count,
            "results": results
        }
        
        logger.info(f"✅ Upload complete: {success_count}/{len(selected_files)} successful")
        return summary
    
    def delete_image_from_blob(self, image_name: str) -> bool:
        """Delete single image from Azure Blob Storage"""
        try:
            blob_name = f"{self.category}/{image_name}"
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Check if blob exists
            if not blob_client.exists():
                logger.warning(f"⚠️  Blob not found: {blob_name}")
                return False
            
            # Delete the blob
            blob_client.delete_blob()
            logger.info(f"✅ Deleted from blob: {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete {image_name} from blob: {e}")
            return False
    
    def delete_from_cosmos(self, image_name: str) -> bool:
        """Delete image document from Cosmos DB"""
        try:
            # Create image_id from filename
            image_stem = Path(image_name).stem
            image_id = f"{self.category}_{image_stem}"
            
            # Delete from Cosmos DB
            self.cosmos_container.delete_item(
                item=image_id,
                partition_key=image_id
            )
            logger.info(f"✅ Deleted from Cosmos DB: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete {image_name} from Cosmos DB: {e}")
            return False
    
    def delete_single_image(self, image_name: str) -> Dict[str, Any]:
        """Delete a single image from both blob storage and cosmos db"""
        result = {
            "filename": image_name,
            "blob_success": False,
            "cosmos_success": False,
            "error": None
        }
        
        try:
            logger.info(f"Deleting: {image_name}")
            
            # Delete from blob storage
            if self.delete_image_from_blob(image_name):
                result["blob_success"] = True
            
            # Delete from Cosmos DB
            if self.delete_from_cosmos(image_name):
                result["cosmos_success"] = True
            
            if not result["blob_success"] and not result["cosmos_success"]:
                result["error"] = "Failed to delete from both blob and cosmos"
            elif not result["blob_success"]:
                result["error"] = "Failed to delete from blob storage"
            elif not result["cosmos_success"]:
                result["error"] = "Failed to delete from cosmos DB"
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Error deleting {image_name}: {e}")
        
        return result
    
    def delete_all_images(self) -> Dict[str, Any]:
        """Delete all images from both blob storage and cosmos db"""
        logger.info("Starting bulk deletion of all satellite images...")
        
        # Get all blobs in the container with satellite prefix
        try:
            blobs = self.blob_service_client.get_container_client(
                self.container_name
            ).list_blobs(name_starts_with=f"{self.category}/")
            
            blob_names = [blob.name.split('/')[-1] for blob in blobs]  # Extract filename from blob path
            
            if not blob_names:
                logger.warning("No images found in blob storage to delete")
                return {"total": 0, "success": 0, "failed": 0, "results": []}
            
            results = []
            success_count = 0
            
            for image_name in blob_names:
                result = self.delete_single_image(image_name)
                results.append(result)
                
                if result["blob_success"] and result["cosmos_success"]:
                    success_count += 1
            
            summary = {
                "total": len(blob_names),
                "success": success_count,
                "failed": len(blob_names) - success_count,
                "results": results
            }
            
            logger.info(f"✅ Deletion complete: {success_count}/{len(blob_names)} successful")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to list blobs for deletion: {e}")
            return {"total": 0, "success": 0, "failed": 0, "results": [], "error": str(e)}
    
    def delete_selected_images(self, selected_names: List[str]) -> Dict[str, Any]:
        """Delete only selected images by filename"""
        logger.info(f"Starting deletion of selected satellite images: {selected_names}")
        
        results = []
        success_count = 0
        
        for image_name in selected_names:
            result = self.delete_single_image(image_name.strip())
            results.append(result)
            
            if result["blob_success"] and result["cosmos_success"]:
                success_count += 1
        
        summary = {
            "total": len(selected_names),
            "success": success_count,
            "failed": len(selected_names) - success_count,
            "results": results
        }
        
        logger.info(f"✅ Deletion complete: {success_count}/{len(selected_names)} successful")
        return summary
    
    def view_blob_images(self) -> List[Dict[str, Any]]:
        """View all images currently in Azure Blob Storage"""
        try:
            blobs = self.blob_service_client.get_container_client(
                self.container_name
            ).list_blobs(name_starts_with=f"{self.category}/")
            
            blob_info = []
            for blob in blobs:
                # Get blob properties
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob.name
                )
                properties = blob_client.get_blob_properties()
                
                blob_data = {
                    "name": blob.name.split('/')[-1],  # Extract filename
                    "full_path": blob.name,
                    "size_mb": round(blob.size / (1024 * 1024), 2),
                    "last_modified": properties.last_modified,
                    "url": blob_client.url,
                    "content_type": properties.content_settings.content_type
                }
                blob_info.append(blob_data)
            
            logger.info(f"Found {len(blob_info)} images in blob storage")
            return blob_info
            
        except Exception as e:
            logger.error(f"❌ Failed to list blob images: {e}")
            return []
    
    def view_cosmos_images(self) -> List[Dict[str, Any]]:
        """View all images currently in Cosmos DB"""
        try:
            # Query for all documents with this category
            query = f"SELECT * FROM c WHERE c.category = '{self.category}'"
            
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            
            # Note: Legacy documents are healthcare-specific (IM-* pattern) so don't include them for satellite
            
            cosmos_info = []
            
            # Process current format documents
            for item in items:
                cosmos_data = {
                    "id": item.get("id"),
                    "filename": item.get("filename"),
                    "blob_url": item.get("blob_url"),
                    "feature_dimension": item.get("feature_dimension"),
                    "upload_timestamp": item.get("metadata", {}).get("upload_timestamp"),
                    "file_size_bytes": item.get("metadata", {}).get("file_size"),
                    "file_size_mb": round(item.get("metadata", {}).get("file_size", 0) / (1024 * 1024), 2) if item.get("metadata", {}).get("file_size") else 0,
                    "extractor_version": item.get("metadata", {}).get("extractor_version"),
                    "satellite_metadata": item.get("satellite_metadata", {}),
                    "document_type": "current"
                }
                cosmos_info.append(cosmos_data)
            
            logger.info(f"Found {len(items)} current satellite images in Cosmos DB")
            return cosmos_info
            
        except Exception as e:
            logger.error(f"❌ Failed to list cosmos images: {e}")
            return []
    
    def count_blob_images(self) -> int:
        """Count images in Azure Blob Storage (optimized for performance)"""
        try:
            blobs = self.blob_service_client.get_container_client(
                self.container_name
            ).list_blobs(name_starts_with=f"{self.category}/")
            
            count = sum(1 for _ in blobs)
            logger.info(f"Found {count} images in blob storage")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to count blob images: {e}")
            return 0
    
    def count_cosmos_images(self) -> int:
        """Count images in Cosmos DB (optimized for performance)"""
        try:
            # Count documents for this category
            query = f"SELECT VALUE COUNT(1) FROM c WHERE c.category = '{self.category}'"
            result = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            count = result[0] if result else 0
            
            logger.info(f"Found {count} satellite images in Cosmos DB")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to count cosmos images: {e}")
            return 0

    def view_current_images(self) -> Dict[str, Any]:
        """View current image counts from both blob storage and cosmos db (optimized)"""
        print("\n🔍 Retrieving current image counts from Azure...")
        
        # Get blob storage count
        print("📁 Counting images in Azure Blob Storage...")
        blob_count = self.count_blob_images()
        
        # Get Cosmos DB count
        print("🗄️  Counting images in Azure Cosmos DB...")
        cosmos_count = self.count_cosmos_images()
        
        # Display results
        print(f"\n📊 Current Images Summary:")
        print(f"   📁 Blob Storage: {blob_count} images")
        print(f"   🗄️  Cosmos DB: {cosmos_count} images")
        
        # Check for basic consistency
        if blob_count != cosmos_count:
            print(f"\n⚠️  Potential Data Inconsistency:")
            print(f"   Blob Storage has {blob_count} images")
            print(f"   Cosmos DB has {cosmos_count} documents")
            if blob_count > cosmos_count:
                print(f"   → {blob_count - cosmos_count} images may be missing from Cosmos DB")
            else:
                print(f"   → {cosmos_count - blob_count} documents may be orphaned in Cosmos DB")
        else:
            print(f"\n✅ Data consistency: Blob storage and Cosmos documents are in sync")
        
        return {
            "blob_count": blob_count,
            "cosmos_count": cosmos_count
        }

    def view_detailed_images(self) -> Dict[str, Any]:
        """View detailed information about all current images (original method)"""
        print("\n🔍 Retrieving detailed images information from Azure...")
        
        # Get blob storage images
        print("📁 Checking Azure Blob Storage...")
        blob_images = self.view_blob_images()
        
        # Get Cosmos DB images
        print("🗄️  Checking Azure Cosmos DB...")
        cosmos_images = self.view_cosmos_images()
        
        # Display results
        print(f"\n📊 Detailed Images Information:")
        print(f"   Blob Storage: {len(blob_images)} images")
        print(f"   Cosmos DB: {len(cosmos_images)} images")
        
        if blob_images:
            print(f"\n📁 Images in Azure Blob Storage:")
            print("   " + "="*80)
            for i, img in enumerate(blob_images, 1):
                print(f"   {i:2d}. {img['name']}")
                print(f"       Size: {img['size_mb']} MB | Modified: {img['last_modified']}")
                print(f"       URL: {img['url'][:60]}...")
                if i < len(blob_images):
                    print()
        
        if cosmos_images:
            print(f"\n🗄️  Images in Azure Cosmos DB:")
            print("   " + "="*80)
            for i, img in enumerate(cosmos_images, 1):
                print(f"   {i:2d}. {img['filename']} (ID: {img['id']})")
                print(f"       Size: {img['file_size_mb']} MB | Features: {img['feature_dimension']}D")
                if img['upload_timestamp'] and img['upload_timestamp'] != 'Unknown':
                    print(f"       Uploaded: {img['upload_timestamp']}")
                # Show satellite-specific metadata
                if img['satellite_metadata']:
                    sat_meta = img['satellite_metadata']
                    print(f"       Satellite: {sat_meta.get('satellite_type', 'N/A')} | Coords: {sat_meta.get('coordinates', 'N/A')}")
                if i < len(cosmos_images):
                    print()
        
        # Check for mismatches
        blob_names = {img['name'] for img in blob_images}
        cosmos_names = {img['filename'] for img in cosmos_images}
        
        only_in_blob = blob_names - cosmos_names
        only_in_cosmos = cosmos_names - blob_names
        
        if only_in_blob or only_in_cosmos:
            print(f"\n⚠️  Data Inconsistencies Found:")
            if only_in_blob:
                print(f"   Only in Blob Storage: {list(only_in_blob)}")
            if only_in_cosmos:
                print(f"   Only in Cosmos DB: {list(only_in_cosmos)}")
        else:
            print(f"\n✅ Data consistency: All images exist in both storage systems")
        
        return {
            "blob_images": blob_images,
            "cosmos_images": cosmos_images,
            "blob_count": len(blob_images),
            "cosmos_count": len(cosmos_images),
            "only_in_blob": list(only_in_blob),
            "only_in_cosmos": list(only_in_cosmos)
        }


def main():
    """Main function to run the satellite uploader"""
    try:
        uploader = SatelliteUploader()
        
        print("🛰️  Satellite Images Uploader")
        print("=" * 50)
        print("Available options:")
        print("1. Upload all satellite images")
        print("2. Upload selected images")
        print("3. List available images")
        print("4. Delete all images")
        print("5. Delete selected images")
        print("6. View current images in Azure (counts only)")
        print("7. View detailed images in Azure (full data)")
        print("8. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == "1":
                print("\nUploading all satellite images...")
                summary = uploader.upload_all_images()
                print(f"\n📊 Summary:")
                print(f"   Total: {summary['total']}")
                print(f"   Success: {summary['success']}")
                print(f"   Failed: {summary['failed']}")
                break
                
            elif choice == "2":
                # List available images first
                image_files = uploader.get_image_files()
                if not image_files:
                    print("❌ No images found in satellite folder")
                    continue
                
                print(f"\nAvailable images ({len(image_files)}):")
                for i, img_path in enumerate(image_files, 1):
                    print(f"   {i}. {img_path.name}")
                
                # Get selected images
                selected_input = input("\nEnter image names (comma-separated) or 'all': ").strip()
                
                if selected_input.lower() == 'all':
                    selected_names = [img.name for img in image_files]
                else:
                    selected_names = [name.strip() for name in selected_input.split(',')]
                
                print(f"\nUploading selected images: {selected_names}")
                summary = uploader.upload_selected_images(selected_names)
                print(f"\n📊 Summary:")
                print(f"   Total: {summary['total']}")
                print(f"   Success: {summary['success']}")
                print(f"   Failed: {summary['failed']}")
                break
                
            elif choice == "3":
                image_files = uploader.get_image_files()
                print(f"\nFound {len(image_files)} satellite images:")
                for i, img_path in enumerate(image_files, 1):
                    size_mb = img_path.stat().st_size / (1024 * 1024)
                    print(f"   {i}. {img_path.name} ({size_mb:.2f} MB)")
                
            elif choice == "4":
                confirm = input("\n⚠️  Are you sure you want to delete ALL satellite images? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    print("\n🗑️  Deleting all satellite images...")
                    summary = uploader.delete_all_images()
                    print(f"\n📊 Deletion Summary:")
                    print(f"   Total: {summary['total']}")
                    print(f"   Success: {summary['success']}")
                    print(f"   Failed: {summary['failed']}")
                    if 'error' in summary:
                        print(f"   Error: {summary['error']}")
                else:
                    print("❌ Deletion cancelled")
                break
                
            elif choice == "5":
                # List images in blob storage first
                try:
                    blobs = uploader.blob_service_client.get_container_client(
                        uploader.container_name
                    ).list_blobs(name_starts_with=f"{uploader.category}/")
                    
                    blob_names = [blob.name.split('/')[-1] for blob in blobs]
                    
                    if not blob_names:
                        print("❌ No images found in blob storage")
                        continue
                    
                    print(f"\nImages in blob storage ({len(blob_names)}):")
                    for i, img_name in enumerate(blob_names, 1):
                        print(f"   {i}. {img_name}")
                    
                    # Get selected images to delete
                    selected_input = input("\nEnter image names to delete (comma-separated): ").strip()
                    
                    if not selected_input:
                        print("❌ No images selected")
                        continue
                    
                    selected_names = [name.strip() for name in selected_input.split(',')]
                    
                    confirm = input(f"\n⚠️  Are you sure you want to delete {len(selected_names)} images? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        print(f"\n🗑️  Deleting selected images: {selected_names}")
                        summary = uploader.delete_selected_images(selected_names)
                        print(f"\n📊 Deletion Summary:")
                        print(f"   Total: {summary['total']}")
                        print(f"   Success: {summary['success']}")
                        print(f"   Failed: {summary['failed']}")
                    else:
                        print("❌ Deletion cancelled")
                    
                except Exception as e:
                    print(f"❌ Error listing images: {e}")
                break
                
            elif choice == "6":
                uploader.view_current_images()
                
            elif choice == "7":
                uploader.view_detailed_images()
                
            elif choice == "8":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-8.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Upload cancelled by user")
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
