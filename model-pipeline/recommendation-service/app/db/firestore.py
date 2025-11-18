import logging
from typing import Optional
from google.cloud.firestore import Client
from google.cloud.firestore import AsyncClient
from google.auth.exceptions import DefaultCredentialsError
from google.api_core.exceptions import GoogleAPIError
import firebase_admin
from firebase_admin import credentials, firestore

from app.config import settings

logger = logging.getLogger(__name__)

class FirestoreConnection:
    """Firestore connection manager with singleton pattern"""
    # _client: Optional[Client] = None
    _client: Optional[AsyncClient] = None
    _initialized: bool = False
    _app: Optional[firebase_admin.App] = None

    @classmethod
    async def initialize(cls):
        """Initialize Firestore connection"""
        # If client is already initialized, do not create a new one (Singleton Pattern)
        if cls._initialized:
            logger.warning("Firestore connection already initialized")
            return
        try:
            logger.info("Initializing Firestore connection")
            # Initialize Firebase Admin SDK with credentials
            cred = credentials.Certificate(settings.gcloud_json)
            cls._app = firebase_admin.initialize_app(cred)
            
            # Get Firestore client (handles connection pooling internally via gRPC)
            # cls._client = firestore.client(database_id='homiehubdb')
            cls._client = firestore.AsyncClient(
                project=cls._app.project_id,
                credentials=cls._app.credential.get_credential(),
                database='homiehubdb'
            )
            logger.info("Firestore connected successfully")
            cls._initialized = True
            
        except DefaultCredentialsError as e:
            logger.error(f"Firestore credentials error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Invalid Firestore credentials: {str(e)}")
        except GoogleAPIError as e:
            logger.error(f"Firestore API error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Could not connect to Firestore: {str(e)}")
        except FileNotFoundError as e:
            logger.error(f"Credentials file not found: {str(e)}", exc_info=True)
            raise RuntimeError(f"Firestore credentials file not found: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error connecting to Firestore: {str(e)}", exc_info=True)
            raise

    @classmethod
    def get_client(cls) -> Client:
        """Get Firestore client"""
        # Client consists of gRPC connection pool that can be reused for requests
        if not cls._initialized or cls._client is None:
            raise RuntimeError("Firestore not initialized. Call initialize() first.")
        return cls._client
    
    @classmethod
    async def close(cls):
        """Close Firestore connection and cleanup Firebase app"""
        if cls._client:
            logger.info("Closing Firestore connection")
            cls._client.close()
            cls._client = None
        if cls._app:
            try:
                firebase_admin.delete_app(cls._app)
                cls._app = None
            except Exception as e:
                logger.warning(f"Error deleting Firebase app: {str(e)}")
        cls._initialized = False

def get_firestore() -> Client:
    """Get Firestore client instance"""
    # Class methods can be directly called without creating an object
    return FirestoreConnection.get_client()