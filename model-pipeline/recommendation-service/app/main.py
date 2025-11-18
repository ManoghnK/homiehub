from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import sys

from app.db.firestore import FirestoreConnection
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Application starting...")
    try:
        await FirestoreConnection.initialize()
        logger.info("All connections initialized")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    try:
        await FirestoreConnection.close()
        logger.info("All connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Recommendation service",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"service": "Recommendation Service", "version": "1.0.0"}

from app.api import recommendation
app.include_router(recommendation.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)