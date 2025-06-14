from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Counter API",
    description="API for real-time face counting from video stream",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store the latest face count and timestamp
class FaceCountData:
    def __init__(self):
        self.count: int = 0
        self.timestamp: datetime = datetime.now()
        self.camera_id: Optional[int] = None

face_data = FaceCountData()

class FaceCountResponse(BaseModel):
    count: int
    timestamp: datetime
    camera_id: Optional[int] = None

@app.get("/face-count", response_model=FaceCountResponse)
async def get_face_count():
    """Get the current face count from the video stream."""
    return FaceCountResponse(
        count=face_data.count,
        timestamp=face_data.timestamp,
        camera_id=face_data.camera_id
    )

@app.post("/face-count")
async def update_face_count(count: int, camera_id: Optional[int] = None):
    """Update the current face count (used by the camera handler)."""
    try:
        face_data.count = count
        face_data.timestamp = datetime.now()
        face_data.camera_id = camera_id
        logger.info(f"Updated face count: {count} (Camera: {camera_id})")
        return {"status": "success", "count": count}
    except Exception as e:
        logger.error(f"Error updating face count: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the API server."""
    logger.info(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server() 