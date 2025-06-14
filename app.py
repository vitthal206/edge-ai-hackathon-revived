from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import cv2
import numpy as np
from model import MediaPipeFaceDetector
import logging
import threading
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Detection API",
    description="API for detecting faces using MediaPipe Face Detection model with ONNX Runtime",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared face count variable with thread-safe access
latest_face_count = 0
lock = threading.Lock()

# Initialize face detector
face_detector = None
try:
    face_detector = MediaPipeFaceDetector(model_path="./mediapipe_face-facedetector-float.onnx")
    logger.info("Face detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize face detector: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Face Detection API is running",
        "endpoints": {
            "/detect": "POST - Get latest face count (updated every 5 minutes)",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if face_detector else "degraded",
        "model_loaded": face_detector is not None
    }

@app.post("/detect")
async def detect_faces():
    """
    Return the most recently calculated face count.
    """
    with lock:
        return {
            "success": True,
            "face_count": latest_face_count
        }

def face_count_worker():
    """
    Background worker that updates face count every 5 minutes.
    """
    global latest_face_count

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Unable to open camera for background detection")
        return

    try:
        while True:
            success, frame = cap.read()
            if success:
                # Convert to RGB and PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                if face_detector:
                    result = face_detector(image)
                    with lock:
                        latest_face_count = result.get("face_count", 0)
                    logger.info(f"Updated face count: {latest_face_count}")
                else:
                    logger.warning("Face detector not available during detection.")
            else:
                logger.warning("Failed to read frame from camera.")

            # Sleep for 5 minutes (300 seconds)
            time.sleep(300)
    finally:
        cap.release()

@app.on_event("startup")
def start_background_worker():
    """
    Start the face detection background thread on app startup.
    """
    if face_detector:
        thread = threading.Thread(target=face_count_worker, daemon=True)
        thread.start()
        logger.info("Started background face detection worker")
    else:
        logger.error("Face detector is not initialized. Background worker not started.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
