import cv2
import time
import logging
import requests
from typing import Optional
from datetime import datetime
from .detector import FaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraHandler:
    def __init__(self, detector, camera_id: int = 0, api_endpoint: Optional[str] = "http://127.0.0.1:8000/face-count", publish_interval: float = 1.0):
        """Initialize camera handler.
        
        Args:
            detector: Face detector instance
            camera_id: Camera device ID (default: 0)
            api_endpoint: API endpoint for publishing face count (default: local server)
            publish_interval: Interval in seconds between API updates (default: 1.0)
        """
        self.detector = detector
        self.camera_id = camera_id
        self.api_endpoint = api_endpoint
        self.publish_interval = publish_interval
        self.last_publish_time = 0
        self.cap = None
        self.last_count = 0  # Track last count for console updates
        
        if self.api_endpoint:
            logger.info(f"API endpoint configured: {self.api_endpoint}")
        else:
            logger.info("No API endpoint configured - face counts will not be published")
    
    def start(self):
        """Start video capture and face detection."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            logger.info(f"Started video capture from camera {self.camera_id}")
            print("\nFace Counter Started!")
            print("Press 'q' to quit\n")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Detect faces
                face_count = self.detector.count_faces(frame)
                
                # Print to console if count changed
                if face_count != self.last_count:
                    print(f"\rFaces detected: {face_count}", end="", flush=True)
                    self.last_count = face_count
                
                # Draw face count on frame
                cv2.putText(
                    frame,
                    f"Faces: {face_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display frame
                cv2.imshow("Face Counter", frame)
                
                # Publish count to API if endpoint is configured and interval has elapsed
                current_time = time.time()
                if self.api_endpoint and (current_time - self.last_publish_time) >= self.publish_interval:
                    self._publish_count(face_count)
                    self.last_publish_time = current_time
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping face counter...")
                    break
                    
        except Exception as e:
            logger.error(f"Error in camera handler: {str(e)}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop video capture and cleanup."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Stopped video capture")
        print("\nFace Counter Stopped")
    
    def _publish_count(self, count: int):
        """Publish face count to API endpoint."""
        if not self.api_endpoint:
            return
            
        try:
            response = requests.post(
                self.api_endpoint,
                params={"count": count, "camera_id": self.camera_id}
            )
            response.raise_for_status()
            logger.debug(f"Published face count: {count}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to publish face count: {str(e)}") 