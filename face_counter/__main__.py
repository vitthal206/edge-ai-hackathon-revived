import argparse
import logging
import os
from .detector import FaceDetector
from .camera_handler import CameraHandler
from .api_server import start_server
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_face_counter(model_path: str, camera_id: int, api_endpoint: str = "http://127.0.0.1:8000/face-count"):
    """Run the face counter with the specified model and camera."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    detector = FaceDetector(model_path)
    camera = CameraHandler(
        detector=detector,
        camera_id=camera_id,
        api_endpoint=api_endpoint
    )
    
    try:
        camera.start()
    except KeyboardInterrupt:
        logger.info("Stopping face counter...")
    finally:
        camera.stop()

def main():
    parser = argparse.ArgumentParser(description="Face Counter with API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--api-host",
        type=str,
        default="127.0.0.1",
        help="Host for the API server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for the API server (default: 8000)"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the API server without the face counter"
    )
    
    args = parser.parse_args()
    
    # Construct API endpoint URL
    api_endpoint = f"http://{args.api_host}:{args.api_port}/face-count"
    
    if args.api_only:
        # Run only the API server
        logger.info("Starting API server only...")
        start_server(host=args.api_host, port=args.api_port)
    else:
        # Start API server in a separate thread
        api_thread = threading.Thread(
            target=start_server,
            args=(args.api_host, args.api_port),
            daemon=True
        )
        api_thread.start()
        
        # Run face counter
        logger.info("Starting face counter...")
        run_face_counter(args.model_path, args.camera_id, api_endpoint)

if __name__ == "__main__":
    main() 