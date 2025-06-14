import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str):
        """Initialize the face detector with ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize ONNX model with CPU provider."""
        try:
            # Configure ONNX Runtime session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create session with CPU provider
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Get model metadata
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Get output metadata
            self.output_names = [output.name for output in self.session.get_outputs()]
            logger.info(f"Model loaded successfully with input shape: {self.input_shape}")
            logger.info(f"Model outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to model input size (assuming 640x480 as per model card)
        resized = cv2.resize(frame, (640, 480))
        
        # Convert BGR to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Convert to uint16 and normalize to 0-65535 range
        # Assuming input image is 8-bit (0-255), we'll scale it to 16-bit
        input_tensor = gray.astype(np.uint16) * 257  # 257 = 65535/255 to scale to full uint16 range
        
        # Add channel dimension (HWC to CHW)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add channel dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def _find_peaks(self, heatmap: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
        """Find peaks in the heatmap that are above threshold.
        
        Args:
            heatmap: 2D numpy array of heatmap values
            threshold: Minimum value to consider as a peak
            
        Returns:
            List of (x, y) coordinates of peaks
        """
        # Normalize heatmap to 0-1 range
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Find local maxima
        peaks = []
        h, w = heatmap.shape
        
        # Use a 3x3 window to find local maxima
        for y in range(1, h-1):
            for x in range(1, w-1):
                if heatmap[y, x] > threshold:
                    # Check if it's a local maximum
                    window = heatmap[y-1:y+2, x-1:x+2]
                    if heatmap[y, x] == window.max():
                        peaks.append((x, y))
        
        return peaks
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the input frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected faces
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(frame)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            
            # Get heatmap from output (shape: [1, 1, 60, 80])
            heatmap = outputs[0][0, 0]  # Remove batch and channel dimensions
            
            # Find peaks in heatmap
            peaks = self._find_peaks(heatmap, threshold=0.5)
            
            # Convert peaks to bounding boxes
            valid_boxes = []
            h, w = frame.shape[:2]
            scale_x = w / heatmap.shape[1]
            scale_y = h / heatmap.shape[0]
            
            # Fixed box size (can be adjusted based on your needs)
            box_width = int(100 * scale_x)
            box_height = int(100 * scale_y)
            
            for x, y in peaks:
                # Convert heatmap coordinates to image coordinates
                img_x = int(x * scale_x)
                img_y = int(y * scale_y)
                
                # Create box centered on peak
                x1 = max(0, img_x - box_width // 2)
                y1 = max(0, img_y - box_height // 2)
                x2 = min(w, img_x + box_width // 2)
                y2 = min(h, img_y + box_height // 2)
                
                w = x2 - x1
                h = y2 - y1
                
                if w > 0 and h > 0:  # Filter out invalid boxes
                    valid_boxes.append((x1, y1, w, h))
            
            return valid_boxes
            
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            logger.error(f"Heatmap shape: {heatmap.shape if 'heatmap' in locals() else 'not available'}")
            return []
    
    def count_faces(self, frame: np.ndarray) -> int:
        """Count number of faces in the input frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Number of faces detected
        """
        boxes = self.detect_faces(frame)
        return len(boxes) 