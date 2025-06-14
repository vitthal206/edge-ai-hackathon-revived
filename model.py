import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> List[int]:
    """
    Basic NMS implementation for overlapping box filtering.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        order = order[1:][iou < iou_threshold]

    return keep


class MediaPipeFaceDetector:
    def __init__(self, model_path: str):
        """
        Initialize the face detector with ONNX Runtime and load anchors.
        """
        self.input_size = (256, 256)
        self.confidence_threshold = 0.9
        self.iou_threshold = 0.3

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.anchors = self._generate_anchors()

        logger.info(f"Model loaded successfully from {model_path}")

    def _generate_anchors(self) -> np.ndarray:
        """
        Generate anchor boxes matching MediaPipe config for input size.
        """
        # Using config that matches MediaPipe face detection
        strides = [8, 16, 16, 16, 16, 16]
        anchor_count = [2, 6, 6, 6, 6, 6]
        feature_map_sizes = [(32, 32), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16)]
        anchors = []

        for f_idx, (f_h, f_w) in enumerate(feature_map_sizes):
            for y in range(f_h):
                for x in range(f_w):
                    for a_idx in range(anchor_count[f_idx]):
                        anchors.append([
                            (x + 0.5) / f_w,
                            (y + 0.5) / f_h
                        ])

        return np.array(anchors)  # shape: (1344, 2)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.input_size, Image.Resampling.BILINEAR)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def decode_boxes(self, raw_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """
        Decode bounding boxes from raw predictions using anchors.
        """
        boxes = np.zeros((raw_boxes.shape[0], 4))
        x_center = raw_boxes[:, 0] / 10.0 * anchors[:, 0] + anchors[:, 0]
        y_center = raw_boxes[:, 1] / 10.0 * anchors[:, 1] + anchors[:, 1]
        w = np.exp(raw_boxes[:, 2] / 5.0) * anchors[:, 0]
        h = np.exp(raw_boxes[:, 3] / 5.0) * anchors[:, 1]

        boxes[:, 0] = x_center - w / 2
        boxes[:, 1] = y_center - h / 2
        boxes[:, 2] = x_center + w / 2
        boxes[:, 3] = y_center + h / 2
        return boxes

    def detect_faces(self, image: Image.Image) -> Dict:
        input_tensor = self.preprocess_image(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        detections = outputs[0][0]  # shape: (896, 16)
        raw_scores = detections[:, 15]  # Confidence
        raw_boxes = detections[:, 0:4]

        valid_indices = raw_scores >= self.confidence_threshold
        scores = raw_scores[valid_indices]
        raw_boxes = raw_boxes[valid_indices]

        if len(scores) == 0:
            return {"face_count": 0}

        filtered_anchors = self.anchors[:len(detections)][valid_indices]
        decoded_boxes = self.decode_boxes(raw_boxes, filtered_anchors)

        # Convert to pixel coordinates
        orig_w, orig_h = image.size
        decoded_boxes[:, [0, 2]] *= orig_w
        decoded_boxes[:, [1, 3]] *= orig_h

        keep_indices = non_max_suppression(decoded_boxes, scores, self.iou_threshold)
        face_count = len(keep_indices)

        return {"face_count": face_count}

    def __call__(self, image: Image.Image) -> Dict:
        return self.detect_faces(image)
