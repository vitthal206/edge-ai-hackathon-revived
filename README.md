# Face Detection API with MediaPipe and ONNX Runtime

This API service uses the MediaPipe Face Detection model from Qualcomm AI Hub, running on ONNX Runtime with QNN execution provider for NPU acceleration.

## Prerequisites

- Python 3.8 or higher
- Qualcomm NPU-enabled device
- ONNX Runtime with QNN provider installed

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the model:
   The model will be automatically downloaded on first run, or you can manually download it from Qualcomm AI Hub.

## Running the API

Start the server:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /

Returns basic API information and available endpoints.

### GET /health

Health check endpoint to verify API and model status.

### POST /detect

Upload an image to detect faces.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**

```json
{
  "success": true,
  "face_count": 2,
  "face_locations": [
    // Array of face bounding boxes
  ]
}
```

## Example Usage

Using curl:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/detect
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Notes

- The API uses ONNX Runtime with QNN execution provider for NPU acceleration
- Input images are automatically resized to 256x256 pixels
- The model supports real-time face detection with sub-millisecond processing
- Face detection includes facial landmarks (eyes, nose, mouth)

## License

This project uses the MediaPipe Face Detection model under the AI-HUB-MODELS-LICENSE from Qualcomm AI Hub.
