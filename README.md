# Face Counter

A real-time face detection and counting application using the Qualcomm Lightweight Face Detection model. The application can run as a standalone face counter or with an integrated API server for publishing face counts.

## Features

- Real-time face detection using ONNX Runtime
- Live video display with face count overlay
- Console output of face counts
- Optional API server for publishing face counts
- Support for multiple cameras
- Configurable API endpoint and update interval

## Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- ONNX model file (Lightweight Face Detection model)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd face-counter
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the model:

```bash
mkdir -p models

# Change to models directory
cd models

# Download the model using wget or curl
# For w8a16 quantized model:
wget https://huggingface.co/qualcomm/Lightweight-Face-Detection/resolve/main/Lightweight-Face-Detection_w8a16.onnx -O Lightweight-Face-Detection_w8a16.onnx

# Or using curl:
curl -L https://huggingface.co/qualcomm/Lightweight-Face-Detection/resolve/main/Lightweight-Face-Detection_w8a16.onnx -o Lightweight-Face-Detection_w8a16.onnx

# Verify the model was downloaded
if [ -f "Lightweight-Face-Detection_w8a16.onnx" ]; then
    echo "Model downloaded successfully"
    ls -l Lightweight-Face-Detection_w8a16.onnx
else
    echo "Error: Model download failed"
    exit 1
fi

# Return to project root
cd ..
```

5. Verify the installation:
```bash
# Check Python environment
python --version  # Should be 3.8 or higher
pip list  # Should show all required packages

# Check ONNX Runtime installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # Should list 'CPUExecutionProvider'

# Verify model exists and is accessible
if [ -f "models/Lightweight-Face-Detection_w8a16.onnx" ]; then
    echo "Model file found:"
    ls -l models/Lightweight-Face-Detection_w8a16.onnx
else
    echo "Error: Model file not found. Please run the download steps in section 4."
    exit 1
fi
```

## Usage

### Running Face Counter with API Server

The application can run in two modes:

1. **Combined Mode** (default) - Runs both the face counter and API server:

```bash
python -m face_counter \
    --model-path models/Lightweight-Face-Detection_w8a16.onnx \
    --camera-id 0 \
    --api-host 127.0.0.1 \
    --api-port 8000
```

2. **API Server Only** - Run only the API server (useful for separate face counter instances):

```bash
python -m face_counter \
    --api-only \
    --api-host 127.0.0.1 \
    --api-port 8000
```

### Command Line Arguments

- `--model-path`: Path to the ONNX model file (required)
- `--camera-id`: Camera device ID (default: 0)
- `--api-host`: Host for the API server (default: 127.0.0.1)
- `--api-port`: Port for the API server (default: 8000)
- `--api-only`: Run only the API server without the face counter

### Console Output

The face counter provides real-time updates in the console:

- Shows "Face Counter Started!" when the application starts
- Displays current face count, updating whenever it changes
- Shows "Stopping face counter..." when quitting
- Press 'q' to quit the application

### API Endpoints

The API server provides the following endpoints:

1. **GET /face-count**

   - Returns current face count and timestamp
   - Example response:
     ```json
     {
       "count": 3,
       "timestamp": "2024-02-14T12:34:56.789Z",
       "camera_id": 0
     }
     ```

2. **POST /face-count**

   - Updates the current face count
   - Parameters:
     - `count`: Number of faces detected
     - `camera_id`: ID of the camera (optional)
   - Example: `POST /face-count?count=5&camera_id=0`

3. **GET /docs**
   - Interactive API documentation (Swagger UI)

### Testing the API

You can test the API using curl:

```bash
# Get current face count
curl http://127.0.0.1:8000/face-count

# Update face count (simulated)
curl -X POST "http://127.0.0.1:8000/face-count?count=5&camera_id=0"
```

## Troubleshooting

1. **Camera Access Issues**

   - Ensure your camera is not being used by another application
   - Try a different camera ID if available
   - Check camera permissions in your operating system

2. **Model Loading Errors**

   - Verify the model file exists at the specified path
   - Ensure the model file is a valid ONNX model
   - Check if the model is compatible with your ONNX Runtime version

3. **API Server Issues**
   - Ensure the specified port is not in use
   - Check if the host is accessible
   - Verify network permissions if using a non-localhost address

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
