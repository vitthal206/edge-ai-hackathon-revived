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

4. Download and prepare the model:
```bash
# Create models directory if it doesn't exist
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