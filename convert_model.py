import onnx

# Load the model
model = onnx.load("./mediapipe_face-facedetector-float.onnx")

# Convert to opset 19
model.opset_import[0].version = 19

# Save the converted model
onnx.save(model, "./mediapipe_face-facedetector-float-opset19.onnx")
print("Model converted to opset 19 successfully!") 