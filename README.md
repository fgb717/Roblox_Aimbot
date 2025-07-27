# Roblox_Aimbot
This is a real-time character detection and auto-aim assist system primarily designed for Roblox games, though theoretically compatible with other games as well. The system can automatically detect and lock onto player characters in-game. It utilizes the YOLOv5 model for object detection, accelerated with DirectML, and supports AMD/NVIDIA/Intel graphics cards 😋. Custom models are supported 😋 (see Releases for model files).
## How to Use?
Run 自瞄测试.py The program will first prompt you to select a model configuration file(ONNX). After selection, detection results and status will be displayed in the top-left corner of the screen.
## Hotkeys
Toggle assist function: V  
Aim point offset: Arrow keys (may not work) ⚠️  
Sensitivity adjustment: - / + (adjust to suit different games)  
Toggle GUI display: .  
Turn on aiming:Long press the left or right mouse button
## Sample Configuration File (YAML format, similar to X-AnyLabeling)
type: yolov5  
name: yolov5s-onnx-r20250726  
display_name: YOLOv5s ONNX  
model_path: C:/Users/Administrator/Desktop/exp/weights/best.onnx  
input_width: 640  
input_height: 640  
stride: 32  
nms_threshold: 0.45  
confidence_threshold: 0.45  
classes:  
-body  
version: '1.0'  
preprocess:  
  scale: 0.00392156862745098  
  swap_rgb: true  
## For additional questions, please contact QQ: 🐧 3814991389
