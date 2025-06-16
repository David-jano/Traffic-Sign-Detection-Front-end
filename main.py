from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (replace 'best.pt' with your actual model path)
model = YOLO("best.pt")

# TTS setup (optional)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Custom messages for detected signs
class_warnings = {
    "Green Light": "Be ready to go",
    "Red Light": "Stop",
    "Stop": "Be ready to stop",
    "Speed Limit 10": "Minimize your speed to 10 kilometers per hour",
    "Speed Limit 20": "Minimize your speed to 20 kilometers per hour",
    "Speed Limit 30": "Minimize your speed to 30 kilometers per hour",
    "Speed Limit 40": "Minimize your speed to 40 kilometers per hour",
    "Speed Limit 50": "Minimize your speed to 50 kilometers per hour",
    "Speed Limit 60": "Minimize your speed to 60 kilometers per hour",
    "Speed Limit 70": "Minimize your speed to 70 kilometers per hour",
    "Speed Limit 80": "Minimize your speed to 80 kilometers per hour",
    "Speed Limit 90": "Minimize your speed to 90 kilometers per hour",
    "Speed Limit 100": "Minimize your speed to 100 kilometers per hour",
    "Speed Limit 110": "Minimize your speed to 110 kilometers per hour",
    "Speed Limit 120": "Minimize your speed to 120 kilometers per hour"
}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    content = await file.read()
    npimg = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    results = model(frame)[0]

    class_names = results.names
    detections = []

    for box in results.boxes:
        class_id = int(box.cls)
        class_name = class_names[class_id]

        xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x_min, y_min, x_max, y_max]
        confidence = float(box.conf.cpu().numpy())

        detections.append({
            "class_name": class_name,
            "bbox": xyxy,
            "confidence": confidence
        })

    detected_classes = list(set([d['class_name'] for d in detections]))
    warnings = [class_warnings.get(c, c) for c in detected_classes]

    # Optional server-side speech:
    # for text in warnings:
    #     engine.say(text)
    # engine.runAndWait()

    return {"detections": detections, "warnings": warnings}
