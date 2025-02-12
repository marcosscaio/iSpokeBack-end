from fastapi import FastAPI, UploadFile, File, Form # type: ignore
import cv2
import numpy as np
import torch
from ultralytics import YOLO # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=False,      
    allow_methods=["*"],          
    allow_headers=["*"],         
)
print("🔄 Carregando modelo YOLO...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/marcos/runs/detect/train17/weights/best.pt"
model = YOLO(model_path).to(device)
print("✅ Modelo carregado com sucesso!")

@app.post("/detect")
async def detect_image(file: UploadFile = File(...), letra: str = Form(...)):
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Imagem inválida."}
    
    results = model.predict(image, device=device, verbose=False)
    
    detection_found = False
    detected_labels = []
    
    if results and len(results) > 0:
        result = results[0] 
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names.get(class_id, "Desconhecido")
                detected_labels.append(class_name)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if class_name.lower() == letra.lower():
                    detection_found = True

    output_path = "yolo_identification.png"
    cv2.imwrite(output_path, image)
    
    response_json = {
        "message": "Imagem processada com sucesso.",
        "detection_found": detection_found,
        "detected_labels": detected_labels,
        "output_image": output_path
    }
    
    print(response_json)
    
    return response_json

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="127.0.0.1", port=8000)
