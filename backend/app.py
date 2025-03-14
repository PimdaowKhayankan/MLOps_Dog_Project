import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import mlflow
import mlflow.keras
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

# โหลดโมเดลจาก mlflow
mlflow.set_tracking_uri("http://34.87.66.93:5000")
model_uri = "models:/SkinDisease/1"
model = mlflow.tensorflow.load_model(model_uri)

# รายชื่อคลาสที่โมเดลสามารถจำแนกได้
class_names = ["Bacterial dermatosis", "Fungal infections", "Healthy", "Hypersensitivity allergic dermatosis"]

app = FastAPI()

def preprocess_image(img):
    img = Image.open(BytesIO(img))
    img = img.resize((224, 224))  # ปรับขนาดให้ตรงกับโมเดล
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Reshape
    return img

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img = preprocess_image(file.file.read())
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return {"Prediction": predicted_class, "Confidence": f"{confidence:.2f}%"}

@app.get("/class_names")
def get_class_names():
    return {"class_names": class_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8087)
