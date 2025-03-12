import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import mlflow
import mlflow.keras
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

# โหลดโมเดลจากไฟล์ best_model.h5
def load_model():
    if os.path.exists("best_model.h5"):
        model = tf.keras.models.load_model("best_model.h5")
        return model
    else:
        raise Exception("Model file not found.")

# โหลดโมเดลที่ดีที่สุดจาก Backend
model = load_model()

# รายชื่อคลาสที่โมเดลสามารถจำแนกได้
# class_names = ["Bacterial dermatosis", "Fungal infections", "Healthy", "Hypersensitivity allergic dermatosis"]

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
