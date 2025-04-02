from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import time
import os
from pathlib import Path
import cv2
import numpy as np

app = Flask(__name__)

# إعداد المجلدات
Path("static").mkdir(exist_ok=True)

# تحميل نموذج الأنمي
model = torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='face_paint_512_v2')
model.eval()

# تحميل نموذج اكتشاف الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image: np.ndarray) -> bool:
    """اكتشاف الوجوه في الصورة"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    return len(faces) > 0  # True إذا تم اكتشاف وجه

def convert_to_anime(image: Image.Image) -> Image.Image:
    """تحويل الصورة إلى أنمي"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_img = transforms.ToPILImage()(output_tensor.squeeze().clamp(-1, 1) * 0.5 + 0.5)
    return output_img.resize(image.size, Image.LANCZOS)

@app.route("/", methods=["GET"])
def home_page():
    """الصفحة الرئيسية"""
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files['file']
        if file:
            # إنشاء أسماء ملفات فريدة
            timestamp = str(int(time.time()))
            input_filename = f"input_{timestamp}.jpg"
            output_filename = f"output_{timestamp}.jpg"
            input_path = os.path.join("static", input_filename)
            output_path = os.path.join("static", output_filename)

            # حفظ الصورة المرفوعة
            file.save(input_path)
            
            # فتح الصورة والتحقق من وجود وجه
            original_img = Image.open(input_path).convert("RGB")
            img_array = np.array(original_img)
            
            # إذا تم اكتشاف وجه، نقوم بالتحويل
            if detect_face(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)):
                anime_img = convert_to_anime(original_img)
                anime_img.save(output_path, quality=95)
                
                return render_template("result.html", 
                                       input_image=f"/static/{input_filename}",
                                       output_image=f"/static/{output_filename}",
                                       message="تم اكتشاف وجه وتحويل الصورة إلى أنمي")
            else:
                return render_template("result.html", 
                                       input_image=f"/static/{input_filename}",
                                       message="لم يتم اكتشاف أي وجه في الصورة")
        else:
            return render_template("error.html", error="لم يتم رفع صورة")

    except Exception as e:
        return render_template("error.html", error=f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
