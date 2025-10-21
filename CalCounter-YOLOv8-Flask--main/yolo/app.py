import os
import uuid
from flask import Flask, request, render_template, flash
from ultralytics import YOLO
from calories_db import CALORIE_DB, YOLO_TO_FOOD

#Flask
app = Flask(__name__)
app.secret_key = "food_calorie_app_secret_key_2025"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "static"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 МБ

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

print("🔄 Загрузка модели YOLOv8n...")
model = YOLO("yolov8n.pt")
print("✅ Модель готова!")

def analyze_image(image_path):
    results = model(image_path)
    result = results[0]

    total_calories = 0
    detected_items = []

    try:
        cls_array = result.boxes.cls.cpu().numpy()
        conf_array = result.boxes.conf.cpu().numpy()

        for i in range(len(cls_array)):
            class_id = int(cls_array[i])
            confidence = float(conf_array[i])

            if confidence < 0.5:
                continue

            if class_id in YOLO_TO_FOOD:
                food_name = YOLO_TO_FOOD[class_id]
                if food_name in CALORIE_DB:
                    calories = CALORIE_DB[food_name]
                    total_calories += calories
                    detected_items.append({
                        "name": food_name,
                        "calories": calories,
                        "confidence": round(confidence, 2)
                    })

        #Сохраняем изображение в папку static
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], "output.jpg")
        plotted_img = result.plot()  # BGR numpy array

        #BGR → RGB и сохраняем через PIL
        import cv2
        from PIL import Image
        if plotted_img.shape[2] == 3:
            plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(plotted_img)
        img_pil.save(output_path)

    except Exception as e:
        print(f"Подробная ошибка в analyze_image: {e}")
        return {
            "items": [],
            "total_calories": 0,
            "output_image": "output.jpg"
        }

    return {
        "items": detected_items,
        "total_calories": round(total_calories),
        "output_image": "output.jpg"
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Файл не выбран")
            return render_template("index.html")

        file = request.files["file"]
        if file.filename == "":
            flash("Файл не выбран")
            return render_template("index.html")

        if file and file.filename.lower().endswith(("png", "jpg", "jpeg")):
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                result = analyze_image(filepath)
                print("✅ Передаю в шаблон:", result)
                return render_template("index.html", result=result)
            except Exception as e:
                print("❌ Ошибка в analyze_image:", e)
                flash("Не удалось обработать изображение")
                return render_template("index.html")
        else:
            flash("Поддерживаются только JPG и PNG")
            return render_template("index.html")

    return render_template("index.html")

# Запуск сервера
if __name__ == "__main__":
    print("\n🚀 Сервер запущен! Откройте в браузере: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)