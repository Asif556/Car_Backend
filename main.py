from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

app = Flask(__name__)

# Load the YOLO model and PaddleOCR
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route('/detect', methods=['POST'])
def detect_plate():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image from uploaded file
    np_arr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model.predict(frame, conf=0.5)
    detected_plates = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            conf = box.conf[0]

            cropped_plate = frame[y1:y2, x1:x2]
            gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
            resized_plate = cv2.resize(gray_plate, (320, 96))

            ocr_results = ocr.ocr(resized_plate, cls=True)
            if ocr_results and ocr_results[0]:
                license_text = ''.join([line[1][0] for line in ocr_results[0]])
                detected_plates.append({
    "text": license_text,
    "confidence": float(conf),  # Convert confidence to float
    "bbox": [int(x1), int(y1), int(x2), int(y2)]  # Convert all coordinates to Python int
})


    return jsonify({"plates": detected_plates})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
