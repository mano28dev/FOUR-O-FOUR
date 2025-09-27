# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
import os
import cv2 as cv
import numpy as np
from werkzeug.utils import secure_filename
import google.generativeai as genai
import re
import json

genai.configure(api_key='AIzaSyCPSAzuan8sQYZG3PPbNjODky1YMLqsDT8')
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# TensorFlow (cpu build)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image as kimage

# Optional: quiet TensorFlow INFO logs (reduce console noise)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# (Optional) tune threading to limit CPU oversubscription (tweak numbers)
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load MobileNetV2 once at startup
print("Loading MobileNetV2 (ImageNet weights)...")
model = MobileNetV2(weights='imagenet')
# warm-up
_ = model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
print("Model ready.")


def is_plant_with_mobilenet(img_path, top=5, prob_thresh=0.08):
    """
    Run MobileNetV2 on the image and check if top predictions contain plant-related labels.
    Returns (is_plant: bool, preds: list of {label, prob})
    """
    img = kimage.load_img(img_path, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=top)[0]  # list of tuples

    # Plant-like keywords (ImageNet label vocabulary is inconsistent, use contains)
    plant_keywords = [
        'plant', 'potted_plant', 'tree', 'leaf', 'flower', 'orchid', 'sunflower', 'daisy',
        'rose', 'tulip', 'corn', 'maize', 'cabbage', 'broccoli', 'cauliflower',
        'banana', 'pineapple', 'strawberry', 'granny_smith', 'lemon', 'orange',
        'artichoke', 'wheat', 'barley', 'palm', 'cactus', 'vegetable', 'fruit'
    ]

    is_plant = False
    preds_list = []
    for _, label, prob in decoded:
        preds_list.append({'label': label, 'prob': float(prob)})
        lab = label.lower()
        if any(pk in lab for pk in plant_keywords) and float(prob) >= prob_thresh:
            is_plant = True
            # break  # we keep collecting preds for frontend debugging

    return is_plant, preds_list


def analyze_plant_health_opencv(img_path):
    """
    Simple OpenCV-based heuristic health check:
      - compute proportion of green pixels (healthy vegetation)
      - compute proportion of yellow/brown pixels (stress)
    Returns a dict with 'health' summary and numeric details.
    """
    img = cv.imread(img_path)
    print(get_disease_and_remedy(img_path=img_path))
    if img is None:
        return {'health': 'error', 'details': 'could not read image'}

    # resize to speed up processing and keep aspect ratio
    H, W = img.shape[:2]
    max_dim = 800
    if max(H, W) > max_dim:
        scale = max_dim / float(max(H, W))
        img = cv.resize(img, (int(W*scale), int(H*scale)))

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # green mask (broad)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    green_count = int(cv.countNonZero(mask_green))

    # yellow/brown mask (broad)
    lower_yellow = np.array([5, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    yellow_count = int(cv.countNonZero(mask_yellow))

    total = img.shape[0] * img.shape[1]
    green_ratio = green_count / total
    yellow_ratio = yellow_count / total

    # simple rule-based health label (tweak thresholds to your dataset)
    # if green_ratio > 0.5:
    #     health = "Healthy 🌿"
    # elif green_ratio > 0.25 and yellow_ratio < 0.15:
    #     health = "Moderate — mixed vegetation"
    if yellow_ratio > 0.15:
        health = "Stressed / unhealthy (yellow/brown detected)"
    else:
        if green_ratio > 0.5:
            health = "Healthy 🌿"
        elif green_ratio > 0.25:
            health = "Moderate — mixed vegetation"
        else:
            health = "Low vegetation / uncertain"

    return {
        'health': health,
        'details': {
            'green_ratio': round(green_ratio, 3),
            'yellow_ratio': round(yellow_ratio, 3),
            # 'image_pixels': int(total)
        }
    }

def get_disease_and_remedy(img_path):
    prompt = """
    You are an agricultural expert. 
    Look at this plant leaf image and tell me:
    Reply with only a Json,
    it should contain 3 fields:
    1.prescence: whether there is a disease or not(Boolean)
    2.disease_name: if disease present, then return disease present or return null
    3.remedy: if disease present, then return remedy if available or return null.
    """
    response = gemini_model.generate_content([prompt, genai.upload_file(img_path)])
    text = response.text

    # Extract first JSON object {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")

    json_str = match.group(0)

    # Parse JSON safely
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON: {json_str}")

    return data

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part (image)'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        is_plant, preds = is_plant_with_mobilenet(save_path, top=5, prob_thresh=0.02)
    except Exception as e:
        return jsonify({'error': 'Model inference failed', 'details': str(e)}), 500

    if not is_plant:
        return render_template(
            'results.html',
            filename=filename,
            message="Image does not appear to be a plant.",
            analysis=None,
            disease=None
        )

    # Plant health analysis
    analysis = analyze_plant_health_opencv(save_path)

    # Gemini disease + remedy detection
    try:
        disease_info = get_disease_and_remedy(save_path)
    except Exception as e:
        disease_info = {"prescence": False, "disease_name": None, "remedy": None}

    return render_template(
        'results.html',
        filename=filename,
        analysis=analysis,
        disease=disease_info,
        message=None
    )

try:
    capture = cv.VideoCapture("http://10.250.161.178:4747/video", cv.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError("Could not open video device")
except RuntimeError:
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open video device")

# Generator to yield frames for streaming
def gen_frames():
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # resize for speed
        frame = cv.resize(frame, (640, 480))

        # --- Simple plant detection: green areas ---
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv.inRange(hsv, lower_green, upper_green)

        # find contours of green blobs
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv.contourArea(cnt) < 500:  # ignore small noise
                continue

            x, y, w, h = cv.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]

            # get health status for this region
            health_info = analyze_plant_health_opencv_roi(roi)  # small helper

            # draw box
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.putText(frame, health_info, (x, y-5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # encode frame
        ret, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
def analyze_plant_health_opencv_roi(roi):
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    green_ratio = cv.countNonZero(mask_green) / (roi.shape[0]*roi.shape[1])

    lower_yellow = np.array([5, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = cv.countNonZero(mask_yellow) / (roi.shape[0]*roi.shape[1])

    if green_ratio > 0.5:
        return "Healthy 🌿"
    elif green_ratio > 0.25 and yellow_ratio < 0.15:
        return "Moderate"
    elif yellow_ratio > 0.15:
        return "Stressed ⚠️"
    else:
        return "Low veg"



@app.route('/live')
def live():
    return render_template('live.html')  # your live.html page

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        capture.release()
