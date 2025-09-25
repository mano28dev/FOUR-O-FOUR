# app.py
from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

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
    img = cv2.imread(img_path)
    if img is None:
        return {'health': 'error', 'details': 'could not read image'}

    # resize to speed up processing and keep aspect ratio
    H, W = img.shape[:2]
    max_dim = 800
    if max(H, W) > max_dim:
        scale = max_dim / float(max(H, W))
        img = cv2.resize(img, (int(W*scale), int(H*scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green mask (broad)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_count = int(cv2.countNonZero(mask_green))

    # yellow/brown mask (broad)
    lower_yellow = np.array([5, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_count = int(cv2.countNonZero(mask_yellow))

    total = img.shape[0] * img.shape[1]
    green_ratio = green_count / total
    yellow_ratio = yellow_count / total

    # simple rule-based health label (tweak thresholds to your dataset)
    if green_ratio > 0.5:
        health = "Healthy 🌿"
    elif green_ratio > 0.25 and yellow_ratio < 0.15:
        health = "Moderate — mixed vegetation"
    elif yellow_ratio > 0.15:
        health = "Stressed / unhealthy (yellow/brown detected)"
    else:
        health = "Low vegetation / uncertain"

    return {
        'health': health,
        'details': {
            'green_ratio': round(green_ratio, 3),
            'yellow_ratio': round(yellow_ratio, 3),
            'image_pixels': int(total)
        }
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload():
    # the form should post 'image' (name attribute)
    if 'image' not in request.files:
        return jsonify({'error': 'No file part (image)'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # 1) run classifier to check plant vs not-plant
    try:
        is_plant, preds = is_plant_with_mobilenet(save_path, top=5, prob_thresh=0.02)
    except Exception as e:
        return jsonify({'error': 'Model inference failed', 'details': str(e)}), 500

    if not is_plant:
        return jsonify({
            'filename': filename,
            'is_plant': False,
            'predictions': preds,
            'message': 'Image does not appear to be a plant (ImageNet classifier).'
        })

    # 2) analyze plant health using OpenCV heuristics
    analysis = analyze_plant_health_opencv(save_path)

    return jsonify({
        'filename': filename,
        'is_plant': True,
        'predictions': preds,
        'analysis': analysis
    })


if __name__ == '__main__':
    # debug True for development; in production use gunicorn / waitress
    app.run(host='0.0.0.0', port=5000, debug=True)
