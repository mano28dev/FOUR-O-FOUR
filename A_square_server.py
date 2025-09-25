from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded file
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    return f"File '{file.filename}' uploaded successfully!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
