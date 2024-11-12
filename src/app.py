# app.py
import os
from flask import Flask, render_template, request, jsonify
from utils import predict  # Import the predict function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/uploadFile", methods=["GET", "POST"])
def uploadFile():
    return render_template("uploadFile.html")

@app.route("/realTime", methods=["GET", "POST"])
def realTime():
    return render_template("realTime.html")

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to a folder
    user_audio_folder = "userAudio"
    user_spectrogram_folder = "userSpectrogram"
    os.makedirs(user_audio_folder, exist_ok=True)
    os.makedirs(user_spectrogram_folder, exist_ok=True)
    
    filepath = os.path.join(user_audio_folder, file.filename)
    output_path = os.path.join(user_spectrogram_folder, f"{os.path.splitext(file.filename)[0]}.png")
    file.save(filepath)

    try:
        predicted_class = predict(filepath, output_path)
        return jsonify({"prediction": predicted_class, "text": "File uploaded successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8001)
