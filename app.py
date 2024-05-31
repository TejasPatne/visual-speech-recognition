from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import tensorflow as tf
from utils import load_video, load_alignments, CTCLoss, num_to_char

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('data', 'uploads')  # Define upload folder inside static folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register custom loss function
tf.keras.utils.get_custom_objects()['CTCLoss'] = CTCLoss

# Load the saved model
loaded_model = tf.keras.models.load_model('models/vispnet')
print("models loaded successfully")

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file uploaded', 400
    
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Return URL relative to the static folder
        return filename.replace("data", "")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return 'No video file uploaded', 400
    
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Save the uploaded video
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        # Load the video and preprocess it
        frames = load_video(filename)
        reshaped_frames = tf.reshape(frames, [-1, 75, 46, 140, 1])
        # Perform prediction
        predictions = loaded_model.predict(reshaped_frames)
        # Decode the predictions
        decoded_predictions = tf.keras.backend.ctc_decode(predictions, input_length=[75], greedy=True)[0][0].numpy()
        # Convert numeric predictions to text
        decoded_text = []
        for sentence in decoded_predictions:
            decoded_text.append(tf.strings.reduce_join([num_to_char(word) for word in sentence]).numpy().decode('utf-8'))
        
        # Return the predictions
        return jsonify({'predictions': decoded_text})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
