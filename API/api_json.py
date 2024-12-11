from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
import os
import shutil
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'C:\Users\Rifat\OneDrive\Documents\Thesis project\API\model\best.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_first_prediction_txt(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()  # Read the first line
        if first_line:
            confidence_score, class_name = first_line.split()  # Split into confidence and class name
            return {'class_name': class_name, 'confidence_score': float(confidence_score)}  # Return in dictionary
    return None  # Return None if the file is empty

def predict_on_image(image):
    try:
        save_dir = os.path.join('runs', 'classify')
        
        results = model.predict(image, imgsz=768, save_txt=True, save=True)  # classification prediction
        save_dir = results[0].save_dir  # Get the new save directory

        predictions = [] # Initializing an empty list

        # Iterate over the saved files in the folder
        for txt_file in os.listdir(os.path.join(save_dir, 'labels')):
            if txt_file.endswith('.txt'):
                file_path = os.path.join(save_dir, 'labels', txt_file)
                first_prediction = read_first_prediction_txt(file_path)
                if first_prediction:
                    predictions.append(first_prediction)

        return predictions  # Return the predictions

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


# POST method
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            try:
                # Read image file as numpy array
                image_data = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if image is None:
                    return jsonify({'error': 'Image decoding failed'}), 400

                # Predict using YOLO model
                predictions = predict_on_image(image)

                if predictions is None:
                    return jsonify({'error': 'Prediction failed'}), 500

                return jsonify({
                    'predictions': predictions
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405


# GET method
@app.route('/predictions', methods=['GET'])
def get_predictions():
    if request.method == 'GET':
        try:
            save_dir = os.path.join('runs', 'classify', 'labels')
            if not os.path.exists(save_dir):
                return jsonify({'message': 'No predictions available'}), 404

            # Initialize an empty list to store predictions
            predictions = []

            # Iterate over the saved files in the folder
            for txt_file in os.listdir(save_dir):
                if txt_file.endswith('.txt'):
                    file_path = os.path.join(save_dir, txt_file)
                    first_prediction = read_first_prediction_txt(file_path)
                    if first_prediction:
                        predictions.append(first_prediction)

            if predictions:
                return jsonify({'predictions': predictions}), 200
            else:
                return jsonify({'message': 'No predictions found'}), 404

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405


# PUT method to update predictions by uploading a new image
@app.route('/update', methods=['PUT'])
def update_prediction():
    if request.method == 'PUT':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            try:
                # Clear old predictions
                save_dir = os.path.join('runs', 'classify')
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)  # Delete old prediction folder

                # Ensure the directory is deleted
                if os.path.exists(save_dir):
                    return jsonify({'error': 'Failed to delete old predictions'}), 500

                # Process the new image
                image_data = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                if image is None:
                    return jsonify({'error': 'Image decoding failed'}), 400

                # Predict using YOLO model
                predictions = predict_on_image(image)

                if predictions is None:
                    return jsonify({'error': 'Prediction failed'}), 500

                return jsonify({
                    'message': 'Predictions updated successfully',
                    'predictions': predictions
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405



# DELETE method
@app.route('/delete', methods=['DELETE'])
def delete_predictions():
    if request.method == 'DELETE':
        try:
            save_dir = os.path.join('runs', 'classify')

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                return jsonify({'message': 'All predictions deleted successfully'}), 200
            else:
                return jsonify({'message': 'No predictions found to delete'}), 404

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method'}), 405


if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')
