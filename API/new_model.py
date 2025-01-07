from flask import Flask, request, jsonify, make_response
import numpy as np
import base64
import cv2
import os
import subprocess
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil
import logging
import uuid
import tensorflow as tf
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO(r'C:\Users\Rifat\Downloads\FInal_thesis_project\API\model\best.pt')


# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# Function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to read predictions from the YOLO text file
def read_first_prediction_txt(file_path):
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                confidence_score, class_name = first_line.split()
                return {'class_name': class_name, 'confidence_score': float(confidence_score)}
    except Exception as e:
        print(f"Error reading prediction file: {e}")
    return None


# Function to make predictions on an image
def predict_on_image(image):
    try:
        # Generate a sequential directory name for the prediction
        base_dir = os.path.join('runs', 'classify')
        os.makedirs(base_dir, exist_ok=True)
        
        # Find the next available directory name
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        next_dir_num = len(existing_dirs) + 1
        unique_dir = os.path.join(base_dir, f'predict{next_dir_num}')

        # YOLO model prediction
        results = model.predict(image, imgsz=768, save_txt=True, save=True, project=base_dir, name=f'predict{next_dir_num}')
        save_dir = results[0].save_dir

        # Collect predictions
        predictions = []
        label_dir = os.path.join(save_dir, 'labels')
        if os.path.exists(label_dir):
            for txt_file in os.listdir(label_dir):
                if txt_file.endswith('.txt'):
                    file_path = os.path.join(label_dir, txt_file)
                    first_prediction = read_first_prediction_txt(file_path)
                    if first_prediction:
                        predictions.append(first_prediction)

        return predictions
    finally:
        # Clear TensorFlow cache to release memory
        tf.keras.backend.clear_session()


# Endpoint: Predict on an image
@app.route('/predict', methods=['POST'])
def predict():
    print("Request files:", request.files)  # Log the request files
    if 'file' not in request.files:
        print('No file in request')  # Log this
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print("File received:", file.filename)  # Log the filename

    if file.filename == '':
        print('File is empty')  # Log this
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Decode the uploaded image
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                print('Image decoding failed')  # Log this
                return jsonify({'error': 'Image decoding failed'}), 400

            # Run prediction
            predictions = predict_on_image(image)
            if predictions is None:
                print('Prediction failed')  # Log this
                return jsonify({'error': 'Prediction failed'}), 500

            return jsonify({'predictions': predictions})

        except Exception as e:
            print(f"Error: {str(e)}")  # Log this
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400



# Endpoint: Get all predicted results
@app.route('/all_predictions', methods=['GET'])
def get_all_predicted_results():
    try:
        # Directory where predictions are saved
        classify_dir = os.path.join('runs', 'classify')
        if not os.path.exists(classify_dir):
            return jsonify({'message': 'No predictions directory found'}), 404

        predictions = []  # List to store all predictions

        # Iterate through all subdirectories in the classify directory
        for subdir in os.listdir(classify_dir):
            subdir_path = os.path.join(classify_dir, subdir)
            if os.path.isdir(subdir_path):
                labels_dir = os.path.join(subdir_path, 'labels')
                if os.path.exists(labels_dir):
                    for txt_file in os.listdir(labels_dir):
                        if txt_file.endswith('.txt'):
                            file_path = os.path.join(labels_dir, txt_file)
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                                file_predictions = []
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        confidence_score = float(parts[0])
                                        class_name = parts[1]
                                        prediction = {
                                            'class_name': class_name,
                                            'confidence_score': round(confidence_score, 3),
                                            'file_name': txt_file
                                        }
                                        file_predictions.append(prediction)
                                if file_predictions:
                                    predictions.extend(file_predictions)

        if predictions:
            return jsonify({
                'message': 'Predictions retrieved successfully',
                'total_predictions': len(predictions),
                'predictions': predictions
            }), 200
        else:
            return jsonify({'message': 'No predictions found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Endpoint: Update predictions with a new image
@app.route('/update', methods=['PUT'])
def update_prediction():
    logging.debug('Received update request.')

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    logging.debug(f"File received: {file.filename}")

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Create base directory if it doesn't exist
            save_dir = os.path.join('runs', 'classify')
            os.makedirs(save_dir, exist_ok=True)
            
            # Instead of removing directory, try to clear contents
            if os.path.exists(save_dir):
                try:
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            logging.warning(f"Error clearing file {file_path}: {str(e)}")
                except Exception as e:
                    logging.warning(f"Error clearing directory {save_dir}: {str(e)}")
                    # Continue execution even if clearing fails
            
            # Decode the new image
            image_data = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({'error': 'Image decoding failed'}), 400

            # Run prediction
            predictions = predict_on_image(image)
            if predictions is None:
                return jsonify({'error': 'Prediction failed'}), 500

            if not predictions:
                return jsonify({'error': 'No objects detected in the image'}), 400

            return jsonify({
                'message': 'Predictions updated successfully',
                'predictions': predictions
            })

        except Exception as e:
            logging.error(f"Error during update: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


# Endpoint: Delete all predictions
@app.route('/delete', methods=['DELETE'])
def delete_predictions():
    try:
        save_dir = os.path.join('runs', 'classify')
        if not os.path.exists(save_dir):
            return jsonify({'message': 'No predictions found to delete'}), 404

        try:
            # First attempt bulk deletion
            shutil.rmtree(save_dir)
        except PermissionError:
            # If bulk deletion fails, try deleting files individually
            success = True
            for root, dirs, files in os.walk(save_dir, topdown=False):
                for name in files:
                    try:
                        os.unlink(os.path.join(root, name))
                    except Exception as e:
                        success = False
                        logging.warning(f"Failed to delete file {name}: {str(e)}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as e:
                        success = False
                        logging.warning(f"Failed to delete directory {name}: {str(e)}")
            
            if not success:
                return jsonify({'warning': 'Some files could not be deleted due to permissions'}), 207

        return jsonify({'message': 'All predictions deleted successfully'}), 200

    except Exception as e:
        logging.error(f"Error during deletion: {str(e)}")
        return jsonify({'error': str(e)}), 500


    # Prevent caching of API responses
@app.after_request
def add_cache_control_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Run the app
if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=True)