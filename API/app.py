from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os
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

def predict_on_image(image_stream):
    try:
        # Reset the stream position before reading
        image_stream.seek(0)

        # Read the image data
        image_data = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        print(f"Image data length: {len(image_data)}")
        
        # Decode the image using OpenCV
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image could not be loaded or decoded.")

        print("Image successfully decoded.")

        # Perform classification prediction
        results = model.predict(image, imgsz=768, save_txt=True, save=True)
        save_dir = results[0].save_dir  # Get the save directory

        # Initialize an empty list to store predictions
        predictions = []

        # Iterate over the saved files in the folder
        for txt_file in os.listdir(os.path.join(save_dir, 'labels')):
            if txt_file.endswith('.txt'):
                file_path = os.path.join(save_dir, 'labels', txt_file)
                first_prediction = read_first_prediction_txt(file_path)
                if first_prediction:
                    predictions.append(first_prediction)

        return image, predictions  # Return the image and predictions

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            predicted_image, predictions = predict_on_image(file.stream)
            if predicted_image is None:
                return render_template('index.html', error='Prediction failed or image decoding issue')

            retval, buffer = cv2.imencode('.png', predicted_image)
            original_img_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template(
                'result.html', 
                original_img_data=original_img_base64, 
                predictions=predictions
            )

    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5000, host='0.0.0.0')
