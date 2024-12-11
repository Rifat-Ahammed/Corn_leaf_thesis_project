from ultralytics import YOLO
import os

def read_first_prediction_txt(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            confidence_score, class_name = first_line.split()  # Split into confidence and class name
            return {'class_name': class_name, 'confidence_score': float(confidence_score)}
    return None

        
model = YOLO(r'C:\Users\Rifat\OneDrive\Documents\Thesis project\YOLO\runs\classify\train\weights\best.pt')
path = r'C:\Users\Rifat\OneDrive\Documents\Thesis project\dataset\test\Gray_Leaf_Spot'
results = model.predict(path, imgsz=768, save=True)
