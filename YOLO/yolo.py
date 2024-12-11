import os
from ultralytics import YOLO


model = YOLO('yolov8n-cls.pt')

data_path = r'C:\Users\Rifat\OneDrive\Documents\Thesis project\dataset'
epochs = 200           
batch_size = 32
img_size = 768
model.train(
    data=data_path, 
    epochs=epochs, 
    imgsz=img_size, 
    batch=batch_size, 
    augment=True,            # Enable data augmentation
    patience=50,              # Early stopping patience (adjusted)
    weight_decay=0.01,       # Increased weight decay to avoid overfitting
    dropout=0.6,             # Increase dropout
)

print("Training completed.")
