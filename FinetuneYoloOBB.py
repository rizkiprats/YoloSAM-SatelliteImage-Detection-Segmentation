import ultralytics
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    ultralytics.checks()

    # Load the pre-trained YOLO OBB model
    # model = YOLO("yolov8x-obb.pt")
    # model = YOLO("runs/obb/train/weights/best.pt")
    model = YOLO("yolo11x-obb.pt")
    
    # Train the model
    model.train(data="Datasets/augmented_dataset/dataset.yaml", epochs=300, imgsz=640, batch=2, device=0)