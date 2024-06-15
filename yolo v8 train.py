import os
from ultralytics import YOLO

ROOT_DIR = r"C:\Users\paruc\Downloads\py files\brain tumour\Brain Tumor Data Set\Brain Tumor Data Set"

# Load or create the model first
model = YOLO("yolov8n-cls.pt")  # build a new model from scratch

# Use the model
results = model.train(data=ROOT_DIR, save_period=5, epochs=25, batch=16)