#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 11n - 10 epochs
from ultralytics import YOLO
# Load a model
model = YOLO("yolo11n.pt")
# Train the model
# results = model.train(data="cyclist-detection-dataset.yaml", epochs=100, imgsz=640, device='cpu')
results = model.train(data="C:/Users/sabai/RA.bike/YOLOv11/datasets/cyclist-detection-dataset.yaml", epochs=10)


# In[ ]:


results = model.train(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/cyclist-detection-dataset.yaml",
    epochs=10,       
    batch=8,        
    imgsz=416,       
    device='cpu',
    fraction=0.2  
)

