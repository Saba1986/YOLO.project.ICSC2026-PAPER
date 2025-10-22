#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 11n
# val on "cyclist-detection-dataset"
from ultralytics import YOLO
# Load a model
model = YOLO("yolo11n.pt")
results = model.val(data="C:/Users/sabai/RA.bike/YOLOv11/datasets/cyclist-detection-dataset.yaml")

