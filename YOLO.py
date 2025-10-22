#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 11n
# all classes
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml")
#    classes=[0, 1]  # COCO: 0=person, 1=bicycle
#)


# In[ ]:


# 5n
from ultralytics import YOLO
model = YOLO("yolo5n.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv5/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 8n
from ultralytics import YOLO
model = YOLO("yolo8n.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv8/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 11n
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 11s
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 11m
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 11l
from ultralytics import YOLO
model = YOLO("yolo11l.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)


# In[ ]:


# 11x
from ultralytics import YOLO
model = YOLO("yolo11x.pt")
results = model.val(
    data="C:/Users/sabai/RA.bike/YOLOv11/datasets/coco.yaml",
    classes=[0, 1]  # COCO: 0=person, 1=bicycle
)

