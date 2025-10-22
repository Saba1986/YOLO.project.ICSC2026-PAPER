Cyclist Detection with YOLO (ICSC 2026):

"Cyclist Detection in Urban Traffic Using Fine-Tuned YOLO Models and Diverse Real-World Video Data"

🔗 The larger codebase IS here:
https://github.com/manikantakotthapalli/Cyclist-detection-using-YOLO

What’s here:

Training and validation scripts for YOLOv5/8/11 variants

A dataset YAML configured for cyclist detection using COCO-style class names

Examples to evaluate only the person and bicycle classes (the two most relevant COCO classes for cyclist scenes)

.
├── YOLO.py                         # Quick validation across YOLO variants (classes=[0,1])
├── cyclist-detection.train.py      # Fine-tune on cyclist dataset
├── cyclist-detection.val.py        # Validate on cyclist dataset
└── cyclist-detection-dataset.yaml  # Dataset config (paths, class names)
