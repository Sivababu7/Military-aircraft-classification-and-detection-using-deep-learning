# Military-aircraft-classification-and-detection-using-deep-learning

This project focuses on detecting and classifying military aircraft from aerial and satellite images using deep learning techniques.

 # 📌 About the Project
We built a two-stage AI system:

YOLOv8 is used to detect and locate aircraft in the image.

Vision Transformer (ViT) is used to classify the type of aircraft detected.

The goal is to reduce manual effort in analyzing satellite images and provide fast, accurate results for surveillance or defense purposes.

# 🧠 Technologies Used
YOLOv8 – for object detection

Vision Transformer (ViT) – for image classification

PyTorch – deep learning framework

PASCAL VOC Format – for dataset annotations

# 🛠️ How It Works
Input an aerial/satellite image.

YOLOv8 finds aircraft in the image.

Detected aircraft are cropped and passed to ViT.

ViT predicts the aircraft type.

 # 📈 Results
High detection accuracy

Fast processing time

Works well even in low-quality images
