import cv2
import numpy as np

def preprocess_image(image, target_size=(100, 100)):
    
    preprocessed_image = cv2.resize(image, target_size)

    preprocessed_image = preprocessed_image.astype(np.float32) / 255.0

    return preprocessed_image
