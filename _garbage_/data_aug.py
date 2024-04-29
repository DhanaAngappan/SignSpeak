import cv2
import numpy as np
import os
from skimage import transform
from random import randint

def rotate_image(image):
    angle = randint(-20, 20)  # Randomly select an angle between -20 and 20 degrees
    return transform.rotate(image, angle)

def flip_image(image):
    return np.fliplr(image)

def translate_image(image):
    x_translation = randint(-20, 20)  # Randomly select a horizontal translation between -20 and 20 pixels
    y_translation = randint(-20, 20)  # Randomly select a vertical translation between -20 and 20 pixels
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

def scale_image(image):
    scale_factor = 1 + (randint(-20, 20) / 100.0)  # Randomly select a scaling factor between 0.8 and 1.2
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def add_noise(image):
    noise = np.random.normal(0, 20, image.shape)  # Generate random noise with mean 0 and standard deviation 20
    noisy_image = image + noise.astype(np.uint8)
    return np.clip(noisy_image, 0, 255)

def augment_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            augmented_images = [rotate_image(image), flip_image(image), translate_image(image),
                                scale_image(image), add_noise(image)]

            for i, augmented_image in enumerate(augmented_images):
                cv2.imwrite(os.path.join(folder_path, f"{filename[:-4]}_aug_{i}.jpg"), augmented_image)

# Example usage:
folder_path = "Aug_Data/9"
augment_images(folder_path)
