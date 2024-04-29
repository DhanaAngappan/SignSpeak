import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

def recognize_sign_language(image_path):
    
    image = cv2.imread(image_path)

    detector = HandDetector(maxHands=2)
    classifier = Classifier("Models/Numbers/keras_model.h5", "Models/Numbers/labels.txt")

    hands, _ = detector.findHands(image, flipType=False)

    if not hands:
        print("No hands detected in the image.")
        return None
    hand = hands[0]  
    bbox = hand["bbox"]

    x, y, w, h = bbox
    hand_roi = image[y:y+h, x:x+w]

    img_size = 300
    hand_roi_resized = cv2.resize(hand_roi, (img_size, img_size))

    
    prediction, _ = classifier.getPrediction(hand_roi_resized)
    predicted_class = prediction[0]
    predicted_class = math.ceil(predicted_class)
    if predicted_class == 0:
        print(0)
    elif predicted_class == 1:
        print(1)
    elif predicted_class == 2:
        print(2)
    elif predicted_class == 3:
        print(3)
    elif predicted_class == 4:
        print(4)
    else:
        print(5)


    return predicted_class

    

# Example usage:
image_path = "images/sample1.jpg"
predicted_class = recognize_sign_language(image_path)
if predicted_class is not None:
    print("Predicted class:", predicted_class)
else:
    print("Failed to recognize sign language gesture in the provided image.")

# import cv2
# import numpy as np
# import math
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# def recognize_sign_language(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Initialize HandDetector and Classifier
#     detector = HandDetector(maxHands=2)
#     classifier = Classifier("Models/Numbers/keras_model.h5", "Models/Numbers/labels.txt")

#     # Process the image to detect hand gestures
#     hands, _ = detector.findHands(image, flipType=False)

#     # If no hands are detected, return None
#     if not hands:
#         print("No hands detected in the image.")
#         return None

#     # Get the bounding box of the hand
#     hand = hands[0]  # Assuming there's only one hand in the image
#     bbox = hand["bbox"]

#     # Crop the region of interest (ROI) containing the hand gesture
#     x, y, w, h = bbox
#     hand_roi = image[y:y+h, x:x+w]

#     # Resize the hand ROI to the required input size for the classifier
#     img_size = 550
#     hand_roi_resized = cv2.resize(hand_roi, (img_size, img_size))

#     # Predict the class of the hand gesture using the classifier
#     prediction, _ = classifier.getPrediction(hand_roi_resized)
#     predicted_class = prediction[1]  # Assuming the classifier returns a list of predictions
    
#     # Check if the predicted class is a number (1 to 10)
#     if predicted_class.isdigit():
#         return int(predicted_class)
#     else:
#         return None  # Return None for non-numeric signs

# # Example usage:
# image_path = "images/sample1.jpg"
# predicted_class = recognize_sign_language(image_path)
# if predicted_class is not None:
#     print("Predicted class:", predicted_class)
#     cv2.imshow("Image", cv2.imread(image_path))  # Display the image
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to recognize a numeric sign language gesture in the provided image.")
