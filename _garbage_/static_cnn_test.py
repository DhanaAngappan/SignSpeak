import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

detector = HandDetector(maxHands=2)
classifier = Classifier("Models/09/keras_model.h5", "Models/09/labels.txt")

img_path = "dataset/0/0_1.jpg"
img = cv2.imread(img_path)

prediction, accuracy = classifier.getPrediction(img)
print("Prediction:", prediction)
print("Accuracy:", accuracy)

cv2.imshow("Image", img)
cv2.waitKey(1)
cv2.destroyAllWindows()
