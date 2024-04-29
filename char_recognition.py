import cv2
import mediapipe as mp
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


def func():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    detector = HandDetector(maxHands=2)
    classifier = Classifier("Models/Alphabets/keras_model.h5", "Models/Alphabets/labels.txt")

    imgSize = 550
    offset = 30
    predicted = False

    cap = cv2.VideoCapture("Static/sign.mp4")

    while cap.isOpened():
        success, img = cap.read()

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoints.append((x, y))
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

                    a = min_x - offset
                    b = min_y - offset
                    c = max_x + offset
                    d = max_y + offset
                    w = c - a
                    h = d - b

            cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[b:b + h, a:a + w]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 4)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 4)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                
            if not predicted: 
                    prediction, index = classifier.getPrediction(imgWhite)
                    predicted = True
                    if index==0:
                        classified="A"
                    elif index==1:
                        classified="B"
                    elif index==2:
                        classified="C"
                    elif index==3:
                        classified="D"
                    elif index==4:
                        classified="E"
                    elif index==5:
                        classified="F"
                    elif index==6:
                        classified="G"
                    elif index==7:
                        classified="H"
                    elif index==8:
                        classified="I"
                    elif index==9:
                        classified="J"
                    elif index==10:
                        classified="K"
                    elif index==11:
                        classified="L"
                    elif index==12:
                        classified="M"
                    elif index==13:
                        classified="N"
                    elif index==14:
                        classified="O"
                    elif index==15:
                        classified="P"
                    elif index==16:
                        classified="Q"
                    elif index==17:
                        classified="R"
                    elif index==18:
                        classified="S"
                    elif index==19:
                        classified="T"
                    elif index==20:
                        classified="U"
                    elif index==21:
                        classified="V"
                    elif index==22:
                        classified="W"
                    elif index==23:
                        classified="X"
                    elif index==24:
                        classified="Y"
                    else :
                        classified="Z"
                    print (classified)

                    
                    return index
                # cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

            accuracy = prediction[index]
            acc = accuracy * 100

            cv2.putText(img, f"Accuracy: {acc:.2f}", (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # cv2.imshow("Hand Tracking", img)
        key = cv2.waitKey(1)
        # print("This is model index value",index)
        if key == ord("q"):
            break

        cap.release()
        cv2.destroyAllWindows()
    
   

if __name__ == "__main__":
    # Call the func() function and print the index value
    index = func()
    print(index)
    
    
    
    # import cv2
# import mediapipe as mp
# import numpy as np
# import math
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# def sign():
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils

#     detector = HandDetector(maxHands=2)
#     classifier = Classifier("Models/Alphabets/keras_model.h5", "Models/Alphabets/labels.txt")

#     imgSize = 550
#     offset = 30
#     predicted = False
#     classified=""

#     cap = cv2.VideoCapture("Static/sign.mp4")

#     while cap.isOpened():
#         success, img = cap.read()

#         image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)

#         if results.multi_hand_landmarks:
#             min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

#             for hand_landmarks in results.multi_hand_landmarks:
#                 keypoints = []

#                 for landmark in hand_landmarks.landmark:
#                     x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
#                     keypoints.append((x, y))
#                     min_x = min(min_x, x)
#                     min_y = min(min_y, y)
#                     max_x = max(max_x, x)
#                     max_y = max(max_y, y)

#                     a = min_x - offset
#                     b = min_y - offset
#                     c = max_x + offset
#                     d = max_y + offset
#                     w = c - a
#                     h = d - b

#             cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[b:b + h, a:a + w]
#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 4)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#                 # if not predicted: 
#                 #     prediction, index = classifier.getPrediction(imgWhite)
#                 #     predicted = True
#                 #     print(index)

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 4)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
            
#             if not predicted:
#                     prediction, index = classifier.getPrediction(imgWhite)
#                     predicted = True
#                     if index==0:
#                         classified="A"
#                     elif index==1:
#                         classified="B"
#                     elif index==2:
#                         classified="C"
#                     elif index==3:
#                         classified="D"
#                     elif index==4:
#                         classified="E"
#                     elif index==5:
#                         classified="F"
#                     elif index==6:
#                         classified="G"
#                     elif index==7:
#                         classified="H"
#                     elif index==8:
#                         classified="I"
#                     elif index==9:
#                         classified="J"
#                     elif index==10:
#                         classified="K"
#                     elif index==11:
#                         classified="L"
#                     elif index==12:
#                         classified="M"
#                     elif index==13:
#                         classified="N"
#                     elif index==14:
#                         classified="O"
#                     elif index==15:
#                         classified="P"
#                     elif index==16:
#                         classified="Q"
#                     elif index==17:
#                         classified="R"
#                     elif index==18:
#                         classified="S"
#                     elif index==19:
#                         classified="T"
#                     elif index==20:
#                         classified="U"
#                     elif index==21:
#                         classified="V"
#                     elif index==22:
#                         classified="W"
#                     elif index==23:
#                         classified="X"
#                     elif index==24:
#                         classified="Y"
#                     else :
#                         classified="Z"
#                     print (classified)

#             # cv2.imshow("ImageCrop", imgCrop)
#             # cv2.imshow("ImageWhite", imgWhite)

#             accuracy = prediction[index]
#             acc = accuracy * 100

#             cv2.putText(img, f"Accuracy: {acc:.2f}", (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#         # cv2.imshow("Hand Tracking", img)
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
    
# sign()


# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier


# def func():
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     mp_drawing = mp.solutions.drawing_utils

#     detector = HandDetector(maxHands=2)
#     classifier = Classifier("Models/Numbers/keras_model.h5", "Models/Numbers/labels.txt")

#     imgSize = 550
#     offset = 30
#     predicted = False

#     cap = cv2.VideoCapture("Static/sign.mp4")

#     while cap.isOpened():
#         success, img = cap.read()

#         image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)

#         if results.multi_hand_landmarks:
#             min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

#             for hand_landmarks in results.multi_hand_landmarks:
#                 keypoints = []

#                 for landmark in hand_landmarks.landmark:
#                     x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
#                     mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                     keypoints.append((x, y))
#                     min_x = min(min_x, x)
#                     min_y = min(min_y, y)
#                     max_x = max(max_x, x)
#                     max_y = max(max_y, y)

#                     a = min_x - offset
#                     b = min_y - offset
#                     c = max_x + offset
#                     d = max_y + offset
#                     w = c - a
#                     h = d - b

#             cv2.rectangle(img, (a, b), (c, d), (0, 255, 0), 2)

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[b:b + h, a:a + w]
#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 4)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
                

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 4)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
                
#             if not predicted: 
#                     prediction, index = classifier.getPrediction(imgWhite)
#                     predicted = True
#                     if index==0:
#                         classified="A"
#                     elif index==1:
#                         classified="B"
#                     elif index==2:
#                         classified="C"
#                     elif index==3:
#                         classified="D"
#                     elif index==4:
#                         classified="E"
#                     elif index==5:
#                         classified="F"
#                     elif index==6:
#                         classified="G"
#                     elif index==7:
#                         classified="H"
#                     elif index==8:
#                         classified="I"
#                     elif index==9:
#                         classified="J"
#                     elif index==10:
#                         classified="K"
#                     elif index==11:
#                         classified="L"
#                     elif index==12:
#                         classified="M"
#                     elif index==13:
#                         classified="N"
#                     elif index==14:
#                         classified="O"
#                     elif index==15:
#                         classified="P"
#                     elif index==16:
#                         classified="Q"
#                     elif index==17:
#                         classified="R"
#                     elif index==18:
#                         classified="S"
#                     elif index==19:
#                         classified="T"
#                     elif index==20:
#                         classified="U"
#                     elif index==21:
#                         classified="V"
#                     elif index==22:
#                         classified="W"
#                     elif index==23:
#                         classified="X"
#                     elif index==24:
#                         classified="Y"
#                     else :
#                         classified="Z"
#                     print (classified)
#                     return classified
#                 # cv2.imshow("ImageCrop", imgCrop)
#             cv2.imshow("ImageWhite", imgWhite)

#             accuracy = prediction[index]
#             acc = accuracy * 100

#             cv2.putText(img, f"Accuracy: {acc:.2f}", (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             # cv2.imshow("Hand Tracking", img)
#         key = cv2.waitKey(1)
#         # print("This is model index value",index)
#         if key == ord("q"):
#             break

#         cap.release()
#         cv2.destroyAllWindows()
    
   

# if __name__ == "__main__":
#     # Call the func() function and print the index value
#     classified = func()
#     print(classified)


