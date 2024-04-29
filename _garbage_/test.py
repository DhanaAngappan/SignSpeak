import cv2
import mediapipe as mp
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

detector = HandDetector(maxHands=2)
classifier = Classifier("Models/Numbers/keras_model.h5","Models/Numbers/labels.txt")


imgSize=550
offset=30

counter=0
#folder="Data/C"
#gesture_label = input("Enter the gesture label: ")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    #if not success:
        #continue
    
   
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    results = hands.process(image_rgb)

    
    if results.multi_hand_landmarks:
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            #mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            keypoints = []

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                keypoints.append((x, y))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                

                a=min_x-offset
                b=min_y-offset
                c=max_x+offset
                d=max_y+offset
                #print(a,b,c,d)
                w=c-a
                h=d-b
        cv2.rectangle(img, (a,b), (c, d), (0, 255, 0), 2)

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[b:b+h,a:a+w]

        cv2.imshow("ImageCrop",imgCrop)

        imgCropShape= imgCrop.shape
        aspectRatio=h/w

        if aspectRatio>1:
                k = imgSize/h
                wCal =math.ceil(k*w)
                imgResize= cv2.resize(imgCrop,(wCal,imgSize))
                imgResizeShape= imgResize.shape
                wGap = math.ceil((imgSize-wCal)/4)
                imgWhite[:,wGap:wCal+wGap]=imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                print(prediction)
            
        else:
                k = imgSize/w
                hCal =math.ceil(k*h)
                imgResize= cv2.resize(imgCrop,(imgSize,hCal))
                imgResizeShape= imgResize.shape
                hGap = math.ceil((imgSize-hCal)/4)
                imgWhite[hGap:hCal+hGap, :]=imgResize
                prediction, index = classifier.getPrediction(imgWhite)
                print(prediction)

        cv2.imshow("ImageCrop",imgCrop)
        
        cv2.imshow("ImageWhite",imgWhite)

        accuracy=prediction[index]
        acc= accuracy*100
        print(acc)
        print(prediction)
        cv2.putText(img, f"Accuracy: {acc:.2f}", (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            
    cv2.imshow("Hand Tracking", img)
    #cv2.imshow("WhiteFrame",imgWhite)
    
    

    #cv2.imshow("Image",img)
    key= cv2.waitKey(1)
    '''if key ==ord ("m"):
        counter +=1
        cv2.imwrite(f'{folder}/{gesture_label}_{counter}.jpg',imgWhite)
        print(counter)'''

cap.release()
cv2.destroyAllWindows()



