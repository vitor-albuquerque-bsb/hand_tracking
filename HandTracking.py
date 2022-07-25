import cv2
import mediapipe as mp
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

brushThickness = 25
eraserThickness = 100
drawColor = (255, 0, 255)

tipIds = [4, 8, 12, 16, 20]
xp, yp = 0, 0

folderPath = "Header"
mylist = os.listdir(folderPath)
overlayList = []

for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

with mp_hands.Hands(max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.85,
                    min_tracking_confidence=0.85) as hands:
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img.flags.writeable = True
        img_height, img_width, _ = img.shape

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                handNo = 0
                draw = True
                xList = []
                yList = []
                bbox = []
                lmList = []
                if results.multi_hand_landmarks:
                    myHand = results.multi_hand_landmarks[handNo]
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        xList.append(cx)
                        yList.append(cy)
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)

                if len(lmList) != 0:
                    #print(lmList)
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]

                    fingers = []
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    if fingers[1] and fingers[2]:
                        print("Selection Mode")
                        if y1 < 137:
                            if 250 < x1 < 450:
                                header = overlayList[0]
                                drawColor = (255, 0, 255)
                            elif 500 < x1 < 650:
                                header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif 700 < x1 < 900:
                                header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif 1000 < x1 < 1200:
                                header = overlayList[3]
                                drawColor = (0, 0, 0)
                        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

                    if fingers[1] and fingers[2] == False:
                        cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                        print("Drawing Mode")
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        if drawColor == (0, 0, 0):
                            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        else:
                            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                        xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:137, 0:1280] = header
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)
        cv2.imshow("Inv", imgInv)
        cv2.waitKey(1)