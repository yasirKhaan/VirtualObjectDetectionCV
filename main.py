import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1440)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
shape_color = (255, 0, 255)
center_x, center_y, width, height = 50, 50, 100, 100

class DragRects():
    def __init__(self, position_center, size=[100, 100]):
        self.position_center = position_center
        self.size = size

    def update_params(self, cursor):
        center_x , center_y = self.position_center
        width, height = self.size
        # If index finger tip is in shape area
        if center_x - width // 2 < cursor[0] < center_x + width // 2 and center_y - height // 2 < cursor[
            1] < center_y + height // 2:  # 1 for Y && 0 for X
            shape_color = (0, 255, 0)
            self.position_center = cursor # Change position of cursor


rect_lst= []
for shapes in range(4):
    rect_lst.append(DragRects([shapes*150+100,150]))



while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img) # FOR FINGER POINT

    if lmList:
        length, _, _ = detector.findDistance(8, 12, img, draw= False) # 8 for index finger & 12 for middle finger
        # print(length)
        if length < 40:
            cursor = lmList[8] # coordinates of tip of index-finger
            for rect in rect_lst:
                rect.update_params(cursor)

    ## DRAW SOLID SHAPES
    for rect in rect_lst:
        center_x , center_y = rect.position_center
        width, height = rect.size
        cv2.rectangle(img, (center_x-width//2, center_y-height//2), (center_x+width//2, center_y+height//2), shape_color, cv2.FILLED)
        cv2.imshow('Image ', img)
        cvzone.cornerRect(img, (center_x-width//2, center_y-height//2, width, height), 10, rt=0)

    ## DRAW TRANSPARENT SHAPES
    img_new = np.zeros_like(img, np.uint8)
    for rect in rect_lst:
        center_x , center_y = rect.position_center
        width, height = rect.size
        cv2.rectangle(img_new, (center_x-width//2, center_y-height//2), (center_x+width//2, center_y+height//2), shape_color, cv2.FILLED)
        cvzone.cornerRect(img_new, (center_x-width//2, center_y-height//2, width, height), 10, rt=0)
    out = img.copy()
    alpha = 0.5
    mask = img_new.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, img_new, 1-alpha, 0)[mask]


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Image ', out)
    cv2.waitKey(1)
