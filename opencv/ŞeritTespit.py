import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture("car8.mp4")
sapma = 100
kernel = np.ones((5,5),np.uint8)

def cropMatrix(img):
    x,y = img.shape[:2]
    value = np.array([[(sapma,x-sapma)
                          , (int((y*3)/8),int(x*0.6))
                          , (int((y*5)/8),int(x*0.6))
                          , (y, x-sapma)]],dtype= np.int32)
    return value


def crop_image(img,matrix):
    x,y = img.shape[:2]
    mask = np.zeros((x,y),np.uint8)
    mask = cv2.fillPoly(mask,matrix,255)
    mask = cv2.bitwise_and(img,img,mask=mask)
    return mask

def filt(img):
    img_copy = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_copy = cv2.inRange(img_copy,155,255)
    img_copy = cv2.erode(img_copy,kernel)
    img_copy = cv2.dilate(img_copy,kernel)
    img_copy = cv2.medianBlur(img_copy,9)
    img_copy = cv2.Canny(img_copy,40,200)
    return  img_copy




img_org = 0

while cam.isOpened():
    ret, frame = cam.read()

    if not ret:
        print("bitti")
        break

    img_org = frame.copy()
    img_org = crop_image(img_org,cropMatrix(img_org))
    img_org = filt(img_org)
    cv2.imshow("Frame", img_org)

    key = cv2.waitKey(16)
    if key == ord("q"):
        print("kapatıldı")
        break


cv2.destroyAllWindows()
cam.release()
