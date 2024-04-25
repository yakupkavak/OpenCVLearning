import cv2
import numpy as np

from matplotlib import pyplot as plt

# ilk ders
"""
resim = cv2.imread("goku.jpeg",0)

cv2.namedWindow("resim penceresi",cv2.WINDOW_NORMAL)
cv2.imshow("resim penceresi",resim)


k= cv2.waitKey(0)
print(k)

cv2.imwrite("gokugri.jpg",resim)

cv2.destroyWindow("resim penceresi")
"""

# kamerayı açmak ve ayarlamak vs
"""
cam = cv2.VideoCapture(0) #kameranın hangisi oldugunu

if not cam.isOpened():
    print("kamera tanınmadı")
    exit()

print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(3,320)
cam.set(4,320)


while True:

    ret,frame = cam.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if not ret:
        print("i cant see")
        break

    cv2.imshow("kamera",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("it's over")
        break

cam.release()
cv2.destroyAllWindows()
"""

# video açmak
"""
cam = cv2.VideoCapture("bleach.mp4")

while cam.isOpened():

    ret, frame = cam.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YCrCb)

    if not ret:
        print("it is not opened")
        break

    cv2.imshow("anime",frame)

    if cv2.waitKey(1)& 0xFF == ord("q"):
        print("it's over")
        break
cam.release()
cv2.destroyAllWindows()
"""

# kayıt almak
"""
cam = cv2.VideoCapture(0)

fourrc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter("dronegri.avi",fourrc,30.0,(640,480))

while cam.isOpened():

    ret, frame = cam.read()

    if not ret:
        print("can't open")
        break

    out.write(frame)

    cv2.imshow("kamerakayit",frame)

    if cv2.waitKey(1) == ord("q"):
        print("it's over")
        break
cam.release()
out.release()
cv2.destroyAllWindows()
"""

# şekiller
"""
img = np.zeros((512,512,3)) #3 boyutlu olacağı için 3 dedik

#cv2.line(img,(0,0),(511,511),(255,123,0),5)
#cv2.line(img,(50,0),(511,511),(255,123,0),5)
#cv2.rectangle(img,(50,50),(100,100),(0,0,255),5)
#cv2.rectangle(img,(300,300),(500,500),(0,0,255),-1)
#cv2.circle(img,(244,244),60,(25,25,25),-1)

pts = np.array([[20,30],[50,70],[100,120],[160,190]],np.int32)
pts2 = pts.reshape(-1,1,2)
print(pts2)
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(img,"Opencv",(50,50),font,(0,24,25),2,cv2.LINE_AA)

cv2.imshow("siyah",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


# Mouse
"""
cizim = False
mod = False
xi,yi = -1,-1
def draw(event, x, y, flags, param):
    # print(x,y)
    global cizim, xi, yi, mod

    if event == cv2.EVENT_LBUTTONDOWN:
        xi,yi = x,y
        cizim = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if cizim == True:
            if mod == True:
                cv2.circle(img, (x, y), 10, (255, 55, 0), -1)
            else:
                cv2.rectangle(img,(xi,yi),(x,y),(0,255,0),2)
        else:
            pass
    elif event == cv2.EVENT_LBUTTONUP:
        cizim = False


img = np.zeros((2048, 2048, 3), np.uint8)  # göresl oluştu

cv2.namedWindow("paint")  # boş flag pencere

cv2.setMouseCallback("paint", draw)  # mouse bildirimi

while (1):
    cv2.imshow("paint", img)
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("m"):
        mod = not mod

cv2.destroyAllWindows()
"""

#Trackbar
"""
img = np.zeros((512,512,3),np.uint8)

cv2.namedWindow("resim")
def nothing(x):
    print(x)
    pass

cv2.createTrackbar("R","resim",0,255,nothing)
cv2.createTrackbar("G","resim",0,255,nothing)
cv2.createTrackbar("B","resim",0,255,nothing)

while(1):
    cv2.imshow("resim",img)

    if(cv2.waitKey(1) &0== 27):
        
        
"""

image = cv2.imread("goku.jpeg")
kirp = image[0:250,130:550]
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(kirp)
plt.show()


aç = False
tok = True
print("aç değil misin")
if not tok:
    print("tok değilim")

elif not aç:
    print("evet aç değilim, waffle no yani")

else:
    print("yemek")

