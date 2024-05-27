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

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

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

#Plt ile ekran göstermek
"""
image = cv2.imread("goku.jpeg")
kirp = image[0:250,130:550]
plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(kirp)
plt.show()
plt.show
"""

#toplam = cv2.addWeighted(img1,0.3,img2,0.7,0)
#RENK COVER EDİP İMPLEMENT ETMEK

img1 = cv2.imread("bleach1920.jpeg")
imglast = img1
img2 = cv2.imread("cv2.png")

x,y,z = img2.shape
roi = img1[0:x,0:y] #BU ANA GÖRSELİN LOGO KADAR KIRPILMIŞ HALİ

img_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#ret eşik değeri döndürüyor, mask ise yeni yapılan arrayi döndürüyor 0 ve 255den oluşan.
ret , mask = cv2.threshold(img_gray,10,255,cv2.THRESH_BINARY)

img_invert = cv2.bitwise_not(mask) #şuan 0 ve 255ler yer değişti arka plan 255 oldu

img_bg = cv2.bitwise_and(roi,roi,mask=img_invert) # çarpma yapıldı logo 0 oldugu için siyah olarak kaldı

img_fg = cv2.bitwise_and(img2,img2,mask=mask) # bu işlem ile 0,0,3 olan veri alınmamış olunuyor img_bg ye

toplam = cv2.add(img_bg,img_fg)

imglast[0:x,0:y] = toplam

cv2.namedWindow("Bleach x OpenCV",cv2.WINDOW_NORMAL)
cv2.imshow("First Picture",img1)
cv2.imshow("First Logo",img2)
cv2.imshow("Main Picture",imglast)
cv2.imshow("New Logo",toplam)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Renkli Nesne Tespiti
"""
cam = cv2.VideoCapture(0)

def nothing():
    pass

cv2.namedWindow("Color")
cv2.createTrackbar("H1","Color",0,359,nothing)
cv2.createTrackbar("H2","Color",0,359,nothing)
cv2.createTrackbar("S1","Color",0,255,nothing)
cv2.createTrackbar("S2","Color",0,255,nothing)
cv2.createTrackbar("V1","Color",0,255,nothing)
cv2.createTrackbar("V2","Color",0,255,nothing)



while cam.isOpened():

    _,frame = cam.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    h1 = int(cv2.getTrackbarPos("H1","Color"))
    h2 = int(cv2.getTrackbarPos("H2","Color"))
    s1 = cv2.getTrackbarPos("S1","Color")
    s2 = cv2.getTrackbarPos("S2","Color")
    v1 = cv2.getTrackbarPos("V1","Color")
    v2 = cv2.getTrackbarPos("V2","Color")

    lower = np.array([h1,s1,v1])
    upper = np.array([h2,s2,v2])

    mask = cv2.inRange(hsv,lower,upper)

    res = cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow("res",res)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
"""

#Yeniden Boyutlandırma vs
img = cv2.imread("bleach1920.jpeg")
res = cv2.resize(img,None,fx=1.2,fy=1.5,interpolation=cv2.INTER_CUBIC)

row,colums = img.shape[:2] #ilk 2 parametresini alıyoruz burada

#add padding
translation_mat = np.float32([[1,0,25], [0,1,25]])
img_trans = cv2.warpAffine(img,translation_mat,(colums+50,row+50))

#rotation matrix
rotation = cv2.getRotationMatrix2D((colums/2,row/2),-30,1)
img_rota = cv2.warpAffine(img,rotation,(int(colums*1.2),int(row*1.2)))

#AfineTransform noktalar seçip bunlar arasında eğilim vermek -1çünkü 0dan başlıyor
src_af = np.float32([
    [0,0],
    [colums-1,0],
    [0,row-1]])
dest_af = np.float32([
    [int(0.1*colums-1),0],
    [int(0.8*(colums-1)),0],
    [int(0.2*(colums-1)),row-1]]
)
affine= cv2.getAffineTransform(src_af,dest_af)
img_aff = cv2.warpAffine(img,affine,(colums,row))



#perspectivetransform 4 nokta seçerek
src_af = np.float32([
    [0,0],
    [colums-1,0],
    [0,row-1],
    [colums-1,row-1]])

dest_af = np.float32([
    [0,0],
    [colums-1,0],
    [int(0.2*(colums-1)),row-1],
    [int(0.8*(colums-1)),row-1]])

projective_affline = cv2.getPerspectiveTransform(src_af,dest_af)

img_new = cv2.warpPerspective(img,projective_affline,(colums,row))




newSource = []
source_list = []
counter = 0
mySource = np.float32([
    [0,0],
    [800,0],
    [0,800],
    [800,800]])
myImage = cv2.imread("uyg.jpg")
myRow,myCol = myImage.shape[:2]
def mouseCall(event,x,y,flags,param):
    global source_list,counter,newSource
    if(event == cv2.EVENT_LBUTTONDOWN):
        if counter < 4:
            source_list.append([x, y])
            counter += 1

        else:
            newSource = np.float32([[source_list[0][0],source_list[0][1]],
                         [source_list[1][0],source_list[1][1]],
                         [source_list[2][0],source_list[2][1]],
                         [source_list[3][0],source_list[3][1]]])

            project = cv2.getPerspectiveTransform(newSource,mySource)
            newImage = cv2.warpPerspective(myImage,project,(800,800))
            cv2.imshow("trans", newImage)
            counter = 0
            source_list = []

while(1):
    cv2.imshow("img", myImage)

    cv2.setMouseCallback("img",mouseCall)
    if cv2.waitKey(1) == ord("q"):
        break


cv2.destroyAllWindows()



