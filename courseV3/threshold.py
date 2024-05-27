def thresshold(imgSource,thresh,maxval):
    """
    This function takes an image as input and thresholds the image
    :src imgSource:
    :param thresh:
    :param maxval:
    :return:
    """

    img = imgSource.copy()
    rows,cols, = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if(img[i,j] < thresh):
                img[i,j] = 0
            else:
                img[i,j] = maxval

    return thresh,img
