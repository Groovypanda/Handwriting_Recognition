#http://opencvpython.blogspot.be/2012/05/skeletonization-using-opencv-python.html

import cv2
import numpy as np
import os
import shutil
import imutils
from matplotlib import pyplot as plt



dir = os.path.dirname(__file__)


def rotateImage(image, angle):
    #inverted = cv2.bitwise_not(image)
    rotated = imutils.rotate_bound(inverted, angle)
    #result = cv2.bitwise_not(rotated)
    return rotated


fileindex = 6;
outputpath = os.path.join(dir, '../data/output/')
filepath = os.path.join(dir, '../data/texts/')
for index in range(0, 1):# len(os.listdir(filepath))):
    wordpath = os.path.join(outputpath, 'text' + str(fileindex) + '/words/')
    characterpath = os.path.join(outputpath, 'text' + str(fileindex) + '/characters/')
    print("#########################")
    print(wordpath)
    for word in os.listdir(wordpath):
        print(word)
        img = cv2.imread(wordpath + word, 0)
        
        cv2.imshow("img", img)
        cv2.waitKey(0)

        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)

        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True


        cv2.imshow("skel", skel)
        cv2.waitKey(0)
