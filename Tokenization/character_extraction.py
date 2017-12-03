# https://github.com/bsdnoobz/zhang-suen-thinning
# https://ac.els-cdn.com/S1877050913001464/1-s2.0-S1877050913001464-main.pdf?_tid=23580d56-d5c6-11e7-b3de-00000aab0f27&acdnat=1512043449_af48c0c94bee8350a664995a634d9317
# https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d


import cv2
import numpy as np
import os
import shutil
import imutils
from matplotlib import pyplot as plt

#searched for a better skeletonizer and found this library
from skimage.morphology import skeletonize

dir = os.path.dirname(__file__)


def rotateImage(image, angle):
    #inverted = cv2.bitwise_not(image)
    rotated = imutils.rotate_bound(inverted, angle)
    #result = cv2.bitwise_not(rotated)
    return rotated

def skeletonize_alternative(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    return skel

fileindex = 4;
outputpath = os.path.join(dir, '../data/output/')
filepath = os.path.join(dir, '../data/texts/')
for index in range(0, 1):# len(os.listdir(filepath))):
    wordpath = os.path.join(outputpath, 'text' + str(fileindex).zfill(3) + '/words/')
    characterpath = os.path.join(outputpath, 'text' + str(fileindex).zfill(3) + '/characters/')
    print("#########################")
    print(wordpath)
    for word in sorted(os.listdir(wordpath)):
        print(word)
        img = cv2.imread(wordpath + word, 0)
        inverted = cv2.bitwise_not(img)

        for angle in (-12, -6, 0, 6, 12):
            rotatedimg = rotateImage(inverted, angle)

            height, width = rotatedimg.shape

            blur = cv2.GaussianBlur(rotatedimg,(1,1),0)
            ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # this skeleton is a bit less good, but saves more of the original image sometimes ...
            #skel = skeletonize_alternative(thresh)


            skel2 = skeletonize(thresh/255)
            result = list()

            resultnpy = np.copy(skel)
            resultnpy.dtype = np.uint8

            index1 = 0
            for line in skel2:
                index2 = 0
                for element in line:
                    if element == False:
                        resultnpy[index1][index2] = 0
                    else:
                        resultnpy[index1][index2] = 255
                    index2 += 1
                index1 += 1

            for line in skel:
                print(line)

            for line in resultnpy:
                print(line)

            skel = resultnpy


            #row_means = cv2.reduce(skel, 1, cv2.REDUCE_AVG)

            #highest_average_row = 0
            #highes_row_index = 0;
            #for index, row_mean in row_means:
            #    if highest_average_row < row_mean:
            #        highest_average_row = row_mean
            #        highes_row_index = index
            #total = sum(row_means)

            #currentcount = highest_average_row;
            #currentlines = list()
            #currentlines.append(index)
            #busy = True;

            #while(currentcount < total // 2 && busy):
            #    busy = False;
            #    lowestline = min(currentlines)
            #    highestline = max(currentlines)
            #    underlowestvalue = row_means[lowestline - 1]
            #    abovehighestvalue = row_means[highestline + 1]

            #    if(underlowestvalue > row_means[lowestline] // 2):
            #        currentlines.append(lowestline - 1)
            #        currentcount += underlowestvalue
            #        busy = True


            #    if(abovehighestvalue > row_means[highestline] // 2):
            #        currentlines.append(highes_row_index + 1)
            #        currentcount += abovehighestvalue
            #        busy = True

            #lowestline = min(currentlines)
            #highestline = max(currentlines)


            # col_means = cv2.reduce(thresh, 0, cv2.REDUCE_AVG);
            col_means = cv2.reduce(skel, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S) // 255;

            col_means_list = col_means[0].tolist()


            # remove trailing zeros so that no unnecessary splits are performed
            while col_means_list[-1] == 0:
                col_means_list.pop()

            x_start_index = 0;
            while col_means_list[0] == 0:
                col_means_list.pop(0)
                x_start_index += 1


            skel = skel[0:height, x_start_index:x_start_index+len(col_means_list)]

            xvals = list(range (0, len(col_means_list)))


            # Find potential split collumns
            potential_cuts = list()

            for indx in range(0, len(col_means_list)):
                col = col_means_list[indx]
                if (col == 0):
                    potential_cuts.append((indx, 0))
                if(col == 1):
                    potential_cuts.append((indx, 1))



            plt.figure(1)
            #plt.plot(xvals, list(col_means_list))
            plt.axis([0, len(col_means_list)-1, 0, height-1])

            for (index, number) in potential_cuts:
                if (number == 0):
                    plt.plot([index, index], [0, height], color='r', linestyle='-', linewidth=1)
                if (number == 1):
                    plt.plot([index, index], [0, height], color='g', linestyle='-', linewidth=1)

            plt.imshow(np.flipud(skel), origin='lower')

            #plt.figure(2)
            #plt.imshow(np.flipud(img), origin='lower')
            #plt.plot(row_sum_mean)
            plt.show()



    os.listdir(wordpath)[1]


    fileindex += 1

    # TODO::
    # 1. Juiste slant zoeken
    # 2. knip punten bepalen (zoeken op col sum average, )
    # 3. knippunten kleur van achtergrond geven
    # 4. zoeken naar nieuwe contouren
    # 5. nieuwe contouren opslaan als karakter en normaliseren.
    # 6. Manier vinden om opeenvolgende terug aaneen te kunnen plakken =>
