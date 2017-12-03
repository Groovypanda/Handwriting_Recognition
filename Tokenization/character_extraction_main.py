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
# Scikit-image dependency! (https://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)
from skimage.morphology import skeletonize
dir = os.path.dirname(__file__)


def rotateImage(image, angle):
    #inverted = cv2.bitwise_not(image)
    rotated = imutils.rotate_bound(image, angle)
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

#fileindex = 4;
#outputpath = os.path.join(dir, '../data/output/')
#filepath = os.path.join(dir, '../data/texts/')
#for index in range(0, 1):# len(os.listdir(filepath))):

    #wordpath = os.path.join(outputpath, 'text' + str(fileindex).zfill(3) + '/words/')
    #characterpath = os.path.join(outputpath, 'text' + str(fileindex).zfill(3) + '/characters/')
    #print("#########################")
    #print(wordpath)
    #for word in sorted(os.listdir(wordpath)):

def extract_characters(word, index=0):
    #img = cv2.imread(wordpath + word, 0)
    #img = cv2.imread(word)

    img = word

    inverted = cv2.bitwise_not(img)

    angeled_splits = list()

    for angle in (-7, -3, 0, 3, 7):
        rotatedimg = rotateImage(inverted, angle)

        height, width = rotatedimg.shape

        blur = cv2.GaussianBlur(rotatedimg,(1,1),0)
        ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # this skeleton is a bit less good, but saves more of the original image sometimes ...
        #skel = skeletonize_alternative(thresh)

        #inv = cv2.bitwise_not(thresh)
        #im2, contours, hierarchy = cv2.findContours(inv,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        #res = cv2.drawContours(inv, contours, -1, (123,123,123), 1)
        #print(contours)
        #print(hierarchy)
        #cv2.imshow ("contours", inv)
        #cv2.waitKey(0)

        skel2 = skeletonize(thresh/255)
        result = list()

        resultnpy = np.copy(thresh)#skel)
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

        #for line in skel:
        #    print(line)

        for line in resultnpy:
            print(line)

        skel = resultnpy


        col_means = cv2.reduce(skel, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S) // 255;

        col_means_list = col_means[0].tolist()


        # remove trailing zeros so that no unnecessary splits are performed
        croppedLeft = 0;
        while col_means_list[-1] == 0:
            col_means_list.pop()
            croppedLeft += 1;

        x_start_index = 0;
        while col_means_list[0] == 0:
            col_means_list.pop(0)
            x_start_index += 1


        skel = skel[0:height, x_start_index:x_start_index+len(col_means_list)]

        xvals = list(range (0, len(col_means_list)))


        # Find potential split collumnsk;
        potential_cuts = list()

        for indx in range(0, len(col_means_list)):
            col = col_means_list[indx]
            if (col == 0):
                potential_cuts.append((indx, 0))
            if(col == 1):
                potential_cuts.append((indx, 1))

        # combining the potential split collumns

        ending = len(col_means_list)-1
        while potential_cuts[-1][0] == ending:
            potential_cuts.pop()
            ending -= 1;

        startpoint = 0;
        while potential_cuts[0][0] == startpoint:
            potential_cuts.pop(0)
            startpoint += 1


        sorted_potential_cuts = sorted(potential_cuts, key=lambda tup: tup[0])

        splitranges = list()

        searchindex = 1
        startindex = 0;

        currentsplit = list()

        index = 0
        for (col, pix) in sorted_potential_cuts:
            if index + 1 < len(sorted_potential_cuts) and sorted_potential_cuts[index + 1][0] == col + 1:
                currentsplit.append((col, pix))
            else:
                currentsplit.append((col, pix))

                #undo single line splits
                if len(currentsplit) > 1:
                    splitranges.append(currentsplit)
                currentsplit = list()
            index += 1


        finalsplits = list()
        for splits in splitranges:

            zero_splits = list()
            one_splits = list()
            for split in splits:
                if split[1] == 1:
                    zero_splits.append(split)
                else:
                    one_splits.append(split)

            if(len(zero_splits) > 0):
                finalsplits.append(zero_splits[ len(zero_splits) // 2 ])
            else:
                finalsplits.append(one_splits[ len(one_splits) // 2 ])


        rotatedimgcropped = rotatedimg[0:height, x_start_index:x_start_index+len(col_means_list)]

        final_realigned_splits = list()
        for split in finalsplits:
            newsplit = (split[0] + croppedLeft, split[1])
            final_realigned_splits.append(newsplit)

        angeled_splits.append((final_realigned_splits, rotatedimg))


    # We have the rotated image and the splits for each of the chosen angles

    most_splits = len( angeled_splits[ len(angeled_splits) // 2 ][0] )
    chosen_split = angeled_splits[ len(angeled_splits) // 2 ]
    for angled_split in angeled_splits:
        if len(angled_split[0]) > most_splits:
            chosen_split = angled_split

    if (chosen_split == None):
        chosen_split = angeled_splits[ len(angeled_splits) // 2 ]

    im = chosen_split[1]
    reinverted_img = cv2.bitwise_not(im)
    finalsplits = chosen_split[0]

    splitcharacters = list()

    height, width = reinverted_img.shape

    #cv2.imshow("img", reinverted_img)
    cv2.waitKey(0)

    last_x_val = 0
    for (xval, val) in finalsplits:
        character = reinverted_imgextracted_word = img[0:height, last_x_val:xval]
        splitcharacters.append(character)
        last_x_val = xval

    character = reinverted_imgextracted_word = img[0:height, last_x_val:width-1]
    splitcharacters.append(character)

    return splitcharacters




    # PLOTTING IMAGE AND SPLITTING LINES


    #plt.figure(1)
    #plt.plot(xvals, list(col_means_list))
    #plt.axis([0, width-1, 0, height-1])

    #for (index, number) in finalsplits:
    #    if (number == 0):
    #        plt.plot([index, index], [0, height], color='r', linestyle='-', linewidth=1)
    #    if (number == 1):
    #        plt.plot([index, index], [0, height], color='g', linestyle='-', linewidth=1)

    #plt.imshow(np.flipud(reinverted_img), origin='lower')

    #plt.figure(2)
    #plt.imshow(np.flipud(img), origin='lower')
    #plt.plot(row_sum_mean)
    #plt.show()



    #os.listdir(wordpath)[1]


    #fileindex += 1

    # TODO::
    # 1. Juiste slant zoeken
    # 2. knip punten bepalen (zoeken op col sum average, )
    # 3. knippunten kleur van achtergrond geven
    # 4. zoeken naar nieuwe contouren
    # 5. nieuwe contouren opslaan als karakter en normaliseren.
    # 6. Manier vinden om opeenvolgende terug aaneen te kunnen plakken =>
