# https://github.com/bsdnoobz/zhang-suen-thinning
# https://ac.els-cdn.com/S1877050913001464/1-s2.0-S1877050913001464-main.pdf?_tid=23580d56-d5c6-11e7-b3de-00000aab0f27&acdnat=1512043449_af48c0c94bee8350a664995a634d9317

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


fileindex = 4;
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
        inverted = cv2.bitwise_not(img)

        for angle in (-30, -15, 0, 15, 30):

            rotatedimg = rotateImage(inverted, angle)

            col_mean = cv2.reduce(rotatedimg, 0, cv2.REDUCE_AVG);
            #row_sum_mean = cv2.reduce(rotatedimg, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S);

            xvals = list()

            col_mean_list = list(col_mean[0])
            for i in range(0, len(col_mean_list)):
                xvals.append(i)

            #print( "img" )
            #print(inverted)
            #print( "mean" )
            #print(col_mean_list)

            maxval = max(col_mean_list)
            average_of_averages = sum(col_mean_list) / len(col_mean_list)

            plt.plot(xvals, list(col_mean_list))
            plt.axis([0, len(col_mean_list), 0, maxval])
            plt.plot([0, len(col_mean_list)], [average_of_averages, average_of_averages], color='k', linestyle='-', linewidth=2)
            plt.imshow(np.flipud(rotatedimg), origin='lower')
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
