# http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/

import cv2
import numpy as np
import os
import shutil
dir = os.path.dirname(__file__)


fileindex = 1;
outputpath = os.path.join(dir, '../data/output/')
filepath = os.path.join(dir, '../data/texts/')

# clear previous output
shutil.rmtree(outputpath)
os.makedirs(outputpath)

# checks if rectangle 2 inside rectangle 1
def rectangle_contains_rectangle(rectangle1, rectangle2):
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)
    if x2 + w2 <= x1 + w1 and x2 >= x1 and y2 >= y1 and y2 + h2 <= y1 + h1:
        return True
    else:
        return False

y_overlap_divisor = 2
# checks if rectangle 1 is left of rectangle right or just next to each other
def rectangle_follows_rectangle(rectangle1, rectangle2, distance):
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)
    # x1 left of x2
    # right side of rect1 less that 1 pixer away from left side rect2 (or inside rect 2)
    if x1 <= x2 and (x1 + w1 - x2 >= distance):
        if y1 > y2 and y1 < y2 + h2:
            overlap = abs(y2 + h2 - y1)
            if (overlap >= h1 / y_overlap_divisor) or (overlap >= h2 / y_overlap_divisor):
                return True;
        if y1 <= y2 and y2 < y1 + h1:
            overlap = abs(y1 + h1 - y2)
            if (overlap >= h1 / y_overlap_divisor) or (overlap >= h2 / y_overlap_divisor):
                return True;
    return False

def check_same_line (rectangle1, rectangle2, distance):
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)


# rectangle 2 is right of rectangle 1
def create_new_rectangle( rectangle1, contour1, rectangle2, contour2 ):
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)

    x3 = min(x1, x2)
    y3 = min(y1, y2)
    w3 = abs( max(x2 + w2, x1 + w1) - min( x1, x2) )
    h3 = abs( max(y1 + h1, y2 + h2) - min(y1, y2) )
    rectangle3 = (x3, y3, w3, h3)

    #TODO::
    contour3 = np.concatenate((contour1, contour2))
    return ( rectangle3 , contour3 )

for file in os.listdir(filepath):
    print(file)

    #PREPROCESSING

    # 1. reading image in greyscale

    img = cv2.imread(filepath + file, 0)

    height, width = img.shape[:2]

    # 2. Thresholding image (maybe?)

    blur = cv2.GaussianBlur(img,(5,5),0)

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3,4)

    #cv2.imwrite(path + "words/text.png", img)

    #cv2.imwrite(path + "words/text1.png", blur)

    threshold_directory_path = os.path.join(dir, outputpath + 'thresholds')
    thresholdpath = os.path.join(dir, outputpath + 'thresholds/image' + str(fileindex) + 'threshold.png')
    if not os.path.exists(threshold_directory_path):
        os.makedirs(threshold_directory_path)

    cv2.imwrite(thresholdpath, thresh)

    # Finding contours

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


    #skews = []
    rectangles_contours = {}
    for index in range (0, len(contours)):
        contour = contours[index]
        rectangle = cv2.boundingRect(contour)

        if abs(rectangle[2] - width) > 2 or abs(rectangle[3] - height) > 2:
            rectangles_contours[rectangle] = contour
        #skew = cv2.minAreaRect(contour)
        #skewbox = cv2.boxPoints(skew)
        #skews.append(np.int0(skewbox))


    #cv2.drawContours(img, skews, -1, (0,255,0), 1)

    rectangles_to_remove = [];
    to_add = {}

    # Sifting out rectangles inside others
    for rectangle1 in rectangles_contours:
        for rectangle2 in rectangles_contours:
            if rectangle1 != rectangle2:
                if rectangle_contains_rectangle(rectangle1, rectangle2):
                    rectangles_to_remove.append(2)


    #print('removing : ' + str(len(rectangles_to_remove)))
    for rectangle in rectangles_to_remove:
        rectangles_contours.pop(rectangle, None)

    # try to add squared that are overlapping

    busy = True;
    while busy:
        interrupted = False
        busy = False
        rectangles_to_remove = [];
        to_add = {}

        for rectangle1 in rectangles_contours:
            if not interrupted:
                for rectangle2 in rectangles_contours:
                    if not interrupted:
                        if rectangle1 != rectangle2:
                            if rectangle_follows_rectangle(rectangle1, rectangle2, -1):
                                (rectangle3, contour3) = create_new_rectangle(rectangle1, rectangles_contours[rectangle1], rectangle2, rectangles_contours[rectangle2])
                                rectangles_to_remove.append(rectangle1)
                                rectangles_to_remove.append(rectangle2)
                                to_add[rectangle3] = contour3
                                #print(rectangle1, rectangle2, rectangle3)
                                interrupted = True
        if interrupted:
            busy = True

            #print('removing : ' + str(len(rectangles_to_remove)))
            for rectangle in rectangles_to_remove:
                rectangles_contours.pop(rectangle, None)

            #print('adding : ' + str(len(to_add)))
            for key in to_add:
                rectangles_contours[key] = to_add[key]

    #print('removing : ' + str(len(rectangles_to_remove)))
    for rectangle in rectangles_to_remove:
        rectangles_contours.pop(rectangle, None)

    #print('adding : ' + str(len(to_add)))
    for key in to_add:
        rectangles_contours[key] = to_add[key]

    # sift out small rectangles (points, lines, ...)
    removedsmalls = False;
    while not (removedsmalls):
        rectangles_to_remove = []

        totalheight = 0
        for rect in rectangles_contours:
            totalheight += rect[3]
        average_height = totalheight / len(rectangles_contours)
        lowest_height = average_height/3

        for rectangle in rectangles_contours:
            if rectangle[3] < lowest_height:
                rectangles_to_remove.append(rectangle)
            #if rectangle[3] > 2 * lowest_height:
                # TODO: split multi line errors.

        if len(rectangles_to_remove) == 0:
            removedsmalls = True
        else:
            # print('removing : ' + str(len(rectangles_to_remove)))
            for rectangle in rectangles_to_remove:
                rectangles_contours.pop(rectangle, None)

    # ordering the found words in lines
    lines = list()
    rectangles_copy = rectangles_contours.keys()
    sorted_rectangles_copy = sorted(rectangles_copy, key=lambda x: (x[1], x[0]))
    linesleft = len(rectangles_copy) != 0
    iteration = 0;
    while (linesleft):
        highestrectangle = (0, 20000, 0, 0);

        #search the highest remaining rectangle
        for rectangle in rectangles_copy:
            if (rectangle[1] < highestrectangle[1]):
                highestrectangle = rectangle

        # finding rest of the line
        for rectangle in rectangles_copy:
                if (rectangle[0] < highestrectangle[0] and rectangle_follows_rectangle(rectangle, highestrectangle):
                        lines[iteration].append(rectangle)
                if (rectangle[0] >= highestrectangle[0] and rectangle_follows_rectangle(highestrectangle, rectangle):
                        lines[iteration].append(rectangle)
        (x, y, w, h) = highestrectangle
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        highestrect = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(thresholdpath, highestrect)
        #linesleft = len(rectangles_copy) != 0
        linesleft = False





    # saving the found rectangles
    ind = 0;
    for rectangle in rectangles_contours:
        (x, y, w, h) = rectangle
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        extracted_word = img[ymin:ymax, xmin:xmax]

        word_directory_path = os.path.join(dir, outputpath + 'text' + str(fileindex) + '/words/')
        wordpath = os.path.join(dir, outputpath + 'text' + str(fileindex) + '/words/word' + str(ind) + '.png')
        if not os.path.exists(word_directory_path):
            os.makedirs(word_directory_path)

        cv2.imwrite(wordpath, extracted_word)
        ind += 1

        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    finalcontours = list()
    for key in rectangles_contours:
        finalcontours.append(rectangles_contours[key])

    # this draws the final contours

    # cv2.drawContours(img, finalcontours, -1, (0,255,0), 1)

    parsed_text_directory = os.path.join(dir, outputpath + 'parsed_texts/')
    parsedtextpath = os.path.join(dir, outputpath + 'parsed_texts/text' + str(fileindex) + ".png")
    if not os.path.exists(parsed_text_directory):
        os.makedirs(parsed_text_directory)

    cv2.imwrite(parsedtextpath, img)

    fileindex += 1



# TODO
# Step 1: removing boxes inside other boxes -> done
# Step 2; dividing multi line in two. -> see TODO, maybe use blob detection, or other binarization ?
# Step 3: adding squares that are next to each other    -> problem with vertical overlap ->  check if horizontal half of the one before
#                                                       -> problem with words after each other -> fix overlap with italic text

# Only save pixel points and create new image from pixel points in original image
