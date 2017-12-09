# http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
# https://www.bytefish.de/blog/extracting_contours_with_opencv/
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
# https://codereview.stackexchange.com/questions/31352/overlapping-rectangles

import os

import cv2
import numpy as np


dir = os.path.dirname(__file__)
outputpath = os.path.join(dir, 'data/output/')
filepath = os.path.join(dir, 'data/texts/')
datapath = os.path.join(dir, 'data/')
# checks if rectangle 2 inside rectangle 1
def rectangle_contains_rectangle(rectangle1, rectangle2):
    """
    rectangle2 is inside rectangle 1
    :return: boolean
    """

    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)
    if x2 + w2 <= x1 + w1 and x2 >= x1 and y2 >= y1 and y2 + h2 <= y1 + h1:
        return True
    else:
        return False

# checks if rectangle 1 is left of rectangle right or just next to each other
y_overlap_divisor = 2
def rectangle_follows_rectangle(rectangle1, rectangle2, distance):
    """
    rectangle2 follows rectangle1 with a y-overlap of at least height / y_overlap_divisor
    :return: boolean
    """
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

WORDSPACING = -200;
def check_same_line (rectangle1, rectangle2):
    """
    Checks if words are on the same y-line.
    :return: boolean
    """
    return rectangle_follows_rectangle(rectangle1, rectangle2, WORDSPACING) or rectangle_follows_rectangle(rectangle2, rectangle1, WORDSPACING)

def vertical_overlap_rectangle(rectangle1, rectangle2):
    """
    Checks if words overlap vertically
    :return: boolean
    """
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)

    hoverlaps = True
    voverlaps = True
    if (x1 > x2 + w2) or (x1 + w1 < x2):
        hoverlaps = False
    if (y1 + h1 < y2) or (y1 > y2 + h2):
        voverlaps = False
    return hoverlaps and voverlaps

# rectangle 2 is right of rectangle 1
def create_new_rectangle( rectangle1, contour1, rectangle2, contour2 ):
    """
    Creates new rectangle from two given rectangles and combines the contours of both given contours
    :return: Rectangle, contours
    """
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)

    x3 = min(x1, x2)
    y3 = min(y1, y2)
    w3 = abs( max(x2 + w2, x1 + w1) - min( x1, x2) )
    h3 = abs( max(y1 + h1, y2 + h2) - min(y1, y2) )
    rectangle3 = (x3, y3, w3, h3)

    contour3 = np.concatenate((contour1, contour2))
    return ( rectangle3 , contour3 )

def write_threshold_image(threshold_image, file_number):

    # Write threshold image for demonstration pirposes
    threshold_directory_path = os.path.join(dir, outputpath + 'thresholds')
    thresholdpath = os.path.join(dir, outputpath + 'thresholds/image' + str(file_number) + 'threshold.png')
    if not os.path.exists(threshold_directory_path):
        os.makedirs(threshold_directory_path)
    cv2.imwrite(thresholdpath, threshold_image)


def fix_vertical_overlap_in_line(line, rectangles_copy):
    busy = True
    while busy:
        busy = False
        removals = set()
        additions = list()
        for rect in line:
            for rect2 in line:
                if (rect != rect2 and vertical_overlap_rectangle(rect, rect2)):
                    busy = True
                    newrectangle = create_new_rectangle( rect, rectangles_contours[rect], rect2, rectangles_contours[rect2] )
                    rectangles_contours[newrectangle[0]] = newrectangle[1]
                    removals.add(rect)
                    removals.add(rect2)
                    additions.append( newrectangle[0] )
    line = [rect for rect in line if rect not in removals]
    rectangles_copy = [rect for rect in rectangles_copy if rect not in removals]
    for element in additions:
        line.append(element)

    return (line, rectangles_copy)

    # Please pass image as greyscale
    # the file index passed is for output purposes
def preprocess_image(img, file_index = 0):


    # clear previous output

    if not os.path.exists(datapath):
        os.makedirs(datapath)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    file_number = str(file_index).zfill(3)

    #PREPROCESSING

    height, width = img.shape[:2]

    # 2. Thresholding image

    blur = cv2.GaussianBlur(img,(3,3),0)
    ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Write threshold image for demonstration pirposes
    write_threshold_image(thresh, file_index)

    # Finding contours
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


    rectangles_contours = {}
    for index in range (0, len(contours)):
        contour = contours[index]
        rectangle = cv2.boundingRect(contour)
        if abs(rectangle[2] - width) > 2 or abs(rectangle[3] - height) > 2:
            rectangles_contours[rectangle] = contour


    #remove rectangles contained by other rectangles
    rectangles_to_remove = [rect2 for rect1 in rectangles_contours for rect2 in rectangles_contours if rect1 != rect2 and rectangle_contains_rectangle(rect1, rect2)];

    #print('removing : ' + str(len(rectangles_to_remove)))
    for rectangle in set(rectangles_to_remove):
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

        if len(rectangles_to_remove) == 0:
            removedsmalls = True
        else:
            # print('removing : ' + str(len(rectangles_to_remove)))
            for rectangle in rectangles_to_remove:
                rectangles_contours.pop(rectangle, None)

    # ordering the found words in lines
    lines = list()
    rectangles_copy = list(rectangles_contours.keys())


    linesleft = len(rectangles_copy) != 0
    iteration = 0;
    lastline = list();
    while (linesleft):
        highestrectangle = (0, 20000, 0, 0);
        line = list();
        #search the highest remaining rectangle
        for rectangle in rectangles_copy:
            if (rectangle[1] < highestrectangle[1]):
                highestrectangle = rectangle

        if (highestrectangle) == (0, 20000, 0, 0):
            break

        line.append(highestrectangle)

        for rect in rectangles_copy:
            if check_same_line(highestrectangle, rect) and rect != highestrectangle:
                line.append(rect)

        # check if line overlaps with last line.
        if len(lines) > 0:

            average = 0;
            average_height = 0;
            for rect in line:
                average += rect[1] + (rect[3] / 2)      # get average height
                average_height += rect[3]
            average /= len(line)
            average_height /= len(line)

            average_old = 0
            average_height_old = 0
            for rect in lines[-1]:
                average_old += rect[1] + (rect[3] / 2)
                average_height_old += rect[3]
            average_old /= len(lines[-1])
            average_height_old /= len(lines[-1])


            if abs(average - average_old) < max(average_height, average_height_old):
                lastline = lines.pop()
                for rect in line:
                    lastline.append(rect)
                line = lastline

        line, rectangles_copy = fix_vertical_overlap_in_line(line, rectangles_copy)

        sortedline = sorted(line, key=lambda tup: tup[0])

        lines.append(sortedline)

        for element in sortedline:
            if (element in rectangles_copy):
                rectangles_copy.remove(element)

        linesleft = len(rectangles_copy) != 0

    # saving the found rectangles

    extracted_words = list()

    ind = 0;
    for line in lines:
        for rectangle in line:
            contour = rectangles_contours[rectangle]
            (x, y, w, h) = rectangle
            extracted_word = img[y:y+h, x:x+w]
            extracted_words.append(extracted_word)

            # Saving the found word images to a file
            word_directory_path = os.path.join(dir, outputpath + 'text' + file_number + '/words/')
            wordpath = os.path.join(dir, outputpath + 'text' + file_number + '/words/word' + str(ind).zfill(4) + '.png')
            if not os.path.exists(word_directory_path):
                os.makedirs(word_directory_path)
            cv2.imwrite(wordpath, extracted_word)
            ind += 1


    # Saving the thresholded text image to a file
    parsed_text_directory = os.path.join(dir, outputpath + 'parsed_texts/')
    parsedtextpath = os.path.join(dir, outputpath + 'parsed_texts/text' + file_number + ".png")
    if not os.path.exists(parsed_text_directory):
        os.makedirs(parsed_text_directory)
    cv2.imwrite(parsedtextpath, img)


    return (extracted_words)