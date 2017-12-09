# http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
# https://www.bytefish.de/blog/extracting_contours_with_opencv/
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
# https://codereview.stackexchange.com/questions/31352/overlapping-rectangles

import os

import cv2
import numpy as np
import sys


dir = os.path.dirname(__file__)
outputpath = os.path.join(dir, 'data/output/')
datapath = os.path.join(dir, 'data/')
if not os.path.exists(datapath):
    os.makedirs(datapath)
if not os.path.exists(outputpath):
    os.makedirs(outputpath)
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
            if (overlap >= min(h1, h2) / y_overlap_divisor):
                return True;
        if y1 <= y2 and y2 < y1 + h1:
            overlap = abs(y1 + h1 - y2)
            if (overlap >= min(h1, h2) / y_overlap_divisor):
                return True;
    return False

WORDSPACING = -200;
def check_same_line (line, rectangle):
    """
    Checks if words are on the same y-line.
    :return: boolean
    """
    average_y_pos = 0;
    for rect in line:
        average_y_pos += rect[1] + (rect[3] // 2)
    average_y_pos /= len(line)

    # Returns true if average_y_pos between the bottom and the top of rectangle
    return rectangle[1] <= average_y_pos and rectangle[3] >= average_y_pos

def vertical_overlap_rectangle(rectangle1, rectangle2):
    """
    Checks if words overlap vertically
    :return: boolean
    """
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)
    if (x1 > x2 + w2) or (x1 + w1 < x2):
        return False
    if y1 > y2 and y1 < y2 + h2:
        overlap = abs(y2 + h2 - y1)
        if (overlap >= min(h1, h2) / y_overlap_divisor):
            return True;
    if y1 <= y2 and y2 < y1 + h1:
        overlap = abs(y1 + h1 - y2)
        if (overlap >= min(h1, h2) / y_overlap_divisor):
            return True;
    return False

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
    """
    Writes the treshold image for demonstration purposes
    """
    # Write threshold image for demonstration pirposes
    threshold_directory_path = os.path.join(dir, outputpath + 'thresholds')
    thresholdpath = os.path.join(dir, outputpath + 'thresholds/image' + str(file_number) + 'threshold.png')
    if not os.path.exists(threshold_directory_path):
        os.makedirs(threshold_directory_path)
    cv2.imwrite(thresholdpath, threshold_image)

def write_word_image(extracted_word, file_number, line_index, word_index):

    """
    Writes the word image to a file for demonstration purposes (can also be done from the return of the preprocess function)
    """
    word_directory_path = os.path.join(dir, outputpath + 'text' + file_number + '/words/')
    wordpath = os.path.join(dir, outputpath + 'text' + str(file_number) + '/words/word' + str(line_index).zfill(3) + "_" + str(word_index).zfill(3) + '.png')
    if not os.path.exists(word_directory_path):
        os.makedirs(word_directory_path)
    cv2.imwrite(wordpath, extracted_word)

def check_for_overlapping_in_line(line):
    """
    Checks for an overlap in the bounding rectangles of the contours and returns the two overlapping rectangles if found within a line of the text
    :return: A tuple with the two overlapping rectangles or a tuple of None values if no overlap was found
    """
    for rect1 in line:
        for rect2 in line:
            if (rect1 != rect2 and ( vertical_overlap_rectangle(rect1, rect2) or rectangle_follows_rectangle(rect1, rect2, -1) ) ):
                return (rect1, rect2)
    return (None, None)

def fix_overlapping_in_line(line, rectangles_copy, rectangles_contours):
    """
    Combines bounding rectangles that are in the same line to one single boundinrectangle. Merges the contours also.
    :return: The newly created line.
    """
    busy = True
    while busy:
        busy = False
        removals = set()
        additions = list()

        # Check if there is overlapping in the line
        # This way of combining ensures that after every step of the while loop, the length of line will decrease with one rectangle, or the method will return
        result = check_for_overlapping_in_line(line)
        if result[0] != None and result[1] != None:
            busy = True
            (rect, rect2) = result
            # Create the new rectangle from the two old ones
            newrectangle = create_new_rectangle( rect, rectangles_contours[rect], rect2, rectangles_contours[rect2] )
            rectangles_contours[newrectangle[0]] = newrectangle[1]
            # Remove the old rectangles and append the new one
            removals.add(rect)
            removals.add(rect2)
            additions.append( newrectangle[0] )

        # Execute removals and addition
        line = [rect for rect in line if rect not in removals]
        rectangles_copy = [rect for rect in rectangles_copy if rect not in removals]
        for element in additions:
            line.append(element)

    return (line, rectangles_copy)

def remove_small_rectangles(rectangles_contours):
    """
    Removing the small line elements, like commas and points floating about.
    This is not the correct way to handle this in a full fledged character recognition,
    but since we do not process anything else than letters and digits, we need to remove the small parts.
    :return: rectangles_contours, but without the small points.
    """
    removedsmalls = False;
    # The use of the while here is solely because when we remove the smallest points,
    # the average height goes up, and the opportunity to delete more small points is presented.
    while not (removedsmalls):
        rectangles_to_remove = []
        totalheight = 0
        # Calculating average word height
        for rect in rectangles_contours:
            totalheight += rect[3]
        average_height = totalheight / len(rectangles_contours)
        lowest_height = average_height/3

        # Remove smallest rectangles
        for rectangle in rectangles_contours:
            if rectangle[3] < lowest_height:
                rectangles_to_remove.append(rectangle)
        if len(rectangles_to_remove) == 0:
            removedsmalls = True
        else:
            for rectangle in rectangles_to_remove:
                rectangles_contours.pop(rectangle, None)
    return rectangles_contours

def add_words_to_line(line, words):
    """
    Adds words from words to line, if it is on the same line,
    as calculated by the check_same_line method.
    :return: The updated line

    """
    for word in words:
        if check_same_line(line, word) and not word in line:
            line.append(word)
    return line


def split_text_in_lines(rectangles_contours):
    """
    The contours and their bounding rectangles are split into lines.
    Algorithm:
        We take the highest contour and put it in the line.
        The average line y-position is calculated
        While there are rectangles that cover the average line y-position, they are added to the line
        Since we work from top to bottom, there are no rectangles above the highest rectangle, so no two lines will be merged this way,
        unless they are very skewed, in which case this algorithm will not suffice, and an angled projection will be needed to measure the angle of the lines.
    :return: A list of lists containing the words on that line (a list of lines)
    """
    rectangles_copy = list(rectangles_contours.keys())
    lines = list()
    linesleft = len(rectangles_copy) != 0
    iteration = 0;
    lastline = list();
    while (linesleft):
        highestrectangle = (0, sys.maxsize, 0, 0);
        line = list();
        #search the highest remaining rectangle
        for rectangle in rectangles_copy:
            if (rectangle[1] < highestrectangle[1]):
                highestrectangle = rectangle
        if (highestrectangle) == (0, sys.maxsize, 0, 0):
            break
        line.append(highestrectangle)

        # append the rectangles to the line
        new_line = add_words_to_line(line, rectangles_copy)
        while new_line != line:
            line = new_line
            new_line = add_words_to_line(line, rectangles_copy)
        line = new_line

        # Check if line overlaps with last line.
        # This is done by comparing the average y-positions, and checking if the distance between is smaller than the largest average line height.
        if len(lines) > 0:
            average = sum([rect[1] + (rect[3] / 2) for rect in line]) / len(line)
            average_height = sum([rect[3] for rect in line]) / len(line)
            average_old = sum([rect[1] + (rect[3] / 2) for rect in lines[-1]]) / len(lines[-1])
            average_height_old = sum([rect[3] for rect in lines[-1]]) / len(lines[-1])
            if abs(average - average_old) < max(average_height, average_height_old):
                lastline = lines.pop()
                for rect in line:
                    lastline.append(rect)
                line = lastline

        # Fix the overlapping within the line
        line, rectangles_copy = fix_overlapping_in_line(line, rectangles_copy, rectangles_contours)

        # Sort the line, remove the contents from the rectangles_copy (this variable contains the unprocessed words)
        # and add the line to the lines list (= text)
        sortedline = sorted(line, key=lambda tup: tup[0])
        lines.append(sortedline)

        for element in sortedline:
            if (element in rectangles_copy):
                rectangles_copy.remove(element)

        linesleft = len(rectangles_copy) != 0

    return lines

    # Please pass image as greyscale
    # the file index passed is for output purposes
def preprocess_image(img, file_index = 0):

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

    # Putting contours in dictionary to process
    rectangles_contours = {}
    for index in range (0, len(contours)):
        contour = contours[index]
        rectangle = cv2.boundingRect(contour)
        if abs(rectangle[2] - width) > 2 or abs(rectangle[3] - height) > 2:
            rectangles_contours[rectangle] = contour


    #remove rectangles contained by other rectangles
    rectangles_to_remove = [rect2 for rect1 in rectangles_contours for rect2 in rectangles_contours if rect1 != rect2 and rectangle_contains_rectangle(rect1, rect2)];

    for rectangle in set(rectangles_to_remove):
        rectangles_contours.pop(rectangle, None)

    # sift out small rectangles (points, lines, ...)
    rectangles_contours = remove_small_rectangles(rectangles_contours)

    # ordering the found words in lines

    lines = split_text_in_lines(rectangles_contours)
    # saving the found rectangles

    extracted_words = list()
    for line_index, line in enumerate(lines):
        for word_index, rectangle in enumerate(line):
            contour = rectangles_contours[rectangle]
            (x, y, w, h) = rectangle
            extracted_word = img[y:y+h, x:x+w]
            extracted_words.append(extracted_word)

            # Saving the found word images to a file
            write_word_image(extracted_word, file_number, line_index, word_index)

    return (extracted_words)
