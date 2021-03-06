import os

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


dir = os.path.dirname(__file__)
datapath = os.path.join(dir, '../data/')
outputpath = os.path.join(datapath, 'output/')
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
ALLOWED_VERTICAL_OVERLAP = 2
def rectangle_follows_rectangle(rectangle1, rectangle2, distance):
    """
    rectangle2 follows rectangle1 with a y-overlap of at least height / ALLOWED_VERTICAL_OVERLAP
    :return: boolean
    """
    ((x1, y1, w1, h1), (x2, y2, w2, h2)) = (rectangle1, rectangle2)
    # x1 left of x2
    # right side of rect1 less that 1 pixer away from left side rect2 (or inside rect 2)
    if x1 <= x2 and (x1 + w1 - x2 >= distance):
        if y1 > y2 and y1 < y2 + h2:
            overlap = abs(y2 + h2 - y1)
            if (overlap >= min(h1, h2) / ALLOWED_VERTICAL_OVERLAP):
                return True;
        if y1 <= y2 and y2 < y1 + h1:
            overlap = abs(y1 + h1 - y2)
            if (overlap >= min(h1, h2) / ALLOWED_VERTICAL_OVERLAP):
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
        if (overlap >= min(h1, h2) / ALLOWED_VERTICAL_OVERLAP):
            return True;
    if y1 <= y2 and y2 < y1 + h1:
        overlap = abs(y1 + h1 - y2)
        if (overlap >= min(h1, h2) / ALLOWED_VERTICAL_OVERLAP):
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
    average_width = sum([rect[2] for rect in line]) / len(line)
    average_characters = 16; # AVERAGE = 8, HALF CHARACTER WIDTH SO *2
    ALLOWED_WORD_GAP = - average_width / average_characters # allowed pixels between parts of the word for correction
    """
    Checks for an overlap in the bounding rectangles of the contours and returns the two overlapping rectangles if found within a line of the text
    :return: A tuple with the two overlapping rectangles or a tuple of None values if no overlap was found
    """
    for rect1 in line:
        for rect2 in line:
            if (rect1 != rect2 and ( vertical_overlap_rectangle(rect1, rect2) or rectangle_follows_rectangle(rect1, rect2, ALLOWED_WORD_GAP) ) ):
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

def remove_image_contour_if_exists(img, rectangles_contours):
    """
    Remove the contour of the image if it is taken by otsu's thresholding
    :return: The updated rectangles_contours
    """
    height, width = img.shape[:2]
    to_remove = list()
    for rect in rectangles_contours:
        if rect[2] >= width-PADDING_SIZE and rect[3] >= height-PADDING_SIZE: # 2 pixels from border, 2 padding
            to_remove.append(rect)
    for rect in to_remove:
        del rectangles_contours[rect]

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

def split_rectangle_contour_horizontally(rectangle, contour, y_val):
    """
    Splits rectangle horizontally on y-val
    :return: The two rectangles with their updated contours
    """
    (x, y, w, h) = rectangle
    rect1 = (x, y, w, y_val-1)
    rect2 = (x, y+y_val, w, h-y_val)
    (cont1, cont2) = (contour, contour) # TODO:: SPLITTING THIS IS A HASSLE, WONT DO IF NOT USED SOMEWHERE ELSE

    return ((rect1, cont1), (rect2, cont2))


MAX_RECTANGLE_HEIGHT_MULTIPLIER = 2
def search_multiline_contours(rectangles_contours, img):
    """
    Search the found words, for mistakenly combined words from two different lines.
    We call the function split_multiline_contours if such a mistake is found.
    :return: rectangles_contours but with the mistakes replaced by the split words.
    """
    average_height = sum([rect[3] for rect in list(rectangles_contours.keys())]) / len(rectangles_contours)
    probable_multiline_rectangles = [(rect, contour) for rect, contour in rectangles_contours.items() if (rect[3] >= MAX_RECTANGLE_HEIGHT_MULTIPLIER * average_height)]

    if len(probable_multiline_rectangles) > 0:
        parsed_probabilities = split_multiline_contours(probable_multiline_rectangles, average_height, img)
        for entry in probable_multiline_rectangles:
            del rectangles_contours[entry[0]]
        for parsed_rectangle in parsed_probabilities:
            rectangles_contours[parsed_rectangle[0]] = parsed_rectangle[1]
        return rectangles_contours
    else:
        return rectangles_contours


SPLIT_MULTILINE_MULTIPLIER = 10
MINIMAL_HEIGHT_MULTIPLIER = 2/3
def split_multiline_contours(probable_multiline_rectangles, average_height, img):
    """
    We split the words based on the minimum value in the horizontal row-projection histogram
    :return: rectangles_contours but with the mistakes replaced by the split words.
    """
    parsed_rectangles = list()
    busy = True
    while busy:
        busy = False
        for rectangle, contour in probable_multiline_rectangles:
            #create skeletonized image
            (x, y, w, h) = rectangle
            if h >=  MAX_RECTANGLE_HEIGHT_MULTIPLIER * average_height:
                extracted_image = img[y:y+h, x:x+w]
                # Inverting the image to view whites as zero and blacks as 255
                inverted_extracted_image = cv2.bitwise_not(extracted_image)
                blur = cv2.GaussianBlur(inverted_extracted_image,(3,3),0)
                ret3,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                row_summation = cv2.reduce(threshold, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S) // 255;

                # Extract the row which is the most likely split points
                center = len(row_summation) // 2
                search_size = int(MINIMAL_HEIGHT_MULTIPLIER * average_height)
                (min_row, min_val) = min( [ (index, element) for index, element in enumerate( row_summation[ int(center - search_size):int(center + search_size) ] ) ], key = lambda t: t[1] )
                min_row += int(center - search_size) # re-add the offset of the list splicing of the previous line

                # Split the rectangle on the row with the lowest black pixel count (is thresholded so we assume this is text or only minor distortion)
                (rect_cont1, rect_cont2) = split_rectangle_contour_horizontally(rectangle, contour, min_row)

                parsed_rectangles.append(rect_cont1)
                parsed_rectangles.append(rect_cont2)
                #TODO:: in case of time left: execute horizontal split on rectangles if col with 0 is found, else could combine two words where not desire
                busy = True
            else:
                parsed_rectangles.append((rectangle, contour))
        probable_multiline_rectangles = parsed_rectangles
        parsed_rectangles = list()

    return probable_multiline_rectangles


    return probable_multiline_rectangles;

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


PADDING_SIZE = 4 #pixels
def pad_img_white(img):
    height, width = img.shape[:2]
    new_h, new_w = height+(2*PADDING_SIZE), width+(2*PADDING_SIZE)

    container = cv2.bitwise_not(np.zeros((new_h, new_w), np.uint8))
    container[PADDING_SIZE:height+PADDING_SIZE ,PADDING_SIZE:width+PADDING_SIZE] = img
    return container

    # Please pass image as greyscale
    # the file index passed is for output purposes
def preprocess_image(img, file_index = 0):
    """
    Convert the given text in words, in order in which they are found in the text
    :return: A list of the word images
    """

    file_number = str(file_index).zfill(3)

    #PREPROCESSING
    img = pad_img_white(img)
    height, width = img.shape[:2]

    # 2. Thresholding image

    blur = cv2.GaussianBlur(img,(3,3),0)
    ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # Finding contours
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Putting contours in dictionary to process
    rectangles_contours = {}
    for index in range (0, len(contours)):
        contour = contours[index]
        rectangle = cv2.boundingRect(contour)
        if abs(rectangle[2] - width) > 2 or abs(rectangle[3] - height) > 2:
            rectangles_contours[rectangle] = contour

    # All rectangles print demonstration purposes
    print_img = img.copy()
    for rectangle in rectangles_contours:
        (x, y, w, h) = rectangle
        cv2.rectangle(print_img,(x,y),(x+w,y+h),(0,255,0),2)
    write_threshold_image(print_img, file_index+20)


    # Remove possible image contour because of padding
    rectangles_contours = remove_image_contour_if_exists(img, rectangles_contours)

    # Remove rectangles contained by other rectangles
    rectangles_to_remove = [rect2 for rect1 in rectangles_contours for rect2 in rectangles_contours if rect1 != rect2 and rectangle_contains_rectangle(rect1, rect2)];

    for rectangle in set(rectangles_to_remove):
        rectangles_contours.pop(rectangle, None)

    # Sift out small rectangles (points, lines, ...)
    rectangles_contours = remove_small_rectangles(rectangles_contours)

    # Split contours that mistakenly span multiple lines

    rectangles_contours = search_multiline_contours(rectangles_contours, img)


    # Ordering the found words in lines
    lines = split_text_in_lines(rectangles_contours)

    # Saving the found rectangles
    extracted_words = list()
    for line_index, line in enumerate(lines):
        for word_index, rectangle in enumerate(line):
            contour = rectangles_contours[rectangle]
            (x, y, w, h) = rectangle
            extracted_word = img[y:y+h, x:x+w]
            extracted_words.append(extracted_word)

            # Saving the found word images to a file
            write_word_image(extracted_word, file_number, line_index, word_index)

    new_img = img.copy()
    greyscale = 73
    color_increment = 180 // len(lines)
    for line_index, line in enumerate(lines):
        for word_index, rectangle in enumerate(line):
            (x, y, w, h) = rectangle
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(greyscale,255,0),5)
        greyscale += color_increment
        #cv2.drawContours(new_img, [rectangles_contours[rect] for rect in rectangles_contours], 0, (120,255,0), 3)

    # Write threshold image for demonstration pirposes
    write_threshold_image(new_img, file_index)
    write_threshold_image(thresh, file_index+10)

    return (extracted_words)
