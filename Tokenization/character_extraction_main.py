# https://github.com/bsdnoobz/zhang-suen-thinning
# https://ac.els-cdn.com/S1877050913001464/1-s2.0-S1877050913001464-main.pdf?_tid=23580d56-d5c6-11e7-b3de-00000aab0f27&acdnat=1512043449_af48c0c94bee8350a664995a634d9317
# https://gist.github.com/jsheedy/3913ab49d344fac4d02bcc887ba4277d


import cv2
import numpy as np
import os
import imutils
import Tokenization.oversegmentation_correction as toc

# Found a good skeltonize-algorithm in the scikit-library
# Scikit-image dependency! (https://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)
from skimage.morphology import skeletonize
import itertools
dir = os.path.dirname(__file__)

MAX_SKEL_COL_SUMMATION = 4
MIN_DISTANCE_EDGE = 8

def rotate_image(image, angle):
    """
    Returns the image rotated by the angle (angle).
    """
    inverted = cv2.bitwise_not(image)
    rotated = imutils.rotate_bound(inverted, angle)
    reinverted = cv2.bitwise_not(rotated)
    return reinverted


#chosen_angles = [-7, -3, 0, 3, 7]
chosen_angles = [0]
def extract_character_separations(word_image, postprocess=True, sessionargs=None):
    """
    Calculate the separation points of the characters;
    :return: A Tuple with the list of chosen separation points, and a thresholded image in the correct rotation of the chosen optimal separation angle.
    """

    rotated_splits = list()

    #for angle in chosen_angles:
    #   rotated_image = rotate_image(word_image, angle)
    blur = cv2.GaussianBlur(word_image,(1,1),0)
    ret3,rotated_threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    splitpoints = find_splits_img(word_image)
    if postprocess and not sessionargs is None: # Use a neural network to check if predicted splitting points are actual splitting points.
        toc.filter_splitpoints(word_image, splitpoints)
    rotated_splits.append(splitpoints)

    '''
    We have the rotated image and the splits for each of the chosen angles
    Now we identify the image with the most splits,
    this will give us oversegmentation probably,
    but will yield the smallest percentage of undersegmentation.
    We start with taking the segmentation of the image without rotation as to ensure no unnecessary rotations are picked
    '''
    if len(rotated_splits) == 0 or len(rotated_splits[0]) == 0:
        return []
    most_splits = len( rotated_splits[ len(rotated_splits) // 2 ][0] )
    chosen_split_layout = rotated_splits[ len(rotated_splits) // 2 ]
    for rotated_split in rotated_splits:
        if len(rotated_split[0]) > most_splits:
            chosen_split_layout = rotated_split

    return chosen_split_layout

def extract_characters(word_image, sessionargs=None):
    """
    Extracts the chracters with segmentation on the word image
    :return: The labels of images, numpy pixel arrpens the dataset and preprocesses the images.ays with the image data, amount of images
    """

    chosen_split_layout = extract_character_separations(word_image, sessionargs=sessionargs)

    blur = cv2.GaussianBlur(word_image,(1,1),0)
    ret3,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # chosen split layout variables
    finalsplits = chosen_split_layout
    height, width = threshold.shape

    # Splitting the characters on the chosen split locations (finalsplits)
    splitcharacters = list()
    last_x_val = 0
    for (xval, val) in finalsplits:
        character = threshold[0:height, last_x_val:xval]
        splitcharacters.append(character)
        last_x_val = xval

    # Splitting the last character untill the end of the image
    character = threshold[0:height, last_x_val:width-1]
    splitcharacters.append(character)

    return splitcharacters

def skeletonize_thresholded_image(treshold_img):
    #skeletonize the image. Division is to normalize white to 1 and black to 0.
    height, width = treshold_img.shape
    skel2 = skeletonize(treshold_img/255)

    resultnpy = np.copy(treshold_img)  # had a weird bug where newly made np matrix did not work
    resultnpy.dtype = np.uint8

    # This loop will copy and normalize the values of the skeletization in the resultnpy numpy array
    index1 = 0
    for col in skel2:
        index2 = 0
        for pixel in col:
            # Because the scikit-images are black-white they are passed with true-false instead of greyscale values.
            if pixel == False:
                # zero value indicates a black pixel
                resultnpy[index1][index2] = 0
            else:
                # 255 indicates a white pixel
                resultnpy[index1][index2] = 255
            index2 += 1
        index1 += 1
    skel = resultnpy
    return skel


def find_splits_img(image):
    """
    :return: A tuple of the rotated image,
             with a list of the x-coordinates on which we should split the image.
    """

    # The image is inverted to facilitate working with the colors later on, as now blacks will be 0.
    inverted = cv2.bitwise_not(image)
    height, width = inverted.shape

    # Thresholding the image with the OTSU-algorithm
    blur = cv2.GaussianBlur(inverted,(1,1),0)
    ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    skel = skeletonize_thresholded_image(thresh)

    # We reduce each collumn to its summation, and normalize it to 1 for each black pixel in the original threshold image
    col_summation = cv2.reduce(skel, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S) // 255;
    col_summation_list = col_summation[0].tolist()

    """
    Here we will try to replace the given splices with the final cuts we will be making.
    Consequent splices will be turned into a single one, and the splices on points with zero pixels in the skeletonized image will take preference above those with one pixel.
    """

    # We make a list of possible points to split on.
    # We make a difference between splits on a col with 0 pixels or on a col with 1 pixel
    potential_cuts = list()
    for indx in range(0, len(col_summation_list)):
        col_sum = col_summation_list[indx]
        if (col_sum == 0):
            potential_cuts.append((indx, 0))
        if(col_sum == 1):
            potential_cuts.append((indx, 1))

    # Rmove the cuts at the end and the start of the word, because they only split off whitespace
    ending = len(col_summation_list)-1
    while len(potential_cuts) > 1 and potential_cuts[-1][0] == ending:
        potential_cuts.pop()
        ending -= 1;

    startpoint = 0;
    while len(potential_cuts) > 1 and potential_cuts[0][0] == startpoint:
        potential_cuts.pop(0)
        startpoint += 1

    sorted_potential_cuts = sorted(potential_cuts, key=lambda tup: tup[0])


        # Add consequent splits (with a len > 1) as a list (currentsplit) to a general list (split_ranges)
    currentsplit = list()
    split_ranges = list()
    index = 0
    for (col, pix) in sorted_potential_cuts:
        if index + 1 < len(sorted_potential_cuts) and sorted_potential_cuts[index + 1][0] == col + 1:
            currentsplit.append((col, pix))
        else:
            currentsplit.append((col, pix))
            #undo single line splits
            #if len(currentsplit) > 1:
            split_ranges.append(currentsplit)
            currentsplit = list()
        index += 1

    to_remove = list()
    for split_range in split_ranges:
        for (col, nr) in split_range:
            if col <= MIN_DISTANCE_EDGE or col >= len(col_summation_list) - 1 - MIN_DISTANCE_EDGE:
                to_remove.append(split_range)

    for split_range in to_remove:
        if split_range in split_ranges:
            split_ranges.remove(split_range)


    # Combining the split_ranges where in between the split ranges are no characters
    combined_ranges = list()
    current_ranges = list()
    for index in range(len(split_ranges)):
        if (index + 1 == len(split_ranges)):
            current_ranges.append(split_ranges[index])
            combined_ranges.append(current_ranges)
        else:
            cur = split_ranges[index]
            highest_splitpoint_current = cur[len(cur)-1][0]
            nxt = split_ranges[index + 1]
            lowest_splitpoint_next = nxt[0][0]

            failed = False
            for col_index in range(highest_splitpoint_current + 1, lowest_splitpoint_next):
                if col_summation_list[col_index] > MAX_SKEL_COL_SUMMATION:
                    failed = True
            if failed:
                current_ranges.append(cur)
                combined_ranges.append(current_ranges)
                current_ranges = list()
            else:
                current_ranges.append(cur)

    new_ranges = [[(split, nr) for split_range in combined_range for (split, nr) in split_range] for combined_range in combined_ranges]


    # We make a list with the final splits we will use
    final_splits = list()
    for splits in new_ranges:
        zero_splits = [split for split in splits if split[1] == 0 ]
        if(len(zero_splits) > 0):
            final_splits.append(zero_splits[ len(zero_splits) // 2 ])
        else:
            one_splits = [split for split in splits if split[1] != 0]
            final_splits.append(one_splits[ len(one_splits) // 2 ])

    #final_realigned_splits = list()
    #for split in finalsplits:
    #   newsplit = (split[0] + x_end_removed, split[1])
    #    final_realigned_splits.append(newsplit)

    return (final_splits)
