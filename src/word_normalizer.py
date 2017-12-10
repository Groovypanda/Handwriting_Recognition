import cv2
import numpy as np

NORMALIZED_HEIGHT = 100 #pixels
PADDING = 5
def normalize_word(word_image):

    height, width = word_image.shape[:2]
    

    difference_percentage = NORMALIZED_HEIGHT / height
    normalized_width = width * difference_percentage

    normalized_word = cv2.resize(word_image, (int(normalized_width), NORMALIZED_HEIGHT))

    norm_height, norm_width = normalized_word.shape[:2]

    container = cv2.bitwise_not(np.zeros((norm_height + (2*PADDING), norm_width + (2*PADDING)), np.uint8))
    container[PADDING:norm_height+PADDING ,PADDING:norm_width+PADDING] = normalized_word

    return container
