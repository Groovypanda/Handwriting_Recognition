import cv2

NORMALIZED_HEIGHT = 100 #pixels
def normalize_word(word_image):

    height, width = word_image.shape[:2]

    difference_percentage = NORMALIZED_HEIGHT / height
    normalized_width = width * difference_percentage

    normalized_word = cv2.resize(word_image, (int(normalized_width), NORMALIZED_HEIGHT))

    return normalized_word
