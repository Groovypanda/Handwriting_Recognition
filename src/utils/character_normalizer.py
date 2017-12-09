import cv2
import numpy as np

CHARACTER_DIMS = 64

def normalize_character(charlist):
    """
    Precondition:: All character slices in the input have the same height! This is correct when the character slices come from the same word.
    :return: a normalized image of the chosen character splits all pasted together
    """


    if len(charlist) == 0:
        return None

    # Horizontal concatenation of the character splices
    character_img = charlist[0]
    index = 0;
    while (index + 1) < len(charlist):
        character_img = cv2.hconcat(character_img, charlist[index + 1])

    # Add padding
    height, width = character_img.shape[:2]
    maxval = max(height, width)
    blank_image = cv2.bitwise_not(np.zeros((maxval,maxval), np.uint8))

    # center our image in the middle of the padding image, this way either the height or width will be padded
    x_offset = (maxval - width) // 2
    y_offset = (maxval - height) // 2
    blank_image[y_offset:y_offset + height, x_offset:x_offset+width] = character_img


    # Resize to correct dimensions
    final_character_image = cv2.resize(blank_image, (CHARACTER_DIMS , CHARACTER_DIMS))

    blur = cv2.GaussianBlur(final_character_image,(3,3),0)
    ret3,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)

    return (thresh)
