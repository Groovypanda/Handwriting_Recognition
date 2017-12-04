import cv2
import numpy as np

def normalize_character(charlist):

    if len(charlist) == 0:
        return None

    character_img = charlist[0]
    index = 0;
    while (index + 1) < len(charlist):
        character_img = cv2.hconcat(character_img, charlist[index + 1])

    height, width = character_img.shape[:2]

    # Add padding
    maxval = max(height, width)
    blank_image = cv2.bitwise_not(np.zeros((maxval,maxval), np.uint8))

    x_offset = (maxval - width) // 2
    y_offset = (maxval - height) // 2


    blank_image[y_offset:y_offset + height, x_offset:x_offset+width] = character_img


    # Resize to 32 x 32
    final_character_image = cv2.resize(blank_image, (32 , 32))

    return (final_character_image)
    # We want to revert the character to 32 x 32
