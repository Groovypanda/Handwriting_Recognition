import cv2
import numpy as np

# This is necessary because test data is also centered in this rectangle
CHARACTER_DIMS = 44
CONTAINER_DIMS = 64
side_padding = (CONTAINER_DIMS - CHARACTER_DIMS) // 2

def delete_white_border(character_image):
    # Invert image to make whites zero
    inverted = cv2.bitwise_not(character_image)
    height, width = inverted.shape[:2]

    col_summation = cv2.reduce(inverted, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S);
    row_summation = cv2.reduce(inverted, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S);

    # Getting border sizes
    left_bound, right_bound, up_bound, down_bound = 0, len(col_summation[0])-1, 0, len(row_summation)-1

    # Left
    while ( left_bound < len(col_summation[0])-1 and col_summation[0][left_bound] == 0 ):
        left_bound += 1
    # RIGHT
    while ( right_bound > 0  and col_summation[0][right_bound] == 0 ):
        right_bound -= 1
    # Up
    while ( up_bound < len(row_summation)-1  and row_summation[up_bound] == 0):
        up_bound += 1
    # Down
    while (down_bound > 0  and row_summation[down_bound] == 0):
            down_bound -= 1

    if (left_bound > right_bound or up_bound > down_bound):
        return character_image

    borderless_image = character_image[up_bound:down_bound, left_bound:right_bound]
    return(borderless_image)

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


    # Treshold the image
    blur = cv2.GaussianBlur(character_img,(3,3),0)
    ret3,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)

    # Delete white border around images
    borderless_image = delete_white_border(thresh)

    # Add padding
    height, width = borderless_image.shape[:2]
    maxval = max(height, width)
    blank_image = cv2.bitwise_not(np.zeros((maxval,maxval), np.uint8))

    # center our image in the middle of the padding image, this way either the height or width will be padded
    x_offset = (maxval - width) // 2
    y_offset = (maxval - height) // 2
    blank_image[y_offset:y_offset + height, x_offset:x_offset+width] = borderless_image

    """
    The provided padding of 10 pixels on all sides is to offset padding in training data
    """

    # Resize to correct dimensions
    final_character_image = cv2.resize(blank_image, (CHARACTER_DIMS , CHARACTER_DIMS))

    container = cv2.bitwise_not(np.zeros((CONTAINER_DIMS,CONTAINER_DIMS), np.uint8))
    container[side_padding:side_padding+CHARACTER_DIMS, side_padding:side_padding+CHARACTER_DIMS] = final_character_image


    return (container)
