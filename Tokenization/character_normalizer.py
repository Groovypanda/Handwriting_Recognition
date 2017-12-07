import cv2
import numpy as np


def remove_outer_contour(contours, image):

    height, width = image.shape
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        print("image:")
        print(height)
        print(width)
        if (x == 0 and y == 0 and w == width and h == height):
            print("cont:")
            print(h)
            print(w)
            print(contour)

def normalize_character(charlist):

    if len(charlist) == 0:
        return None

    character_img = charlist[0]
    index = 0;
    while (index + 1) < len(charlist):
        character_img = cv2.hconcat(character_img, charlist[index + 1])

    height, width = character_img.shape[:2]



    #ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #mask = cv2.bitwise_not(np.zeros((height, width), np.uint8))

    #cv2.drawContours(mask, contours, -1, (0),1)

    #remove_outer_contour(contours, mask)

    # Add padding
    maxval = max(height, width)
    blank_image = cv2.bitwise_not(np.zeros((maxval,maxval), np.uint8))

    x_offset = (maxval - width) // 2
    y_offset = (maxval - height) // 2


    blank_image[y_offset:y_offset + height, x_offset:x_offset+width] = character_img


    # Resize to 32 x 32
    final_character_image = cv2.resize(blank_image, (32 , 32))

    blur = cv2.GaussianBlur(final_character_image,(3,3),0)
    ret3,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)

    return (thresh)
