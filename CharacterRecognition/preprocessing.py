import time as t

import cv2
import numpy as np

import settings

'''
For this project we use the Chars74K dataset. It contains 64 classes, with about 74K images.
'''

# Global constants
start = t.time()
SIZE = settings.SIZE
EMPTY_VALUE = 0.0
WRITE_VALUE = 1.0


def imageBorders(image):
    threshold = SIZE / 16  # Allow a smart part of the letter to be outside of the image after translation.
    # Note, this threshold could give errors with empty images...
    # Count written pixels per row and per column
    written_pixels = np.where(image != EMPTY_VALUE)
    written_pixels_x, written_pixels_y = written_pixels[0], written_pixels[1]
    amount_hor = np.zeros(SIZE)
    amount_ver = np.zeros(SIZE)
    for x in np.nditer(written_pixels_x):
        amount_hor[x] += 1
    for y in np.nditer(written_pixels_y):
        amount_ver[y] += 1
    possible_horizontal_borders = np.where(amount_hor >= threshold)[0]
    possible_vertical_borders = np.where(amount_ver >= threshold)[0]
    up = np.ndarray.item(possible_horizontal_borders, 0)
    down = np.ndarray.item(possible_horizontal_borders, -1)
    left = np.ndarray.item(possible_vertical_borders, 0)
    right = np.ndarray.item(possible_vertical_borders, -1)
    return up, down, left, right


def getMinAndMaxScale(perc):
    threshold = 0.30
    return (1.0, 1.5) if perc < threshold else (0.75, 1.25)


# Is the given pixel in the image after a translation of dx pixels?
def inBounds(x):
    return SIZE > x >= 0


def noisyImage(img, stddev):
    noise = np.random.normal(loc=0, scale=stddev, size=img.shape)
    aug_img = img + noise.reshape(img.shape)
    aug_img[aug_img < 0] = 0.0
    aug_img[aug_img > 1] = 1.0
    return aug_img


# Helper functions
def rotateImage(image, angle):
    rotation_matrix = cv2.getRotationMatrix2D(center=(SIZE / 2, SIZE / 2), angle=angle, scale=1.0)
    return cv2.warpAffine(image, rotation_matrix, settings.SHAPE)


def translateImage(image, tx, ty):
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, settings.SHAPE)


def scaleImage(image, sx, sy):
    scale_matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(image, scale_matrix, settings.SHAPE)


def shearImage(image, s):
    shear_matrix = np.float32([[1, s, 0], [0, 1, 0]])
    return cv2.warpAffine(image, shear_matrix, settings.SHAPE)

def erodeImage(image):
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    return cv2.erode(image, kernel=element)


# Expects a normalized image as input.
# Returns an array of augmented images.
def augmentImage(img, addNoise=True, addRotations=True, addTranslations=True, addScales=True, addShearing=True):
    # Array with augmented images
    images = [img]
    up, down, left, right = imageBorders(img)
    ## Addition of Gaussian noise
    if addNoise:
        images.append(noisyImage(img, stddev=0.05))

    ## Rotations of image
    if addRotations:
        for angle in np.arange(-30, 60, 30):  # evenly spaced values within a given interval
            if (angle != 0):
                images.append(rotateImage(img, angle))

    ## Translation of image
    if addTranslations:
        for tx in range(-8, 16, 8):
            for ty in range(-4, 4, 4):
                canTranslate = inBounds(up + ty) and inBounds(down + ty) and inBounds(left + tx) and inBounds(
                    right + tx)
                if not tx == 0 and not ty == 0 and canTranslate:
                    images.append(translateImage(img, tx, ty))

    ## Scaling of image
    if addScales:
        step = 0.25
        width_perc = (left - right) / SIZE
        height_perc = (up - down) / SIZE
        min_scale_x, max_scale_x = getMinAndMaxScale(width_perc)
        min_scale_y, max_scale_y = getMinAndMaxScale(height_perc)
        for sx in np.arange(min_scale_x, max_scale_x + step, step):
            for sy in np.arange(min_scale_y, max_scale_y + step, step):
                if not sy == 1 and not sy == 1:
                    if inBounds(down * sy) and inBounds(right * sx):
                        images.append(scaleImage(img, sx, sy))
    if addShearing:
        # Shear images to make the dataset more robust to different slant when writing.
        # The character seems slightly translated after the shearing operation. So we place the character back in the center
        # with a translate.
        shearedl = translateImage(shearImage(img, 0.2), -5, 0)
        shearedr = translateImage(shearImage(img, -0.2), 5, 0)
        images.append(shearedl)
        images.append(shearedr)
        # images.append(sheared)

    return images


def read_image(file_name):
    """
    Read an image.
    :param file_name: The path to an image
    :return: A normalized np array with correct dimensions
    """
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)


def preprocess_image(img, inverse=False):
    if img.shape != settings.SHAPE:
        img = cv2.resize(src=img, dsize=settings.SHAPE)
    # Contrast normalisation and inverting color values
    # We add a simple threshold for distinguishing the foreground and background.
    # Maybe this should actually be done before character recognition and not in preprocessing.
    conf = cv2.THRESH_BINARY if not inverse else cv2.THRESH_BINARY_INV
    thr, img = cv2.threshold(img, 127, 255, conf)
    return np.reshape(img, settings.IMG_SHAPE)


def augment_data(add_noise=True, add_rotations=True, add_translations=True, add_scales=True, add_shearing=True):
    # Opening files
    in_file_names = open(settings.CHAR_DATA_TXT_PATH, 'r').read().splitlines()
    out_file_names = open(settings.PREPROCESSED_CHAR_DATA_TXT_PATH, 'w')

    i = 0
    for file_name in in_file_names:
        # Input
        label = int(file_name.split('/')[1][-2:])
        file_name = settings.CHAR_DATA_PATH + file_name
        img = preprocess_image(read_image(file_name), inverse=True)
        erodeImage(img)
        # Data augmentation
        images = augmentImage(erodeImage(img), addNoise=add_noise, addRotations=add_rotations, addTranslations=add_translations,
                              addScales=add_scales, addShearing=add_shearing)
        images2 = augmentImage(img, addNoise=add_noise, addRotations=add_rotations, addTranslations=add_translations,
                              addScales=add_scales, addShearing=add_shearing)
        images.extend(images2)
        # Output
        for aug_img in images:
            aug_img = np.subtract(255, aug_img)  # Save augmented images as images without inverted color values.
            new_file_name = settings.PREPROCESSED_CHARS_PATH + 'image-{}-{}.png'.format(label, i)
            cv2.imwrite(new_file_name, aug_img)
            out_file_names.write(new_file_name + '\n')
            i += 1
            # Keeping track of time...
            if i % 250 == 0:
                print('({}) - {}: About {} seconds have passed...'.format(file_name, i, t.time() - start))

    end = t.time()
    print('\nThe execution time is {}'.format(end - start))

# Run this function for augmentation...
# augment_data()