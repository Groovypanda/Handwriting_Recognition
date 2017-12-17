import time as t

import cv2
import numpy as np
import random
import definitions

'''
For this project we use the Chars74K dataset. It contains 64 classes, with about 74K images.
'''

# Global constants
start = t.time()
SIZE = definitions.SIZE
EMPTY_VALUE = 0.0
WRITE_VALUE = 1.0


def imageBorders(image):
    threshold = 2  # Allow a smart part of the letter to be outside of the image after translation.
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
    if len(possible_horizontal_borders) == 0:  # Fail safe mechanism.
        possible_horizontal_borders = np.where(amount_hor >= threshold / 2)[0]
    if len(possible_vertical_borders) == 0:
        possible_vertical_borders = np.where(amount_ver >= threshold / 2)[0]
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
    return cv2.warpAffine(image, rotation_matrix, definitions.SHAPE)


def translateImage(image, tx, ty):
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, translation_matrix, definitions.SHAPE)


def scaleImage(image, sx, sy):
    scale_matrix = np.float32([[sx, 0, 0], [0, sy, 0]])
    return cv2.warpAffine(image, scale_matrix, definitions.SHAPE)


def shearImage(image, s):
    shear_matrix = np.float32([[1, s, 0], [0, 1, 0]])
    return cv2.warpAffine(image, shear_matrix, definitions.SHAPE)


def erodeImage(image):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(image, kernel=element)

# Expects a normalized image as input.
# Returns an array of augmented images.
def augmentImage(image, add_noise, add_rotations, add_translations, add_scales, add_shearing, add_one=False):
    """
    Augment an image with the chosen techniques.
    :param image: Image to augment
    :param add_noise:
    :param add_rotations:
    :param add_translations:
    :param add_scales:
    :param add_shearing:
    :param add_one: True -> Indicates one of the given techniques, False -> add atleast one of the given technique.
    :return:
    """
    # Array with augmented images
    images = [image]

    up, down, left, right = imageBorders(image)
    ## Addition of Gaussian noise
    if add_noise:
        images.append(noisyImage(image, stddev=0.05))

    ## Rotations of image
    if add_rotations:
        rotatedImages = []
        for angle in np.arange(-30, 60, 30):  # evenly spaced values within a given interval
            if angle != 0:
                    rotatedImages.append(rotateImage(image, angle))
        if add_one:
            images.append(random.choice(rotatedImages))
        else:
            images.extend(rotatedImages)

    ## Translation of image
    if add_translations:
        translatedImages = []
        for tx in range(-16, 32, 16):
            for ty in range(-8, 8, 8):
                # if bool(getrandbits(1)):  # Enough variation in translation, but limit amount of examples
                    canTranslate = inBounds(up + ty) and inBounds(down + ty) and inBounds(left + tx) and inBounds(
                        right + tx)
                    if not tx == 0 and not ty == 0 and canTranslate:
                        translatedImages.append(translateImage(image, tx, ty))
        if len(translatedImages) > 0:
            if add_one:
                images.append(random.choice(translatedImages))
            else:
                images.extend(translatedImages)

    ## Scaling of image
    if add_scales:
        scaledImages = []
        step = 0.25
        width_perc = (left - right) / SIZE
        height_perc = (up - down) / SIZE
        min_scale_x, max_scale_x = getMinAndMaxScale(width_perc)
        min_scale_y, max_scale_y = getMinAndMaxScale(height_perc)
        for sx in np.arange(min_scale_x, max_scale_x + step, step):
            for sy in np.arange(min_scale_y, max_scale_y + step, step):
                if not sy == 1 and not sy == 1:
                    if inBounds(down * sy) and inBounds(right * sx):
                        scaledImages.append(scaleImage(image, sx, sy))
        if len(scaledImages) > 0:
            if add_one:
                images.append(random.choice(scaledImages))
            else:
                images.extend(scaledImages)
    if add_shearing:
        # Shear images to make the dataset more robust to different slant when writing.
        shearedImages = [shearImage(image, 0.2), shearImage(image, -0.2)]
        if(add_one):
            images.append(shearedImages[random.randrange(0, len(shearedImages))])
        else:
            images.extend(shearedImages)
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
    if img.shape != definitions.SHAPE:
        img = cv2.resize(src=img, dsize=definitions.SHAPE)
    # Contrast normalisation and inverting color values
    # We add a simple threshold for distinguishing the foreground and background.
    # Maybe this should actually be done before character recognition and not in preprocessing.
    conf = cv2.THRESH_BINARY if not inverse else cv2.THRESH_BINARY_INV
    thr, img = cv2.threshold(img, 127, 1, conf)
    return np.reshape(img, definitions.IMG_SHAPE)


def augment_data(add_noise=False, add_rotations=True, add_translations=True, add_scales=True, add_shearing=True,  add_erode = True, add_one=False, confirm=True):
    if confirm:
        inp = input("Are you sure you want to preprocess the data? Insert y(es) to continue.\n")
        if inp != 'y' and inp != 'yes':
            return
    # Opening files
    in_file_names = open(definitions.CHARSET_INFO_PATH, 'r').read().splitlines()
    out_file_names = open(definitions.PREPROCESSED_CHARSET_INFO_PATH, 'w')

    i = 0
    for file_name in in_file_names:
        # Input
        label = int(file_name.split('/')[1][-2:])
        file_name = definitions.CHARSET_PATH + file_name
        img = preprocess_image(read_image(file_name), inverse=True)
        # Data augmentation
        images = augmentImage(img, add_noise=add_noise, add_rotations=add_rotations, add_translations=add_translations,
                              add_scales=add_scales, add_shearing=add_shearing, add_one=add_one)
        if add_erode:
            if add_one:
                images.append(erodeImage(img))
            else:
                images2 = augmentImage(erodeImage(img), add_noise=add_noise, add_rotations=add_rotations, add_translations=add_translations,
                                add_scales=add_scales, add_shearing=add_shearing, add_one=add_one)
                images.extend(images2)
        # Output
        for aug_img in images:
            aug_img = np.multiply(255, np.subtract(1, aug_img))  # Save augmented images as images without inverted color values.
            new_file_name = definitions.PREPROCESSED_CHARSET_PATH + 'image-{}-{}.png'.format(label, i)
            cv2.imwrite(new_file_name, aug_img)
            out_file_names.write(new_file_name + '\n')
            i += 1
            # Keeping track of time...
            if i % 250 == 0:
                print('({}) - {}: About {} seconds have passed...'.format(file_name, i, t.time() - start))

    end = t.time()
    print('\nThe execution time is {}'.format(end - start))

def augment_data_fake():
    in_file_names = open(definitions.CHARSET_INFO_PATH, 'r').read().splitlines()
    out_file_names = open(definitions.PREPROCESSED_CHARSET_INFO_PATH, 'w')

    i = 0
    for file_name in in_file_names:
        # Input
        label = int(file_name.split('/')[1][-2:])
        file_name = definitions.CHARSET_PATH + file_name
        img = preprocess_image(read_image(file_name), inverse=True)
        # Data augmentation
        images = [img, img]
        # Output
        for aug_img in images:
            aug_img = np.multiply(255, np.subtract(1,
                                                   aug_img))  # Save augmented images as images without inverted color values.
            new_file_name = definitions.PREPROCESSED_CHARSET_PATH + 'image-{}-{}.png'.format(label, i)
            cv2.imwrite(new_file_name, aug_img)
            out_file_names.write(new_file_name + '\n')
            i += 1
            # Keeping track of time...
            if i % 250 == 0:
                print('({}) - {}: About {} seconds have passed...'.format(file_name, i, t.time() - start))

    end = t.time()
    print('\nThe execution time is {}'.format(end - start))

def reset_data():
    in_file_names = open(definitions.CHARSET_INFO_PATH, 'r').read().splitlines()
    out_file_names = open(definitions.PREPROCESSED_CHARSET_INFO_PATH, 'w')

    i = 0
    for file_name in in_file_names:
        # Input
        label = int(file_name.split('/')[1][-2:])
        file_name = definitions.CHARSET_PATH + file_name
        img = preprocess_image(read_image(file_name), inverse=True)
        # Data augmentation
        images = [img]
        # Output
        for aug_img in images:
            aug_img = np.multiply(255, np.subtract(1,
                                                   aug_img))  # Save augmented images as images without inverted color values.
            new_file_name = definitions.PREPROCESSED_CHARSET_PATH + 'image-{}-{}.png'.format(label, i)
            cv2.imwrite(new_file_name, aug_img)
            out_file_names.write(new_file_name + '\n')
            i += 1
            # Keeping track of time...
            if i % 250 == 0:
                print('({}) - {}: About {} seconds have passed...'.format(file_name, i, t.time() - start))

    end = t.time()
    print('\nThe execution time is {}'.format(end - start))

def show_aug_img():
    in_name = definitions.CHARSET_PATH + "Img/Sample012/img012-002.png"
    out_name = definitions.PROJECT_PATH + "Report/images/"
    img = preprocess_image(read_image(in_name), inverse=True)
    names = ["original", "noisy", "translated", "rotated", "scaled", "sheared", "eroded"]
    imgs = [img, noisyImage(img, 0.05), translateImage(img, -8, 4), rotateImage(img, 30),scaleImage(img, 1.25, 0.75), translateImage(shearImage(img, -0.2), 0, 0), erodeImage(img)]
    imgs = [np.multiply(255, np.subtract(1, aug_img)) for aug_img in imgs]
    for i, aug_img in enumerate(imgs):
        cv2.imwrite(out_name + names[i] + '.png',  aug_img)
