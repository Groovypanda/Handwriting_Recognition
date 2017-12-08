import settings
import cv2
import numpy as np
import Tokenization.character_extraction_main as chrextr
from pathlib import Path
import sys
import pickle

WINDOW_SIZE = 5
MATRIX_DX = 5
START_LINE = 18


def open_images(start=0, amount=1000):
    file_entries = [x.split(' ') for x in
                    open(settings.CHAR_SEGMENTATION_DATA_TXT_PATH, 'r').read().splitlines()[
                    start + START_LINE:START_LINE + start + amount]]
    print("Reading new dataset of {} images, starting at image {}".format(len(file_entries), start))
    images = []
    files = [x[0] for x in file_entries]
    for file_name in files:
        parts = file_name.split('-')
        file_path = '/'.join([parts[0], parts[0] + '-' + parts[1], file_name + '.png'])
        images.append((file_path, cv2.imread(settings.CHAR_SEGMENTATION_DATA_PATH + file_path)))
    return images


def find_start(file_name):
    file_entries = [x.split(' ') for x in
                    open(settings.CHAR_SEGMENTATION_DATA_TXT_PATH, 'r').read().splitlines()[START_LINE:]]
    for (i, entry) in enumerate(file_entries):
        if entry[0] + '.png' == file_name:
            return i
    raise FileNotFoundError


def create_training_data(start=0, amount=1000):
    show_range = 1
    images = open_images(start, amount)
    n = len(images)
    with open(settings.CHAR_SEGMENTATION_TRAINING_TXT_PATH, "ab") as out_file:
        for (i, (img_path, img)) in enumerate(images):
            print("Showing image number {} of {}".format(i, n))
            split_data = []
            splits = chrextr.extract_character_separations(img[:, :, 0])
            for (x, y) in splits:
                split = False
                img_tmp = img.copy()
                for y in range(-show_range, show_range + 1):
                    img_tmp[:, x + y] = [0, 0, 255]
                cv2.imshow("img", img_tmp)  # Present splitpoint to user
                key = cv2.waitKey(0)
                if key == 13 or key == 32:  # User decides x is a splitpoint
                    split = True
                    # Create pixel matrix around splitpoint
                elif key == 27:  # escape
                    return
                else:  # user decides x is not a splitpoint
                    pass
                split_data.append((x, split))
            pickle.dump((start + i, img_path, split_data), out_file)


def start_training(requested_start=0):
    path = Path(settings.CHAR_SEGMENTATION_TRAINING_TXT_PATH)
    start = requested_start
    if path.exists():
        entries = read_training_data()
        if len(entries) != 0:
            start = entries[-1][0]
    create_training_data(start + 1)


def read_training_data():
    with open(settings.CHAR_SEGMENTATION_TRAINING_TXT_PATH, "rb") as in_file:
        entries = []
        EOF = False
        while not EOF:  # Non ideal way of reading files... But easiest way to make program fool proof.
            try:
                entry = pickle.load(in_file)
                entries.append(entry)
            except EOFError:
                EOF = True
        return entries


def feature_extractor(img, x):
    # This matrix has the shape: (2*MATRIX_DX, IMG_HEIGHT)
    # Where IMG_HEIGHT is the height of the image
    # Pixel matrix has reduced dimensions (color channels reduced to 1, grayscale)

    # Normalize the pixel_matrix
    pixel_matrix = img[:, x - MATRIX_DX:x + MATRIX_DX, 0]
    return pixel_matrix


"""
Source: https://research-repository.griffith.edu.au/bitstream/handle/10072/15242/11185.pdf%3Bsequence=1
For each segmentation point in a particular word (given by its xcoordinate),
a matrix of pixels is extracted and stored in an
A” training file. Each matrix is first normalised in size,
and then significantly reduced in size by a simple feature
extractor. The feature extractor breaks the segmentation
point matrix down into small windows of equal size and
analyses the density of black and white pixels. Therefore,
instead of presenting the raw pixel values of the
segmentation points to the ANN, only the densities of each
window are presented.
"""

if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == '-t' or arg == '--test':
            read_training_data()
        else:
            start_training(int(arg))
    else:
        start_training(0)
