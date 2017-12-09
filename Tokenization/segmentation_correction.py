import settings
import cv2
import Tokenization.character_extraction_main as chrextr
from pathlib import Path
import pickle
import numpy as np
import CharacterRecognition.character_recognition as net  # Remove this dependency?
import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import time

WINDOW_SIZE = 5
MATRIX_DX = 5
MATRIX_Y = 40
START_LINE = 18
SIZE = (MATRIX_Y / WINDOW_SIZE) * ((MATRIX_DX * 2) / WINDOW_SIZE)
LEARNING_RATE = 1e-4
BATCH_SIZE = 8

def open_images(start=0, amount=-1):
    start = start + START_LINE
    if amount == -1:
        files = open(settings.CHAR_SEGMENTATION_DATA_TXT_PATH, 'r').read().splitlines()[start:]
    else:
        files = open(settings.CHAR_SEGMENTATION_DATA_TXT_PATH, 'r').read().splitlines()[start:start + amount]
    file_entries = [x.split(' ') for x in files]
    print("Reading new dataset of {} images, starting at image {}".format(len(file_entries), start))
    images = []
    files = [x[0] for x in file_entries]
    for file_name in files:
        parts = file_name.split('-')
        file_path = '/'.join([parts[0], parts[0] + '-' + parts[1], file_name + '.png'])
        images.append((file_path, cv2.imread(settings.CHAR_SEGMENTATION_DATA_PATH + file_path)))
    return images


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
    # Take the pixel matrix of the segmentation point
    pixel_matrix = img[:, x - MATRIX_DX:x + MATRIX_DX, 0]
    # Normalize the matrix values
    thr, pixel_matrix = cv2.threshold(pixel_matrix, 127, 1, cv2.THRESH_BINARY)
    if pixel_matrix is None:
        return None
    (w, h) = pixel_matrix.shape
    if w == 0 or h == 0:
        return None
    # Normalise the matrix size
    pixel_matrix = cv2.resize(pixel_matrix, dsize=(2 * MATRIX_DX, MATRIX_Y))
    densities = []
    # Create small windows of equal size
    for row in range(0, pixel_matrix.shape[0], WINDOW_SIZE):
        for col in range(0, pixel_matrix.shape[1], WINDOW_SIZE):
            window = pixel_matrix[row:row + WINDOW_SIZE, col:col + WINDOW_SIZE]
            density = calculate_density(window)
            densities.append(density)
    return densities


def calculate_density(window):
    (w, h) = window.shape
    size = w * h
    # non_zero are white characters. We have to count zero.
    # which is size minus non zero.
    return (size - np.count_nonzero(window)) / size


def convert_training_data(images, entries):
    """
    Convert the training data into density matrices which can be fed to the neural network
    :param images: Images of entries
    :param entries: Corresponding training data entries
    :return: A list of density matrices and their labels
    """
    labels = []
    pixel_matrices = []
    for (i, (name, img)) in enumerate(images):
        _, _, split_data = entries[i]
        for (splitpoint, is_splitpoint) in split_data:
            pixel_matrix = img[:, splitpoint - MATRIX_DX:splitpoint + MATRIX_DX, 0]
            # Normalize the matrix values
            thr, pixel_matrix = cv2.threshold(pixel_matrix, 127, 1, cv2.THRESH_BINARY)
            if not pixel_matrix is None:
                pixel_matrix = cv2.resize(pixel_matrix, dsize=(2 * MATRIX_DX, MATRIX_Y))
                pixel_matrix = np.reshape(pixel_matrix, (MATRIX_Y, 2*MATRIX_DX, 1))
                label = [0, 1] if is_splitpoint else [1, 0]
                labels.append(label)
                pixel_matrices.append(pixel_matrix)
    return pixel_matrices, labels


def convert_training_data2(images, entries):
    """
    Convert the training data into density matrices which can be fed to the neural network
    :param images: Images of entries
    :param entries: Corresponding training data entries
    :return: A list of density matrices and their labels
    """
    labels = []
    density_matrices = []
    for (i, (name, img)) in enumerate(images):
        _, _, split_data = entries[i]
        for (splitpoint, is_splitpoint) in split_data:
            density_matrix = feature_extractor(img, splitpoint)
            if not density_matrix is None:
                label = [0, 1] if is_splitpoint else [1, 0]
                labels.append(label)
                density_matrices.append(density_matrix)
    return density_matrices, labels


def create_neural_net():
    """
    Builds a neural network which can be trained with an optimizer to decide if potential splitting points are actual splitting points.
    :return: The input layer x, the output layer with the predicted values and a placeholder for the expected values.
    """
    OUT_SIZE = 2
    NUM_CHANNELS = 1
    _x = tf.placeholder(tf.float32, (None, MATRIX_Y, 2*MATRIX_DX, NUM_CHANNELS))
    _y = tf.placeholder(tf.float32, (None, OUT_SIZE))
    h1 = net.new_conv_layer(name=1, input=_x, num_in_channels=NUM_CHANNELS, num_filters=3, filter_size=5)
    h2 = tf.contrib.layers.flatten(h1)
    h3 = net.new_fc_layer(name=2, input=h2, num_in=h2.shape[1], num_out=16)
    h4 = tf.nn.dropout(h3, keep_prob=0.7)
    h = net.new_fc_layer(name=3, input=h4, num_in=16, num_out=OUT_SIZE)
    return _x, _y, h


def create_neural_net2():
    """
    Builds a neural network which can be trained with an optimizer to decide if potential splitting points are actual splitting points.
    :return: The input layer x, the output layer with the predicted values and a placeholder for the expected values.
    """
    OUT_SIZE = 2
    _x = tf.placeholder(tf.float32, (None, SIZE))
    _y = tf.placeholder(tf.float32, (None, OUT_SIZE))
    h = net.new_fc_layer(name=1, input=_x, num_in=SIZE, num_out=OUT_SIZE)
    return _x, _y, h


def create_training_operation(h, _y, learning_rate=LEARNING_RATE):
    # Probability of each class (The closer the 0, the more likely it has that class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    # Optimisation of the neural network
    training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_operation)
    return training_operation


def train_net(epochs):
    """
    Trains the network for n epochs with a new session
    Note: if this function has never been run, set restore to false!
    :param epochs: Amount of epochs to be trained
    :return: Returns the trained session
    """
    session = tf.Session()
    (x_train, y_train), (x_validation, y_validation) = get_data()
    _x, _y, h = create_neural_net()
    training_operation = create_training_operation(h, _y)
    session.run(tf.global_variables_initializer())
    # Actual training
    num_train = len(x_train)
    print('Training:\n')
    start = time()
    for i in range(epochs):
        for offset in range(0, num_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={_x: batch_x, _y: batch_y})
        validation_accuracy = session.run(net.get_accuracy(h, _y), feed_dict={_x: x_validation, _y: y_validation})
        if i % 4 == 0:
            print('EPOCH {}: Validation Accuracy = {:.3f}'.format(i, validation_accuracy))
    print("The training took: " + str(time() - start) + " seconds.")
    return session


def get_data():
    entries = read_training_data()
    start = entries[0][0]
    end = entries[-1][0]
    amount = end - start
    imgs = open_images(start, amount)
    (x, y) = convert_training_data(imgs, entries)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8)
    return (x_train, y_train), (x_validation, y_validation)


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

net.save_session(train_net(300), settings.CHAR_SEGMENTATION_NET_SAVE_PATH)
