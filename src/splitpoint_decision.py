import pickle
from pathlib import Path
from time import time

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import character_extraction as char_ex
import character_recognition as net  # Remove this dependency?
import definitions

WINDOW_SIZE = 5
MATRIX_DX = 6
MATRIX_Y = 42
START_LINE = 18
SIZE = (MATRIX_Y / WINDOW_SIZE) * ((MATRIX_DX * 2) / WINDOW_SIZE)
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
DECAY = 1e-3
DROPOUT = 0.7


def open_images(start=0, amount=-1):
    start = start + START_LINE
    if amount == -1:
        files = open(definitions.WORDSET_INFO_PATH, 'r').read().splitlines()[start:]
    else:
        files = open(definitions.WORDSET_INFO_PATH, 'r').read().splitlines()[start:start + amount]
    file_entries = [x.split(' ') for x in files]
    print("Reading new dataset of {} images, starting at image {}".format(len(file_entries), start))
    images = []
    files = [(x[0], x[-1]) for x in file_entries]
    for (file_name, word) in files:
        parts = file_name.split('-')
        file_path = '/'.join([parts[0], parts[0] + '-' + parts[1], file_name + '.png'])
        images.append((file_path, word, cv2.imread(definitions.WORDSET_PATH + file_path)))
    return images


def create_training_data(start=0, amount=10):
    images = open_images(start, amount)
    n = len(images)
    with open(definitions.WORD_SPLITTING_PATH, "ab") as out_file:
        for (i, (img_path, word, img)) in enumerate(images):
            print("Displaying word {} - Showing image number {} of {}".format(word, i, n))
            splits = char_ex.extract_character_separations(img[:, :, 0])
            split_data = manual_split_point_detection(img, splits)
            if split_data is None:
                return
            pickle.dump((start + i, img_path, split_data), out_file, protocol=2)


def show_splitpoints(img, splits):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    show_range = 1
    for (x, _) in splits:
        for y in range(-show_range, show_range + 1):
            img[:, x + y] = [0, 0, 255]
    return img

def manual_split_point_detection(img, splits):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    split_data = []
    show_range = 1
    for (x, _) in splits:
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
    return split_data


def start_data_creation(requested_start=0):
    path = Path(definitions.WORD_SPLITTING_PATH)
    start = requested_start if not requested_start is None else 0
    if path.exists():
        entries = read_training_data()
        if len(entries) != 0:
            start = entries[-1][0]
    create_training_data(start + 1)


def read_training_data():
    with open(definitions.WORD_SPLITTING_PATH, "rb") as in_file:
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
    for (i, (name, word, img)) in enumerate(images):
        _, _, split_data = entries[i]
        for (splitpoint, is_splitpoint) in split_data:
            pixel_matrix = get_pixel_matrix(img, splitpoint)
            if not pixel_matrix is None:
                label = [0, 1] if is_splitpoint else [1, 0]
                labels.append(label)
                pixel_matrices.append(pixel_matrix)
    return pixel_matrices, labels


def get_pixel_matrix(img, splitpoint):
    pixel_matrix = img[:, splitpoint - MATRIX_DX:splitpoint + MATRIX_DX, 0]
    # Normalize the matrix values
    thr, pixel_matrix = cv2.threshold(pixel_matrix, 127, 1, cv2.THRESH_BINARY)
    if not pixel_matrix is None:
        pixel_matrix = cv2.resize(pixel_matrix, dsize=(2 * MATRIX_DX, MATRIX_Y))
        pixel_matrix = np.reshape(pixel_matrix, (MATRIX_Y, 2 * MATRIX_DX, 1))
        return pixel_matrix
    else:
        return None


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


def create_neural_net(global_weights=None, train=True):
    """
    Builds a neural network which can be trained with an optimizer to decide if potential splitting points are actual splitting points.
    :return: The input layer x, the output layer with the predicted values and a placeholder for the expected values.
    """
    OUT_SIZE = 2
    NUM_CHANNELS = 1
    with tf.variable_scope("SplitpointDecision"):
        _x = tf.placeholder(tf.float32, (None, MATRIX_Y, 2 * MATRIX_DX, NUM_CHANNELS))
        _y = tf.placeholder(tf.float32, (None, OUT_SIZE))
        h1 = net.new_conv_layer(name=1, input=_x, num_in_channels=NUM_CHANNELS, num_filters=3, filter_size=5,
                                global_weights=global_weights, use_pooling=False)
        #h2 = net.new_conv_layer(name=2, input=h1, num_in_channels=3, num_filters=6, filter_size=5,
        #                        global_weights=global_weights, use_pooling=False)
        h3 = tf.contrib.layers.flatten(h1)
        h4 = net.new_fc_layer(name=3, input=h3, num_in=h3.shape[1], num_out=64, global_weights=global_weights)
        if False:
            h6 = net.new_fc_layer(name=4, input=h4, num_in=64, num_out=16, global_weights=global_weights)
            h5 = tf.nn.dropout(h6, keep_prob=DROPOUT)
        else:
            h5 = net.new_fc_layer(name=4, input=h4, num_in=64, num_out=16, global_weights=global_weights)
        h = net.new_fc_layer(name='final', input=h5, num_in=16, num_out=OUT_SIZE, global_weights=global_weights)
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


def train_net(epochs, min_save=1.0):
    """
    Trains the network for n epochs with a new session
    Note: if this function has never been run, set restore to false!
    Min_save indicates from which value the session should start saving. The weights with the highest accuracy will be in the final model.
    :param epochs: Amount of epochs to be trained
    :return: Returns the trained session
    """
    session = tf.Session()
    global_weights = []
    (x_train, y_train), (x_validation, y_validation) = get_data()
    _x, _y, h = create_neural_net(global_weights)
    training_operation = net.create_training_operation(h, _y, learning_rate=LEARNING_RATE, decay=DECAY,
                                                       global_weights=global_weights)
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
        if validation_accuracy > min_save:
            print("New maximum accuracy achieved. {}".format(validation_accuracy))
            net.save_session(session, definitions.MODEL_CHAR_SEGMENTATION_PATH)
            min_save = validation_accuracy
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


def init_session():
    """
    Fully creates an initialised session and returns an initialized neural network.
    :return:
    """
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with session.graph.as_default():
        # Create variables and ops.
        _x, _y, h = create_neural_net(train=False)
        session.run(tf.global_variables_initializer())
        net.restore_session(session, path=definitions.MODEL_CHAR_SEGMENTATION_PATH)
        return session, _x, _y, h


def decide_splitpoints(img, potential_split_points, sessionargs):
    """
    Checks if the given splitpoints are actual splitpoints in the image using a neural network to classify the splitpoint 
    as correct or incorrect.
    :param img: Image of a word
    :param potential_split_points: List of potential splitpoints 
    :param sessionargs: Session and neural network variables.
    :return: A list where every item indicates if the corresponding value in potential_split_points is a split point.
    """
    if len(potential_split_points) > 0:
        (session, _x, _y, h) = sessionargs
        img = np.expand_dims(img, axis=2)
        pixel_matrices = []
        for (splitpoint,y) in potential_split_points:
            pixel_matrix = get_pixel_matrix(img, splitpoint)
            if not pixel_matrix is None :
                pixel_matrices.append(pixel_matrix)
        actual_splitpoints = session.run(tf.nn.softmax(h), feed_dict={_x: pixel_matrices})
        return [bool(x) for x in np.argmax(actual_splitpoints, axis=1)]
    else:
        return []


def convert_pickle_data():
    entries = read_training_data()
    with open(definitions.WORD_SPLITTING_PATH, "wb") as out_file:
        for entry in entries:
            pickle.dump(entry, out_file, protocol=2)
