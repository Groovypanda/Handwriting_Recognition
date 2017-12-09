import time as t
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from CharacterRecognition import graphics as gr
import settings
import cv2
from CharacterRecognition import utils
import CharacterRecognition.preprocessing as preprocess

'''
For this project we use the Chars74K dataset. It contains 62 classes, with about 3.4K handwritten characters.
'''

# Global constants
LEARNING_RATE = 1e-4
DECAY = 1e-3
KEEP_PROB = 0.5
BATCH_SIZE = 128
FILTER_SIZE = 5
NUM_CLASSES = 62
SIZE = settings.SIZE
NUM_CHANNELS = settings.NUM_CHANNELS


# Helper functions
def open_images():
    """
    Opens the dataset and preprocesses the images.
    :return: The labels of images, numpy pixel arrays with the image data, amount of images
    """
    file_names = open(settings.PREPROCESSED_CHAR_DATA_TXT_PATH, 'r').read().splitlines()
    print("Reading dataset of {} images".format(len(file_names)))
    file_names = shuffle(file_names)
    labels = [int(x.split('-')[1]) for x in file_names]
    length = len(file_names)
    data = np.zeros((length, SIZE, SIZE, 1), dtype=np.float32)
    i = 0
    for name in file_names:
        data[i] = preprocess.preprocess_image(preprocess.read_image(name))
        i += 1
    return labels, data, length


def label2vector(label):
    """
    Converts a label into one-hot encoding
    :param label:
    :return:
    """
    vector = np.zeros(NUM_CLASSES, dtype=np.int)
    vector[label - 1] = 1
    return vector


def vector2label(vector):
    """
    Converts the one-hot encoding into labels
    :param vector: The one-hot encoding
    :return: A label
    """
    return 1 + vector.index(1)


def new_weights(shape, weights):
    w = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    if weights:
        weights.append(w)
    return w


def new_biases(length):
    return tf.get_variable('biases', shape=[length], initializer=tf.constant_initializer(0.1))


def new_conv_layer(name, input, num_filters, filter_size, num_in_channels, global_weights=None, use_pooling=True):
    with tf.variable_scope("conv_layer_" + str(name)):
        shape = [filter_size, filter_size, num_in_channels, num_filters]
        weights = new_weights(shape, global_weights)
        biases = new_biases(num_filters)
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + biases
        if use_pooling:
            # ksize -> size of window
            # strides -> distance to move window
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(layer)


def new_fc_layer(name, input, num_in, num_out, global_weights=None):
    with tf.variable_scope("fc_layer_" + str(name)):
        weights = new_weights(shape=[num_in, num_out], weights=global_weights)
        biases = new_biases(length=num_out)
        return tf.nn.relu(tf.matmul(input, weights) + biases)


def show_progress(session, x, y, test_set):
    """
    This function is not complete
    :param test_set: 
    :return: 
    """
    y_cls = [vector2label(var) for var in test_set[1]]
    y_test = test_set[1]
    x_test = test_set[0]
    cls_pred = session.run(tf.argmax(y, 1), feed_dict={x: x_test, y: y_test})
    correct = (cls_pred == y)
    mask_false = correct == False
    incorrect_images = x[mask_false]
    gr.plot_images(incorrect_images[:9], y_cls[mask_false][:9], cls_pred[mask_false][:9])


def get_data():
    """
    :return: The training, validation and testset. These sets contain a list of respectively images, labels as vectors, labels
    """
    # Opening files
    labels, images, size = open_images()
    vector_labels = [label2vector(x) for x in labels]
    # Split the dataset into 3 parts
    train_X, rest_X, train_Y, rest_Y = train_test_split(images, vector_labels, train_size=0.8)
    validation_X, test_X, validation_Y, test_Y = train_test_split(rest_X, rest_Y, train_size=0.5)
    return (train_X, train_Y), (validation_X, validation_Y), (test_X, test_Y)


def create_neural_net(global_weights=None, train=True, base=1, filter_size=FILTER_SIZE, keep_prob=KEEP_PROB):
    """
    Builds a neural network which can be trained with an optimizer to recognise characters.
    :param train: Indicates if the net is used for training.
    :param keep_prob: Probability that connections after every fully connected layers are used.
    :param filter_size: Size of the filters
    :param base: experimental parameter, a higher base should produce better results. This should be a strict positive integer.
    :return: The input layer x, the output layer with the predicted values and a placeholder for the actual values. 
    """
    _x = tf.placeholder(tf.float32, (None, SIZE, SIZE, NUM_CHANNELS))  # batch size - height - width - channels
    _y = tf.placeholder(tf.int64, (None, NUM_CLASSES))  # batch size - classes
    base1 = base * 8
    base2 = base * 1024
    h1 = new_conv_layer(name=1, input=_x, filter_size=filter_size, num_filters=base1, num_in_channels=NUM_CHANNELS,
                        use_pooling=True, global_weights=global_weights)
    h2 = new_conv_layer(name=2, input=h1, filter_size=filter_size, num_filters=2 * base1, num_in_channels=base1,
                        use_pooling=True, global_weights=global_weights)
    h3 = new_conv_layer(name=3, input=h2, filter_size=filter_size, num_filters=3 * base1, num_in_channels=2 * base1,
                        use_pooling=True, global_weights=global_weights)
    h4 = tf.contrib.layers.flatten(h3)
    if train:
        h5 = new_fc_layer(name=4, input=h4, num_in=h4.shape[1], num_out=base2, global_weights=global_weights)
        h6 = tf.nn.dropout(h5, keep_prob=keep_prob)
        h7 = new_fc_layer(name=7, input=h6, num_in=base2, num_out=base2 / 2, global_weights=global_weights)
        h8 = tf.nn.dropout(h7, keep_prob=keep_prob)
    else:
        h7 = new_fc_layer(name=4, input=h4, num_in=h4.shape[1], num_out=base2, global_weights=global_weights)
        h8 = new_fc_layer(name=7, input=h7, num_in=base2, num_out=base2 / 2, global_weights=global_weights)
    h = new_fc_layer(name='final', input=h8, num_in=base2 / 2, num_out=NUM_CLASSES, global_weights=global_weights)
    return _x, _y, h


def create_training_operation(h, _y, learning_rate=LEARNING_RATE, decay=DECAY, global_weights=None):
    # Probability of each class (The closer the 0, the more likely it has that class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=_y)
    if global_weights:
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in global_weights]))
    else:
        weight_decay = 0
    loss_operation = tf.reduce_mean(cross_entropy) + decay * weight_decay

    # Optimisation of the neural network
    training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_operation)
    return training_operation


def train_net(n, restore=True, min_save=1.0):
    """
    Trains the network for n epochs with a new session
    Note: if this function has never been run, set restore to false!
    :param restore: Indicates if the previous session should be restored
    :param n: Amount of epochs to be trained
    :return: Returns the trained session
    """

    # Some variables required for training
    total_epochs = 0
    training_set, validation_set, test_set = get_data()
    weights = []
    x_train = training_set[0]
    y_train = training_set[1]
    x_validation = validation_set[0]
    y_validation = validation_set[1]
    _x, _y, h = create_neural_net(global_weights=weights)
    training_operation = create_training_operation(h, _y, global_weights=weights)
    session = create_session()
    # Initialize variables of neural network
    session.run(tf.global_variables_initializer())
    if restore:
        # Initialize variables of neural network with values of previous session.
        restore_session(session)

    # Actual training
    num_train = len(x_train)
    print('Training:\n')
    start = t.time()
    for i in range(total_epochs, total_epochs + n):
        for offset in range(0, num_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={_x: batch_x, _y: batch_y})
        validation_accuracy = session.run(get_accuracy(h, _y), feed_dict={_x: x_validation, _y: y_validation})
        if i % 10 == 0:
            print('EPOCH {}: Validation Accuracy = {:.3f}'.format(total_epochs, validation_accuracy))
        total_epochs += 1
        if validation_accuracy > min_save:
            print("New maximum accuracy achieved.")
            save_session(session)
            min_save = validation_accuracy
    validation_accuracy = session.run(get_accuracy(h, _y), feed_dict={_x: x_validation, _y: y_validation})
    print('EPOCH {}: Validation Accuracy = {:.3f}'.format(total_epochs, validation_accuracy))
    print("The training took: " + str(t.time() - start) + " seconds.")
    return session


def correct_prediction(predicted_output, expected_output):
    # Compare actual label with predicted label
    y_pred_cls = tf.argmax(predicted_output, 1)
    y_true_cls = tf.argmax(expected_output, 1)
    return tf.equal(y_pred_cls, y_true_cls)


def get_accuracy(predicted_output, expected_output):
    # Calculate the mean of the estimations to get the accuracy.
    return tf.reduce_mean(tf.cast(correct_prediction(predicted_output, expected_output), tf.float32))


def create_session():
    # Solving memory issues in GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create a tensorflow session and initialize the variables
    session = tf.Session(config=config)
    return session


def save_session(session, path=settings.SAVE_PATH):
    """
    Save variables of current session. 
    Side effect: closes session
    :return: None 
    """
    saver = tf.train.Saver()
    saver.save(session, path)


def restore_session(session, path=settings.SAVE_PATH):
    """
    Create a session initialized with the session as defined in the SAVE_PATH.
    :return: None
    """
    tf.train.Saver().restore(session, path)


def restore_train_save(epochs):
    """
    Helper method to continue training easily
    :param epochs: Amount of epochs to train
    :return: None
    """
    save_session(train_net(epochs, restore=True))


def train_save(epochs):
    """
        Helper method to train easily
        :param epochs: Amount of epochs to train
        :return: None
        """
    save_session(train_net(epochs, restore=False))


def img_to_prob(img, sessionargs):
    """
    Converts an image to a character probabilities.
    This function assumes there is a Model_500it subdirectory with a trained network model.
    :param sessionargs: Session and the neural network placeholders
    :param img: Path to an image which contains a character.
    :return: A list containing the probabilities
    of the image being a certain class (representing a letter or number).
    """
    img = preprocess.preprocess_image(img)
    (session, _x, _y, h) = sessionargs
    # Initialize variables of neural network
    return session.run(tf.nn.softmax(h), feed_dict={_x: [img]})[0]


def imgs_to_prob_list(images, sessionargs):
    prob_list = []
    for img in images:
        prob_list.append(img_to_prob(img, sessionargs))
    return prob_list


def img_to_text(image, sessionargs, n=1):
    """
    Converts an image into a character.
    :param Image: The input image
    :param n: Indicates the amount of results to be returned. 
              If n is higher than 1, the most probable characters and their probabilities will be returned.
    :param sessionargs: Session and the neural network placeholders
    :return: A list of possible characters and their probabilities. Size of this list equals n.
    """
    if n == 1:
        return utils.index2str(np.argmax(img_to_prob(image, sessionargs)))
    else:
        return most_probable_chars(img_to_prob(image, sessionargs), n)


def most_probable_chars(cls_pred, n):
    return list(reversed(sorted([(utils.index2str(i), x) for i, x in enumerate(cls_pred)], key=lambda x: x[1])[-n:]))


def init_session():
    """
    Fully creates an initialised session and returns an initialized neural network. 
    :return: 
    """
    session = tf.Session()
    _x, _y, h = create_neural_net(train=False)
    session.run(tf.global_variables_initializer())
    restore_session(session)
    return session, _x, _y, h


def examples():
    sessionargs = init_session()
    examples = ['a', '3', 'g', 'L']
    for ex in reversed(examples):
        for i in range(4):
            print(ex,
                  img_to_text(cv2.imread(settings.EXAMPLE_CHAR_PATH + ex + '_' + str(i) + ".png", 0),
                              sessionargs, n=1))

# train_net(1000, restore=False, min_save=0.79).close()

