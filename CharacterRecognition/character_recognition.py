import time as t
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from CharacterRecognition import graphics as gr
import settings
import cv2
from CharacterRecognition import utils

'''
For this project we use the Chars74K dataset. It contains 62 classes, with about 3.4K handwritten characters.
'''

# Global constants
LEARNING_RATE = 1e-4
DECAY = 1e-4
KEEP_PROB = 0.5
EPOCHS = 2000
BATCH_SIZE = 512
FILTER_SIZE = 3
NUM_CLASSES = 62
SIZE = settings.SIZE
NUM_CHANNELS = settings.NUM_CHANNELS

# Global variable
weights = []


# Helper functions
def open_images():
    file_names = open(settings.PREPROCESSED_CHAR_DATA_TXT_PATH, 'r').read().splitlines()
    print("Reading dataset of {} images".format(len(file_names)))
    file_names = shuffle(file_names)
    labels = [int(x.split('-')[1]) for x in file_names]
    length = len(file_names)
    data = np.zeros((length, SIZE, SIZE, 1), dtype=np.float32)
    i = 0
    for name in file_names:
        data[i] = read_image(name)
        i += 1
    return labels, data, length


def read_image(file_name):
    """
    Read an image.
    :param invert: Indicates if color values should be inverted.
    :param file_name: The path to an image
    :return: A normalized np array with correct dimensions
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # Data normalisation and inverting color values
    if img.shape != settings.SHAPE:
        img = cv2.resize(src=img, dsize=settings.SHAPE)
    img = np.subtract(1, np.divide(img, 255))
    return np.reshape(img, settings.IMG_SHAPE)


def label2vector(label):
    vector = np.zeros(NUM_CLASSES, dtype=np.int)
    vector[label - 1] = 1
    return vector


def vector2label(vector):
    return 1 + vector.index(1)


def new_weights(shape):
    w = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    weights.append(w)
    return w


def new_biases(length):
    return tf.get_variable('biases', shape=[length], initializer=tf.constant_initializer(0.1))


def new_conv_layer(name, input, filter_size, num_filters, num_in_channels, use_pooling=True):
    with tf.variable_scope("conv_layer_" + str(name)):
        shape = [filter_size, filter_size, num_in_channels, num_filters]
        weights = new_weights(shape)
        biases = new_biases(num_filters)
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + biases
        if use_pooling:
            # ksize -> size of window
            # strides -> distance to move window
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return tf.nn.relu(layer)


def new_fc_layer(name, input, num_in, num_out):
    with tf.variable_scope("fc_layer_" + str(name)):
        weights = new_weights(shape=[num_in, num_out])
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
    parts = [0.7, 0.2, 0.1]
    sizes = [int(x * size) for x in parts]
    ends = [sizes[0], sizes[0] + sizes[1]]
    train = (np.array(images[:ends[0]]), np.array(vector_labels[:ends[0]]))
    validate = (np.array(images[ends[0]:ends[1]]), np.array(vector_labels[ends[0]:ends[1]]))
    test = (np.array(images[ends[1]:]), np.array(vector_labels[ends[1]:]))
    return train, validate, test


def create_neural_net(base=1):
    """
    Builds a neural network which can be trained with an optimizer to recognise characters.
    :param base: experimental parameter, a higher base should produce better results. This should be a strict positive integer.
    :return: The input layer x, the output layer with the predicted values and a placeholder for the actual values. 
    """
    x = tf.placeholder(tf.float32, (None, SIZE, SIZE, NUM_CHANNELS))  # batch size - height - width - channels
    y = tf.placeholder(tf.int64, (None, NUM_CLASSES))  # batch size - classes
    base1 = base * 8
    base2 = base * 1024
    h1 = new_conv_layer(name=1, input=x, filter_size=FILTER_SIZE, num_filters=base1, num_in_channels=NUM_CHANNELS,
                        use_pooling=True)
    h2 = new_conv_layer(name=2, input=h1, filter_size=FILTER_SIZE, num_filters=2 * base1, num_in_channels=base1,
                        use_pooling=True)
    h3 = new_conv_layer(name=3, input=h2, filter_size=FILTER_SIZE, num_filters=3 * base1, num_in_channels=2 * base1,
                        use_pooling=True)
    h4 = tf.contrib.layers.flatten(h3)
    h5 = new_fc_layer(name=4, input=h4, num_in=h4.shape[1], num_out=base2)
    h6 = tf.nn.dropout(h5, keep_prob=KEEP_PROB)
    h7 = new_fc_layer(name=7, input=h6, num_in=base2, num_out=base2 / 2)
    h8 = tf.nn.dropout(h7, keep_prob=KEEP_PROB)
    final_h = new_fc_layer(name='final', input=h8, num_in=base2 / 2, num_out=NUM_CLASSES)

    return x, y, final_h


def create_training_operation(predicted_output, expected_output):
    # Probability of each class (The closer the 0, the more likely it has that class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predicted_output, labels=expected_output)
    weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights]))
    loss_operation = tf.reduce_mean(cross_entropy) + DECAY * weight_decay

    # Optimisation of the neural network
    training_operation = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_operation)
    return training_operation


def train_net(n, restore=True):
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
    x_train = training_set[0]
    y_train = training_set[1]
    x_validation = validation_set[0]
    y_validation = validation_set[1]
    x, y, predicted_y = create_neural_net()
    training_operation = create_training_operation(predicted_y, y)
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
            session.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy = session.run(get_accuracy(predicted_y, y), feed_dict={x: x_validation, y: y_validation})
        if i % 10 == 0:
            print('EPOCH {}: Validation Accuracy = {:.3f}'.format(total_epochs, validation_accuracy))
        total_epochs += 1
    validation_accuracy = session.run(get_accuracy(predicted_y, y), feed_dict={x: x_validation, y: y_validation})
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


def save_session(session):
    """
    Save variables of current session. 
    Side effect: closes session
    :return: None 
    """
    saver = tf.train.Saver()
    saver.save(session, settings.SAVE_PATH)
    session.close()


def restore_session(session):
    """
    Create a session initialized with the session as defined in the SAVE_PATH.
    :return: The restored session
    """
    saver = tf.train.Saver()
    saver.restore(session, settings.SAVE_PATH)
    return session


def restore_train_save(epochs):
    """
    Helper method to continue training easily
    :param epochs: Amount of epochs to train
    :return: None
    """
    save_session(train_net(epochs))


def img_to_cls_pred(file_name):
    """
    Converts an image to a character probabilities.
    This function assumes there is a Model subdirectory with a trained network model.
    :param file_name: Path to an image which contains a character.
    :return: A list containing the probabilities
    of the image being a certain class (representing a letter or number).
    """
    img = read_image(file_name)
    x, y, predicted_y = create_neural_net()
    session = create_session()
    y_pred_cls = tf.argmax(predicted_y, 1)
    # Initialize variables of neural network
    session.run(tf.global_variables_initializer())
    restore_session(session)
    return session.run(predicted_y, feed_dict={x: [img]})[0]


def img_to_text(file_name):
    return utils.cls2str(np.argmax(img_to_cls_pred(file_name)))


start = ord('a')
end = ord('n')
for x in range(start, end + 1):
    print(img_to_text(settings.EXAMPLE_PATH + chr(x) + '.jpg'))
