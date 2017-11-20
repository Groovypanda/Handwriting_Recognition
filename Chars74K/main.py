import time as t

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan) #For complete printing of conf matrix.
import tensorflow as tf
import sklearn as sk
from scipy import misc  # used for interacting with images, need pillow dependency
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from Chars74K import graphics as gr

'''
For this project we use the Chars74K dataset. It contains 62 classes, with about 3.4K handwritten characters.
'''

# Global constants
FILE_NAMES_PATH = 'Chars74K/images.txt'
LEARNING_RATE = 1e-4
DECAY = 1e-4
KEEP_PROB = 0.5
EPOCHS = 5000
BATCH_SIZE = 512
SIZE = 32
FILTER_SIZE = 3
num_classes = 62
num_channels = 1
img_shape = (SIZE, SIZE, num_channels)


# Helper functions
def open_images():
    # There should be a better way of doing this.
    file_names = open(FILE_NAMES_PATH, 'r').read().splitlines()
    file_names = shuffle(file_names)
    labels = [int(x.split('-')[1]) for x in file_names]
    length = len(file_names)
    data = np.zeros((length, SIZE, SIZE, 1), dtype=np.float32)
    i = 0
    for name in file_names:
        img = misc.imread(name)
        #Scaling to [-1,1]
        #np.subtract(np.divide(np.multiply(2, img), 255), 1)
        img = np.divide(img, 255) #Scaling to [0,1]
        data[i] = np.reshape(img, img_shape)
        i += 1
    return labels, data, length


def vector_label(label):
    vector = np.zeros(num_classes, dtype=np.int)
    vector[label - 1] = 1
    return vector


def new_weights(name, shape):
    w = tf.get_variable('W' + name, shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    weights.append(w)
    return w


def new_biases(name, length):
    return tf.get_variable('B' + name, shape=[length], initializer=tf.constant_initializer(0.1))


def new_conv_layer(name, input, filter_size, num_filters, num_in_channels, use_pooling=True, strides=[1, 1, 1, 1]):
    shape = [filter_size, filter_size, num_in_channels, num_filters]
    weights = new_weights(name, shape)
    biases = new_biases(name, num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=strides, padding='SAME') + biases
    if use_pooling:
        # ksize -> size of window
        # strides -> distance to move window
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(layer)


def new_fc_layer(name, input, num_in, num_out):
    weights = new_weights(name, shape=[num_in, num_out])
    biases = new_biases(name, length=num_out)
    return tf.nn.relu(tf.matmul(input, weights) + biases)


def train(epochs):
    global total_epochs
    num_train = len(x_train)
    print('Training:\n')
    start = t.time()
    for i in range(total_epochs, total_epochs + epochs):
        for offset in range(0, num_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy = session.run(accuracy, feed_dict={x: x_validation, y: y_validation})
        if i%10==0:
            print('EPOCH {}: Validation Accuracy = {:.3f}\n'.format(total_epochs, validation_accuracy))
        total_epochs += 1
    print("The training took: " + str(t.time() - start) + " seconds.")


def test():
    cls_pred = session.run(y_pred_cls, feed_dict={x: x_test, y: y_test})
    correct = (cls_pred == y_test_cls)
    mask_false = correct == False
    mask_true = correct == True
    #correct_images = x_test[mask_true]
    incorrect_images = x_test[mask_false]
    gr.plot_images(incorrect_images[:9], y_test_cls[mask_false][:9], cls_pred[mask_false][:9])

# Opening files
labels, images, size = open_images()
vector_labels = [vector_label(x) for x in labels]

# Split the dataset into 3 parts
parts = [0.6, 0.2, 0.2]
sizes = [int(x * size) for x in parts]
ends = [sizes[0], sizes[0] + sizes[1]]
x_train, y_train, y_train_cls = np.array(images[:ends[0]]), np.array(vector_labels[:ends[0]]), np.array(
    labels[:ends[0]])
x_validation, y_validation, y_validation_cls = np.array(images[ends[0]:ends[1]]), np.array(
    vector_labels[ends[0]:ends[1]]), np.array(labels[ends[0]:ends[1]])
x_test, y_test, y_test_cls = np.array(images[ends[1]:]), np.array(vector_labels[ends[1]:]), np.array(labels[ends[1]:])

x = tf.placeholder(tf.float32, (None, SIZE, SIZE, num_channels))  # batch size - height - width - channels
y = tf.placeholder(tf.int64, (None, num_classes))  # batch size - classes
weights = []
base1 = 8
base2 = 1024
h1 = new_conv_layer(name='1', input=x, filter_size=FILTER_SIZE, num_filters=base1, num_in_channels=num_channels,
                    use_pooling=True)
h2 = new_conv_layer(name='2', input=h1, filter_size=FILTER_SIZE, num_filters=2 * base1, num_in_channels=base1,
                    use_pooling=True)
h3 = new_conv_layer(name='3', input=h2, filter_size=FILTER_SIZE, num_filters=3 * base1, num_in_channels=2 * base1,
                    use_pooling=True)
h4 = tf.contrib.layers.flatten(h3)
h5 = new_fc_layer(name='5', input=h4, num_in=h4.shape[1], num_out=base2)
h6 = tf.nn.dropout(h5, keep_prob=KEEP_PROB)
h7 = new_fc_layer(name='7', input=h6, num_in=base2, num_out=base2/2)
h8 = tf.nn.dropout(h7, keep_prob=KEEP_PROB)
final_h = new_fc_layer(name='final', input=h8, num_in=base2/2, num_out=num_classes)
# Probability of each class (The closer the 0, the more likely it has that class)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=final_h, labels=y)
weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights]))
loss_operation = tf.reduce_mean(cross_entropy) + DECAY * weight_decay
cost = tf.reduce_mean(cross_entropy)

# Optimisation of the neural network
training_operation = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_operation)
# Compare actual label with predicted label
y_pred_cls = tf.argmax(final_h, 1)
y_true_cls = tf.argmax(y, 1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# Calculate the mean of the estimations to get the accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Solving memory issues in GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a tensorflow session and initialize the variables
total_epochs = 0
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
train(EPOCHS)

print(max_validation_accuracy)