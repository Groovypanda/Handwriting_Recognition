import tensorflow as tf

import definitions
from character_recognition import new_conv_layer
from character_recognition import new_fc_layer

"""
Python file for variations of neural networks and training operation functions. 
It uses the same default valus as CharacterRecognition.py
It never uses weight decay. 
"""

# Global constants (default values)
LEARNING_RATE = 1e-4
DECAY = 1e-4
KEEP_PROB = 0.5
BATCH_SIZE = 512
FILTER_SIZE = 3
NUM_CLASSES = 62
SIZE = definitions.SIZE
NUM_CHANNELS = definitions.NUM_CHANNELS

f = 'fully connected layer'
fx = 'fully connected layer with dropout'
c = 'convolutional layer'
cx = 'convolutional layer with max pooling'
'''
Compact string representation for the neural network in order to easily build variations to experiment with.
Arguments for these layers are arbitrarily chosen as experiments for these values already exist.
Array of these arguments represents the neural network configuration. For ease the configuration for the convolutional layers
and fully connected layers are split in 2. 
fc -> fully connected layer
fc_drop -> fully connected layer + dropout
conv -> convolutional layer
conv_pool -> convolutional layer + 2x2 max pooling
'''


def net_configurations():
    conf_default = ("default", ([cx, cx, cx], [fx, fx]))
    conf_many_c = ("many_c", ([cx, c, cx, c, cx, c], [fx, fx]))
    conf_no_max_pool = ("no_max_pool", ([c, c, c], [fx, fx]))
    conf_4_cx = ("4_cx", ([cx, cx, cx, cx], [fx, fx]))
    conf_2_cx = ("2_cx", ([cx, cx], [fx, fx]))
    conf_2_c_1_f = ("2_c_1_f", ([cx, cx], [fx]))
    conf_many_f = ("many_f", ([cx, cx, cx], [fx, fx, f]))
    conf_very_large = ("very_large", ([cx, c, cx, c, cx, c], [fx, fx, f]))
    conf_very_small = ("very_small", ([cx], [fx]))
    return [conf_default, conf_many_c, conf_no_max_pool, conf_4_cx, conf_2_cx, conf_2_c_1_f, conf_many_f, conf_very_large,
            conf_very_small]


def create_neural_net(configuration):
    """
    TODO: This function does not work correctly... 
    Builds an experimental neural network which can be trained with an optimizer to recognise characters.
    :param A string configuration for the neural network.   
    :return: The input layer x, the output layer with the predicted values and a placeholder for the actual values. 
    """
    _x = tf.placeholder(tf.float32, (None, SIZE, SIZE, NUM_CHANNELS))  # batch size - height - width - channels
    _y = tf.placeholder(tf.int64, (None, NUM_CLASSES))  # batch size - classes
    layers = [_x]
    base1 = 8
    base2 = 1024
    prev_filters = NUM_CHANNELS
    for (i, conv_layer) in enumerate(configuration[0]):
        num_filters = (i + 1) * base1
        if conv_layer == c:
            pooling = False
        elif conv_layer == cx:
            pooling = True
        else:  # Error
            raise Exception("Incorrect neural network configuration")
        layers.append(new_conv_layer(name=str(i), input=layers[-1], filter_size=FILTER_SIZE, num_filters=(i + 1) * base1,
                                    num_in_channels=prev_filters, use_pooling=pooling))
        prev_filters = num_filters

    layers.append(tf.contrib.layers.flatten(layers[-1]))

    for (i, fc_layer) in enumerate(configuration[1]):
        num_filters = (i + 1) * base2
        if fc_layer == f:
            layers.append(new_fc_layer(name=i, input=layers[-1], num_in=layers[-1].shape[1], num_out=num_filters))
        elif fc_layer == fx:
            layers.append(new_fc_layer(name=i, input=layers[-1], num_in=layers[-1].shape[1], num_out=num_filters))
            layers.append(tf.nn.dropout(layers[-1], keep_prob=KEEP_PROB))
        else:  # Error
            raise Exception("Incorrect neural network configuration")
    h = new_fc_layer(name='final', input=layers[-1], num_in=layers[-1].shape[1], num_out=NUM_CLASSES)
    layers.append(h)
    return _x, _y, h


def training_operation(h, _y, optimizer):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    # Optimisation of the neural network
    training_operation = optimizer.minimize(loss_operation)
    return training_operation


# See: https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
def training_op_with_gd(h, _y):
    return training_operation(h, _y, tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE))


def training_op_with_mom(h, _y):
    return training_operation(h, _y, tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9))


def training_op_with_adadelta(h, _y):
    return training_operation(h, _y, tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE))


def training_op_with_adagrad(h, _y):
    return training_operation(h, _y, tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE))

def training_op_with_adam(h, _y):
    return training_operation(h, _y, tf.train.AdamOptimizer(learning_rate=LEARNING_RATE))