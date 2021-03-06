import os
import time as t

import tensorflow as tf
from numpy import savetxt

from src.experiments import experiment_nets as en
import definitions, character_recognition, character_preprocessing

"""
This script tries to test the effectiveness of certain parameters to find an optimal solution
for recognising characters.
"""

# Increase this for actual experiments.
ITERATIONS = 4
EPOCHS = 200


def compare_base():
    """
    Check the effect of different layer sizes.
    :return:
    """
    for i in range(1, 4):
        print("Running experiments with base " + str(i))
        run_experiment("base_" + str(i), base=i)


def compare_learning_rate():
    """
    Check the effect of different learning rates.
    :return:
    """
    learning_rates = [1e-3, 1e-4, 1e-5]
    for x in learning_rates:
        print("Running experiments with learning rate " + str(x))
        run_experiment("learningrate_" + str(x), learning_rate=x)


def compare_filter_size():
    """
    Check the effect of different filter sizes.
    :return:
    """
    sizes = [3, 5, 7]
    for x in sizes:
        print("Running experiments with filter_size " + str(x))
        run_experiment("filtersize_" + str(x), filter_size=x)


def compare_keep_prob():
    """
    Check the effect of different percentage of dropout.
    :return:
    """
    keep_probs = [0.3, 0.5, 0.7, 0.9]
    for x in keep_probs:
        print("Running experiments with keep probability " + str(x))
        run_experiment("keepprob_" + str(x), keep_prob=x)


def compare_weight_decay():
    """
    Check the effect of regulation with weight decay.
    :return:
    """
    decays = [0, 1e-3, 1e-4, 1e-5]
    for x in decays:
        print("Running experiments with weight decay " + str(x))
        run_experiment("decay_" + str(x), weight_decay=x)


def compare_batch_size():
    """
        Check of feeding different batch sizes as training data.
        :return:
        """
    # sizes = [128, 256, 512, 1024]
    sizes = [1024]
    for x in sizes:
        print("Running experiments with batch size " + str(x))
        run_experiment("batchsize_" + str(x), batch_size=x, start=2)


def compare_epochs():
    """
        Check the effect of running more or less iterations.
        As I already know the exact effect of this, and this function will take very long.
        I might skip this experiment.
    :return:
    """
    epochs = [100, 250, 500, 1000]
    for x in epochs:
        print("Running experiments with " + str(x) + " iterations")
        run_experiment("epochs_" + str(x), n=x)

def run_experiment_with_orig_net(name):
    for i in range(ITERATIONS):
        with tf.variable_scope("Experiment_" + name + "_" + str(i)):
            print("Experiment: " + str(i))
            character_recognition.train_net(name=name, iteration=i, n=200)


def compare_preprocessing_params():
    """
    Check the effect of different preprocessing steps. Each preprocessing step is tested on its own.
    :return:
    """
    character_preprocessing.augment_data_fake()
    run_experiment_with_orig_net("preprocess_" + "none")
    character_preprocessing.augment_data(add_noise=False, add_rotations=True, add_scales=False, add_translations=False,
                                         add_shearing=False, add_one=True, confirm=False, add_erode=False)
    run_experiment_with_orig_net("preprocess_" + "rotate")
    character_preprocessing.augment_data(add_noise=False, add_rotations=False, add_scales=True, add_translations=False,
                                         add_shearing=False, add_one=True, confirm=False, add_erode=False)
    run_experiment_with_orig_net("preprocess_" + "scale")
    character_preprocessing.augment_data(add_noise=False, add_rotations=False, add_scales=False, add_translations=True,
                                         add_shearing=False, add_one=True, confirm=False, add_erode=False)
    run_experiment_with_orig_net("preprocess_" + "translate")
    character_preprocessing.augment_data(add_noise=False, add_rotations=False, add_scales=False, add_translations=False,
                                         add_shearing=True, add_one=True, confirm=False, add_erode=False)
    run_experiment_with_orig_net("preprocess_" + "shear")
    character_preprocessing.augment_data(add_noise=False, add_rotations=False, add_scales=False, add_translations=False,
                                         add_shearing=False, add_erode=True, add_one=True, confirm=False)
    run_experiment_with_orig_net("preprocess_" + "erode")


def compare_optimizer():
    """
    Check the effect of changing the training operation
    :return:
    """
    #run_experiment("optimizer_" + "adam", training_operation=en.training_op_with_adam)
    run_experiment("optimizer_" + "mom", training_operation=en.training_op_with_mom)
    run_experiment("optimizer_" + "adadelta", training_operation=en.training_op_with_adadelta)
    run_experiment("optimizer_" + "adagrad", training_operation=en.training_op_with_adagrad)
    run_experiment("optimizer_" + "gd", training_operation=en.training_op_with_gd)


def save_output(name, accuracies, time, iteration=None):
    """
    Saves the output of an experiment to out.
    :param name: Name of the outputfile.
    :param accuracies: Array of accuracies in each epoch.
    :param time: Array of time measurements in each epoch.
    :param iteration: Amount of times this experiment has been run.
    :return:
    """
    extension = '.txt'
    if iteration is not None:
        extension = '_' + str(iteration) + extension
    out_path = definitions.EXPERIMENTS_CHAR_PATH + name + '/'
    try:
        os.makedirs(out_path)
    except OSError:
        pass  # dir already exists.
    savetxt(out_path + 'accuracy' + extension, accuracies)
    savetxt(out_path + 'time' + extension, time)


def experiment_net(n, learning_rate, batch_size, keep_prob, weight_decay):
    """
    A highly configurable function for training the neural network with all kinds of different parameters.
    All of these arguments have a default value and can be left out. Running this function without any arguments is
    equivalent to running the actual neural network as described in CharacterRecognition.py
    :param n: Amount of epochs to train the neural network.
    :param learning_rate: The learning rate of the neural network.
                          This is decremented over iterations when using the AdamOptimizer.
    :param batch_size: The batch size fed to the neural network.
    :param keep_prob: The probability of a connection between 2 perceptons being active in the fully connected layers.
    :param weight_decay: A regulation term to avoid overfitting. This indicates how much large weights should be penalized.
    :return: An array of the time passed and the accuracy at each iteration.
    """
    # Some variables required for training
    (x_train, y_train), (x_validation, y_validation) = character_recognition.get_data()
    _x, _y, h = character_recognition.create_neural_net(keep_prob=keep_prob)
    training_operation = character_recognition.create_training_operation(h, _y,
                                                                             learning_rate=learning_rate,
                                                                             decay=weight_decay)

    session = character_recognition.create_session()
    # Initialize variables of neural network
    session.run(tf.global_variables_initializer())
    accuracy = []
    time = []
    # Actual training
    num_train = len(x_train)
    print('Training:\n')
    start = t.time()
    for i in range(n):
        for offset in range(0, num_train, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            session.run(training_operation, feed_dict={_x: batch_x, _y: batch_y})
        validation_accuracy = session.run(character_recognition.get_accuracy(h, _y),
                                          feed_dict={_x: x_validation, _y: y_validation})
        accuracy.append(validation_accuracy)
        time.append(t.time() - start)
        if i % 10 == 0:
            print('EPOCH {}: Validation Accuracy = {:.3f}'.format(i, validation_accuracy))
    print("The training took: " + str(t.time() - start) + " seconds.")
    session.close()
    return accuracy, time


def run_experiment(name, iterations=ITERATIONS, start=0, n=EPOCHS, learning_rate=1e-4, batch_size=256,
                   keep_prob=0.5,
                   weight_decay=1e-3, training_operation=None):
    """
    :param name: Name of the experiment
    :param iterations: Amount of times to run the experiment
    :param start: Can be used to continue an experiment. This indicates the number of the first iteration.
    :return: None, the output is saved to EXPERIMENT_DATA_PATH
    """
    for i in range(iterations):
        with tf.variable_scope("Experiment_" + name + "_" + str(i)):
            print("Experiment: " + str(i))
            accuracy, time = experiment_net(n, learning_rate, batch_size, keep_prob, weight_decay)
            save_output(name, accuracy, time, start + i)

# character_preprocessing.augment_data(confirm=False)
# character_recognition.train_net(1000)
# -character_preprocessing.reset_data()
#compare_batch_size()
#compare_learning_rate()
#compare_keep_prob()
#compare_weight_decay()
compare_preprocessing_params()
