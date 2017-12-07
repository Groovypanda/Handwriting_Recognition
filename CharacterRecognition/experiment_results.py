import settings
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig

# from CharacterRecognition.experiments import ITERATIONS
experiments = ['base', 'batchsize', 'decay', 'filtersize', 'keepprob', 'learningrate', 'preprocess']

ITERATIONS = 4


def get_experiment_results(experiment_name, max=True):
    confs, accuracies, times = [], [], []
    for conf, dir in find_experiment_location(experiment_name):
        # Find the best result (result for which accuracy is highest in the end)
        avg_accuracy = []
        avg_time = []
        max_accuracy = [0]
        max_time = []
        for i in range(ITERATIONS):
            file_name = os.path.join(dir, '{}_' + str(i) + '.txt')
            accuracy = np.loadtxt(file_name.format('accuracy'))
            time = np.loadtxt(file_name.format('time'))
            if max:
                if accuracy[-1] > max_accuracy[-1]:
                    max_accuracy = accuracy
                    max_time = time
            else:
                avg_accuracy.append(accuracy)
                avg_time.append(time)
        confs.append(conf)
        if max:
            accuracies.append(max_accuracy)
            times.append(max_time)
        else:  # Calculate mean if not max
            accuracies.append(np.mean(avg_accuracy, axis=0))
            times.append(np.mean(avg_time, axis=0))
    return confs, accuracies, times


def get_configuration_results(experiment_name, conf):
    accuracies, times = [], []
    _, dir = find_experiment_conf_location(experiment_name, conf)
    for i in range(ITERATIONS):
        file_name = os.path.join(dir, '{}_' + str(i) + '.txt')
        accuracies.append(np.loadtxt(file_name.format('accuracy')))
        times.append(np.loadtxt(file_name.format('time')))
    return accuracies, times


def find_experiment_location(experiment_name):
    return [(x[0].split('/')[-1], x[0]) for x in os.walk(settings.EXPERIMENTS_CHAR_PATH) if experiment_name in x[0]]


def find_experiment_conf_location(experiment_name, conf):
    return find_experiment_location('_'.join([experiment_name, str(conf)]))[0]


def visualise(accuracies, times, name='', show=True, save=False, labels=None):
    plots = []
    for i in range(len(accuracies)):
        if labels[i]:
            print("{}: {:.0f}%, {:.0f}".format(labels[i], 100 * accuracies[i][-1], times[i][-1]))
        else:
            print(accuracies[i][-1], times[i][-1])
        if labels is None:
            plot, = plt.plot(times[i], accuracies[i])
        else:
            plot, = plt.plot(times[i], accuracies[i], label=labels[i])
        plots.append(plot)
    if labels:
        plt.legend(handles=plots)
    plt.ylabel("Accuracy")
    plt.xlabel("Time (s)")
    axes = plt.gca()
    axes.set_ylim([0, 0.9])
    if save and name != '':
        savefig(os.path.join(settings.EXPERIMENTS_CHAR_GRAPHS_PATH, name + '.png'))
    if show:
        plt.show()


def visualise_experiment(experiment_name, save=False, max=True):
    confs, accuracy_avgs, time_avgs = get_experiment_results(experiment_name, max)
    visualise(accuracy_avgs, time_avgs, name=experiment_name, labels=confs, save=save)


def visualise_experiment_configuration(experiment_name, conf, save=False):
    accuracies, times = get_configuration_results(experiment_name, conf)
    visualise(accuracies, times, save=save, name='_'.join([experiment_name, conf]))


for exp in experiments:
    visualise_experiment(exp, max=True, save=True)
# visualise_experiment_configuration("preprocess", conf="all", save=True)
