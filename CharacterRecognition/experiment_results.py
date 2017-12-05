import settings
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig

# from CharacterRecognition.experiments import ITERATIONS
experiments = ['base', 'batchsize', 'decay', 'filtersize', 'keepprob', 'learningrate', 'preprocess']

ITERATIONS = 4


def get_experiment_results(experiment_name):
    confs, accuracy_avgs, time_avgs = [], [], []
    for conf, dir in find_experiment_location(experiment_name):
        accuracy = []
        time = []
        for i in range(ITERATIONS):
            file_name = os.path.join(dir, '{}_' + str(i) + '.txt')
            accuracy.append(np.loadtxt(file_name.format('accuracy')))
            time.append(np.loadtxt(file_name.format('time')))
        confs.append(conf)
        accuracy_avgs.append(np.mean(accuracy, axis=0))
        time_avgs.append(np.mean(time, axis=0))
    return confs, accuracy_avgs, time_avgs


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
    if show:
        plt.show()
    if save and name != '' :
        savefig(os.path.join(settings.EXPERIMENTS_CHAR_PATH, name + '.png'))

def visualise_experiment(experiment_name):
    confs, accuracy_avgs, time_avgs = get_experiment_results(experiment_name)
    visualise(accuracy_avgs, time_avgs, name=experiment_name, labels=confs)

def visualise_experiment_configuration(experiment_name, conf):
    accuracies, times = get_configuration_results(experiment_name, conf)
    visualise(accuracies, times)



visualise_experiment_configuration("keepprob", "0.9")