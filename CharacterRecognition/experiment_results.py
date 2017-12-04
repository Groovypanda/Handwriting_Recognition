import settings
import os
experiments = ['base', 'batchsize', 'decay', 'filtersize', 'keepprob', 'learningrate', 'preprocess']


def get_experiment_results(experiment_name):
    """
    Finds the results of an experiment. 
    :param experiment_name: The name of the experiment
    :return: A list of tuples, each tuple contains a configuration (string) and the results (4 lists).
    """

    for root, dirs, files in os.walk(settings.EXPERIMENTS_CHAR_PATH):
        print(root, dirs, files)

get_experiment_results("base")