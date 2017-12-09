import os

"""
File with global variables. 
This file mainly contains paths. 
"""


SIZE = 64
NUM_CHANNELS = 1
IMG_SHAPE = (SIZE, SIZE, NUM_CHANNELS)
SHAPE = (SIZE, SIZE)

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/') + '/'
DATA_PATH = PROJECT_PATH + 'Data/'
TXT_PATH = DATA_PATH + 'ascii/'

CHARSET_PATH = DATA_PATH + 'charset/'
PREPROCESSED_CHARSET_PATH = DATA_PATH + 'processed_charset/'

CHARSET_INFO_PATH = TXT_PATH + 'chars.txt'
PREPROCESSED_CHARSET_INFO_PATH = TXT_PATH + 'preprocessed_chars.txt'

MODELS_PATH = PROJECT_PATH + 'Models/'
MODEL_CHAR_RECOGNITION_PATH = MODELS_PATH + 'CharacterRecognition/Model/model.ckpt'
MODEL_CHAR_SEGMENTATION_PATH = MODELS_PATH + 'SplitPointDecision/Model/model.ckpt'

EXPERIMENTS_CHAR_PATH = PROJECT_PATH + 'Experiments/'
EXPERIMENTS_CHAR_GRAPHS_PATH = PROJECT_PATH + 'Graphs/'

WORDSET_INFO_PATH = TXT_PATH + 'words.txt'
WORDSET_PATH = DATA_PATH + 'words/'
WORDET_SPLITTING_PATH = DATA_PATH + 'segmentation/' + 'segmentation.txt'
