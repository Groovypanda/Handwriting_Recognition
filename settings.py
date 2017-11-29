import os

SIZE = 32
NUM_CHANNELS = 1
IMG_SHAPE = (SIZE, SIZE, NUM_CHANNELS)
SHAPE = (SIZE, SIZE)

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + '/'
DATA_PATH = PROJECT_PATH + 'Data/'
TXT_PATH = DATA_PATH + 'ascii/'

CHAR_DATA_PATH = DATA_PATH + 'charset/'
PREPROCESSED_CHARS_PATH = DATA_PATH + 'processed_charset/'

CHAR_DATA_TXT_PATH = TXT_PATH + 'chars.txt'
PREPROCESSED_CHAR_DATA_TXT_PATH = TXT_PATH + 'preprocessed_chars.txt'

SAVE_PATH = PROJECT_PATH + 'CharacterRecognition/Model/model.ckpt'
