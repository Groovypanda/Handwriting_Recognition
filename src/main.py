

import cv2
import splitpoint_decision as toc
import  character_extraction as ce
from language_model import n_gram_model
from vocabulary import most_likely_words
from word_extraction import preprocess_image
import splitpoint_decision as sd
import character_recognition as cr
from character_preprocessing import augment_data
import word_normalizer as wn
import character_normalizer as cn

from pathlib import Path
import shutil
import definitions
import os
import sys

def read_image(file_name):
    if Path(file_name).exists():
        return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    else:
        raise FileNotFoundError(file_name)


def recognise_character(file_name):
    """
    Convert an image to a character.
    :param file_name: File name of the image
    :return: The character
    """
    return cr.img_to_text(read_image(file_name), cr.init_session())


def recognise_word(file_name):
    """
    Convert an image to a word.
    :param file_name: File name of the word
    :return: The word
    """
    tocsessionargs = toc.init_session()
    chrsessionargs = cr.init_session()
    return max(recognise_possible_words(read_image(file_name), chrsessionargs, tocsessionargs),
               key=lambda x: x[1])


def recognise_possible_words(img, sessionargs_char_recognition, sessionargs_oversegmentation_correction, postprocess=True, verbose=False):
    """
    Algorithm:

    1) Split the sentence in images of characters.
    2) Convert every character to a list of probabilities. A list of these lists is named the char_probabilities.
    3) Find the most likely words given the char_probabilities. Now we have a list of words and their probabilities.
    (These probabilities are based on word distances with actual words in the English dictionary)

    :param sessionargs_char_recognition: Session and the neural network placeholders
    :param image: Image of a word
    :return: A list of pairs, the pairs consist of likely words and their probabilities.
    """
    normalized_word_image = wn.normalize_word(img)
    char_imgs = ce.extract_characters(normalized_word_image, sessionargs=sessionargs_oversegmentation_correction, postprocess=postprocess)

    # normalized_characters = [cm.normalize_character(character) for character in char_imgs]
    # ccb.normalize_and_combine_characters(char_imgs, sessionargs_char_recognition)

    # Call character_combinator

    # evaluated_chars = evaluate_character_combinations(char_imgs, sessionargs_char_recognition)
    normalized_character_images = [cn.normalize_character(character_im) for character_im in char_imgs]
    char_probabilities = cr.imgs_to_text(normalized_character_images, sessionargs_char_recognition, n=3, verbose=verbose)
    return most_likely_words(char_probabilities)


def recognise_text(file_name):
    """
    Algorithm:

    1) Split the sentence in images of words.
    2) Convert the image of a word to a list of likely words (See word_img_to_most_likely_words)
    3) For every word: (language modelling step)
        3.1) Choose the most likely word in the given context. For this we use an n-gram model.

    :param images: Image of a sentence
    :return: Text of the sentence
    """
    alpha = 0.7  # Indicates importance of correct vocabulary
    beta = 0.3  # Indicates importance of language model
    words = preprocess_image(read_image(file_name))
    text = []  # Converted image into list of words
    for word in words:
        # Find a list of possible words using the vocabularium
        voc_words = recognise_possible_words(word, cr.init_session(), toc.init_session())
        # Find the most likely words using the language model
        lang_words = n_gram_model(text, voc_words.keys())
        most_likely_word = \
            max([(word, alpha * voc_words[word] + beta * prob) for word, prob in lang_words.items()],
                key=lambda x: x[0])[0]
        text.append(most_likely_word)
        print("Found word: " + most_likely_word[0])
    return ' '.join(text)


def main(argv):
    if len(argv) >= 1:
        option = argv[0]
        arg = argv[1] if len(argv) > 1 else None
        if option == '--character' or option == '-c':  # Recognise a character
            print(recognise_character(arg))
        elif option == '--word' or option == '-w':  # Recognise a word
            print(recognise_word(arg))
        elif option == '--text' or option == '-t':  # Recognise a text
            print(recognise_text(arg))
        elif option == '--train-rec' or option == '-tr':  # Train a character segmentation model for 'arg' epochs
            epochs = int(arg) if arg is not None else 500
            cr.train_net(epochs, min_save=0.77)
        elif option == '--train-split' or option == '-ts':  # Train a character recognition model for 'arg' epochs
            epochs = int(arg) if arg is not None else 250
            sd.train_net(epochs, min_save=0.71)
        elif option == '--create-data' or option == '-cd':  # Create new data for the character segmentation training.
            sd.start_data_creation(arg)
        elif option == '--augment-data' or option == '-ad':  # Augment the character dataset
            augment_data()
    else:
        print(
            '''
            main.py [options] [file]

            Project of Ruben Dedecker and Jarre Knockaert. The goal of this script is to convert images which consist
            of handwritten text into text. It can be called with the following options.

            -c, --character : Convert the given image into a character.
            -w, --word: Convert the given image into a word.
            -t, --text: Convert the given image into text. This can be a sentence, a paragraph, ...
            -h, --help: Prints this output.
            '''
        )

if __name__ == "__main__":
    main(sys.argv[1:])
