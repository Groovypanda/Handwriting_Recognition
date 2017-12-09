from Tokenization.character_extraction_main import extract_characters
from Tokenization.word_extraction import preprocess_image
from Tokenization.character_combinator import evaluate_character_combinations
from Postprocessing.language_model import n_gram_model
import CharacterRecognition.character_recognition as chr
import Tokenization.oversegmentation_correction as toc
from Postprocessing.vocabulary import most_likely_words
import sys
import cv2


def read_image(file_name):
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)


def recognise_character(file_name):
    """
    Convert an image to a character.
    :param file_name: File name of the image
    :return: The character
    """
    return chr.img_to_text(read_image(file_name), chr.init_session())


def recognise_word(file_name):
    """
    Convert an image to a word.
    :param file_name: File name of the word
    :return: The word
    """
    return max(recognise_possible_words(read_image(file_name), chr.init_session(), toc.init_session()), key=lambda x: x[1])


def recognise_possible_words(img, sessionargs_char_recognition, sessionargs_oversegmentation_correction):
    """
    Algorithm:

    1) Split the sentence in images of characters.
    2) Convert every character to a list of probabilities. A list of these lists is named the cls_pred_list.
    3) Find the most likely words given the cls_pred_list. Now we have a list of words and their probabilities.
    (These probabilities are based on word distances with actual words in the English dictionary)

    :param sessionargs_char_recognition: Session and the neural network placeholders
    :param image: Image of a word
    :return: A list of pairs, the pairs consist of likely words and their probabilities.
    """
    char_imgs = extract_characters(img, sessionargs=sessionargs_oversegmentation_correction)

    # Call character_combinator

    evaluated_chars = evaluate_character_combinations(char_imgs, sessionargs_char_recognition)
    cls_pred_list = chr.imgs_to_prob_list(char_imgs, sessionargs_char_recognition)
    return most_likely_words(cls_pred_list)


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
        voc_words = recognise_possible_words(word, chr.init_session(), toc.init_session())
        # Find the most likely words using the language model
        lang_words = n_gram_model(text, voc_words.keys())
        most_likely_word = \
            max([(word, alpha * voc_words[word] + beta * prob) for word, prob in lang_words.items()],
                key=lambda x: x[0])[0]
        text.append(most_likely_word)
        print(most_likely_word)
    return ' '.join(text)


def main(argv):
    if len(argv) == 2:
        conf = argv[0]
        file_name = argv[1]
        if conf == '--character' or conf == '-c':
            print(recognise_character(file_name))
        elif conf == '--word' or conf == '-w':
            print(recognise_word(file_name))
        elif conf == '--text' or conf == '-t':
            print(recognise_text(file_name))
    else:
        print(
            '''
            recognise_text.py [options] [file]

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
