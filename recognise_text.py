from Tokenization.character_extraction_main import extract_characters
from Tokenization.word_extraction import preprocess_image
from Tokenization.character_combinator import evaluate_character_combinations
from Postprocessing.language_model import n_gram_model
from CharacterRecognition.character_recognition import init_session
from CharacterRecognition.character_recognition import imgs_to_prob_list
from CharacterRecognition.character_recognition import img_to_text
from Postprocessing.vocabulary import most_likely_words
import sys
import cv2
from pathlib import Path

def read_image(file_name):
    if Path(file_name).is_file():
        return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    else:
        raise FileNotFoundError


def recognise_character(file_name):
    """
    Convert an image to a character.
    :param file_name: File name of the image
    :return: The character
    """
    return img_to_text(read_image(file_name), init_session())


def recognise_word(file_name):
    """
    Convert an image to a word.
    :param file_name: File name of the word
    :return: The word
    """
    return max(recognise_possible_words(read_image(file_name), init_session()), key=lambda x: x[1])


def recognise_possible_words(img, sessionargs):
    """
    Algorithm:

    1) Split the sentence in images of characters.
    2) Convert every character to a list of probabilities. A list of these lists is named the cls_pred_list.
    3) Find the most likely words given the cls_pred_list. Now we have a list of words and their probabilities.
    (These probabilities are based on word distances with actual words in the English dictionary)

    :param sessionargs: Session and the neural network placeholders
    :param image: Image of a word
    :return: A list of pairs, the pairs consist of likely words and their probabilities.
    """
    char_imgs = extract_characters(img)
    # Call character_combinator

    evaluated_chars = evaluate_character_combinations(char_imgs, sessionargs)
    cls_pred_list = imgs_to_prob_list(char_imgs, sessionargs)
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
        voc_words = recognise_possible_words(word, init_session())
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
            print("Hello, my name is Ronald, let me take a quick look at that character you have there.")
            print(recognise_character(file_name))
        elif conf == '--word' or conf == '-w':
            print("Hello, my name is Jane, and I'll assist you today in reading words.")
            print(recognise_word(file_name))
        elif conf == '--text' or conf == '-t':
            print("Hello, my name is Gerald, shall I read this text for you?.")
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
    print(recognise_word("Data/words/a01/a01-000u/a01-000u-02-03.png"))
    #main(sys.argv[1:])
