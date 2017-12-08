import cv2

import CharacterRecognition.character_recognition as cr
from Postprocessing.language_model import n_gram_model
from Postprocessing.vocabulary import most_likely_words
from Tokenization.character_combinator import evaluate_character_combinations
from Tokenization.character_extraction_main import extract_characters
from Tokenization.word_extraction import preprocess_image

'''
Postprocessing consists of 2 steps:
1) Find the most likely word given a list of probabilities for every character in the word.
   This list contains the probability for a character (image) to be equal to that given character.
   We use an English dictionary to find possible likely words.
2) Find the most likely word in a certain context.
   Given a list of likely words (see previous step), we can find the most likely word based on the context.
   We use Markov models, n-gram models in particular to solve this problem.

These 2 techniques are often used for voice recognition and other techniques, but can be applied for this project aswell
to handle the high error rate of character recognition (+/- 20%).
'''


def read_image(file_name):
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)


def predict_actual_word(cls_pred_list):
    words = most_likely_words(cls_pred_list)
    print(words)


def sentence_img_to_text(image):
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
    words = preprocess_image(image)
    text = []  # Converted image into list of words
    # Network for recognising individual characters.
    # These variables need to be this high up in the python project.
    # In order to avoid initialising multiple sessions or neural networks.
    session, _x, _y, h = cr.init_session()
    for word in words:
        # Find a list of possible words using the vocabularium
        voc_words = word_img_to_most_likely_words(word, session, _x, _y, h)
        # Find the most likely words using the language model
        lang_words = n_gram_model(text, voc_words.keys())
        most_likely_word = \
        max([(word, alpha * voc_words[word] + beta * prob) for word, prob in lang_words.items()], key=lambda x: x[0])[0]
        text.append(most_likely_word)
        print(most_likely_word)
    return ' '.join(text)


def word_img_to_most_likely_words(image, session=None, _x=None, _y=None, h=None):
    """
    Algorithm:

    1) Split the sentence in images of characters.
    2) Convert every character to a list of probabilities. A list of these lists is named the cls_pred_list.
    3) Find the most likely words given the cls_pred_list. Now we have a list of words and their probabilities.
    (These probabilities are based on word distances with actual words in the English dictionary)

    :param image: Image of a word
    :return: A list of pairs, the pairs consist of likely words and their probabilities.
    """
    char_imgs = extract_characters(image)
    if session is None:
        session, _x, _y, h = cr.init_session()

    #Call character_combinator

    sessionargs = (session, _x, _y, h)

    evaluated_chars = evaluate_character_combinations(char_imgs, sessionargs)

    cls_pred_list = cr.imgs_to_prob_list(char_imgs, session, _x, _y, h)
    return most_likely_words(cls_pred_list)
