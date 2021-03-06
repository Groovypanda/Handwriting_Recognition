import random
from difflib import SequenceMatcher
from difflib import get_close_matches

import nltk

import character_utils

nltk.download('words')

word_list = nltk.corpus.words.words()
length = len(word_list)


# Returns the most likely words given a certain word in the dictionary
def correct_written_words(word, amount=3):
    close_matches = []
    for match in get_close_matches(word.lower(), word_list, n=amount):
        score = SequenceMatcher(None, word, match).ratio()
        close_matches.append((match, score))
    if len(close_matches) == 0:
        # Fail safe mechanism. If a word does not exist in the vocabulary, we still have to return the possible words.
        return [(word, 0)]
    else:
        return close_matches


def possible_written_characters(char_probabilities, branching_factor=3):
    """
    We find the most likely words based purely on character probabilities. This making a tree of possible words.
    The depth indicates the length of the word. The leafs are full words. Each node contains a pair with the word and its
    probability.
    :param char_probabilities:  list of list with probabilities for each character (62 classes) per character in the word.
    :param branching_factor:  Decides how many possibilities to check for each character
    :return: a list of pairs with words and their probabilities
    """

    possibilities = []
    for char, cls_pred_p in char_probabilities[0]:  # Go trough most likely characters (index and probability)
        if len(char_probabilities) > 1:
            for word, probability in possible_written_characters(char_probabilities[1:], branching_factor):
                possibilities.append((char + word, cls_pred_p * probability))
        else:
            possibilities.append((char, cls_pred_p))
    return possibilities


def most_likely_words(char_probabilities):
    """
    Finds the most likely words given the class probabilities using a vocabularium.
    :param char_probabilities: A list of tuples with characters and their probabilities for every character in the word
    :return: A dictionary with words as keys and probabilities as values.
    """
    alpha = 0.7  # Importance of original word
    beta = 0.3  # Importance of word for voc
    most_possible_characters = possible_written_characters(char_probabilities)
    words = {}
    for characters, probability in most_possible_characters[:3]:
        correctly_written_words = correct_written_words(characters)
        if len(correctly_written_words) > 0:
            for word, score in correctly_written_words:
                words[word] = alpha * probability + beta * (score) # Smoothing technique, is score is 0 algorithm won't fail
        else:
            words[characters] = probability * 0.01
    return words


def most_likely_words_without_voc(char_probabilities):
    """
        Finds the most likely words given the class probabilities without a vocabularium.
        :param char_probabilities: A list of tuples with characters and their probabilities for every character in the word
        :return: A dictionary with words as keys and probabilities as values.
        """
    alpha = 0.7  # Importance of original word
    beta = 0.3  # Importance of word for voc
    most_possible_characters = possible_written_characters(char_probabilities)
    return most_possible_characters
'''
FOLLOWING FUNCTIONS ARE FOR TESTING PURPOSES
'''


def zeros(n):
    list = []
    for i in range(n):
        list.append(0)
    return list


def rand_array(index):
    array = zeros(62)
    array[index[0]] = 0.7
    indices = rand(index, 3)
    for i in indices:
        array[i] = 0.1
    return array


def rand(not_indices, amount):
    if amount > 0:
        x = random.randint(0, 61)
        if x in not_indices:
            return rand(not_indices, amount)
        else:
            not_indices.append(x)
            rands = rand(not_indices, amount - 1)
            rands.append(x)
            return rands
    else:
        return []

def test():
    print(correct_written_words("artificol"))
    print(correct_written_words("inteligance"))