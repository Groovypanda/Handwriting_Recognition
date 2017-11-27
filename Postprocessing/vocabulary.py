import nltk
import random
from difflib import get_close_matches
from difflib import SequenceMatcher
from CharacterRecognition import utils

nltk.download('words')

word_list = nltk.corpus.words.words()
length = len(word_list)


# Returns the most likely words given a certain word in the dictionary
def correct_written_words(word, amount=3):
    close_matches = []
    for match in get_close_matches(word.lower(), word_list, n=amount):
        score = SequenceMatcher(None, word, match).ratio()
        close_matches.append((match, score))
    return close_matches


# Returns array of indices and value of 'amount' max values in list.
def max_x(list, amount):
    copy = list[:]
    maxs = []
    for i in range(amount):
        max_value = max(copy)
        max_index = copy.index(max_value)
        del copy[max_index]
        maxs.append((max_index, max_value))
    return maxs


# cls_pred_list:  list of list with probabilities for each character (62 classes) per character in the word.
# Returns a list of pairs with words and their probabilities
def possible_written_characters(cls_pred_list, branching_factor=3):
    cls_preds = max_x((cls_pred_list[0]), branching_factor)
    possibilities = []
    for cls_pred_i, cls_pred_p in cls_preds:  # Go trough most likely characters (index and probability)
        if len(cls_pred_list) > 1:
            for word, probability in possible_written_characters(cls_pred_list[1:], branching_factor):
                possibilities.append((utils.cls2str(1 + cls_pred_i) + word, cls_pred_p * probability))
        else:
            possibilities.append((utils.cls2str(1 + cls_pred_i), cls_pred_p))
    return possibilities


def most_likely_words(cls_pred_list):
    most_possible_characters = possible_written_characters(cls_pred_list)
    most_possible_characters.sort(key=lambda x: x[1])
    words = []
    for characters, probability in most_possible_characters[-3:]:
        for word, score in correct_written_words(characters):
            words.append((word, probability * score))
    return sorted(words, key=lambda x: x[1], reverse=True)


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


# A   p   p   l   e
# 10 52 52  48  41
A_pred = rand_array([10])
p_pred = rand_array([51])
l_pred = rand_array([47])
e_pred = rand_array([40])

print(most_likely_words([A_pred, p_pred, p_pred, l_pred, e_pred]))
