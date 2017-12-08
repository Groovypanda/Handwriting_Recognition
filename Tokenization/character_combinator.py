import CharacterRecognition.character_recognition as crj
import Tokenization.character_normalizer as normalizer


def evaluate_character_combinations(character_images, session_args):
    nlist = list()
    for char_img in character_images:
        nlist.append(normalizer.normalize_character([char_img]))

    return nlist


def evaluate_character_combinations2(character_images, session_args):

    (session, _x, _y, h) = session_args

    lastIteration = [(index, normalizer.normalize_character([char_img])) for index, char_img in character_images]
    print(lastIteration)
    busy = True
    while busy:
        busy = False

        current_iteration = list()
        #zoek die met beste

        # WHITEBOARD VOOR REST



    character_probabilities = cr.imgs_to_prob_list(character_images, session, _x, _y, h)
    return return_characters
