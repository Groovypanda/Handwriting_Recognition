import CharacterRecognition.character_recognition as cr
import Tokenization.character_normalizer as normalizer

def evaluate_character_combinations(character_images, session_args):

    (session, _x, _y, h) = session_args

    return_characters = list()

    for character_image in character_images:
        return_characters.append(normalizer.normalize_character([character_image]))

    #character_probabilities = cr.imgs_to_prob_list(character_images, session, _x, _y, h)

    return return_characters
