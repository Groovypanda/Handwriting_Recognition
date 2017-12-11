import character_normalizer as normalizer
import character_recognition as cr
import character_utils
import cv2


UPPER_ERROR_MARGIN = 0.85
MAX_WIDTH_HEIGHT_RATIO = 3
def normalize_and_combine_characters(split_characters_images, sessionargs_char_recognition):
    split_characters_list = list()

    max_iterations = min(8, len(split_characters_images))
    identifier = 1;

    for character_image in split_characters_images:
        split_characters_list.append( (identifier, 0) )
        identifier += 1

    for iteration_length in range(1, max_iterations+1):
        for startindex in range(0, len(split_characters_list) - iteration_length + 1):

            combined_character_splices = split_characters_images[startindex:startindex + iteration_length]

            combined_character = normalizer.combine_characters(combined_character_splices)
            height, width = combined_character.shape[:2]

            normalized_combined_character_splices = normalizer.normalize_character(combined_character)

            probabilities = cr.img_to_prob(normalized_combined_character_splices, sessionargs_char_recognition)
            print(cr.most_probable_chars(probabilities, 5))

            cv2.imshow("c2", normalized_combined_character_splices)
            cv2.waitKey(0)


            #########################################################
            # RESULTS ARE UNFAVORABLE BECAUSE OF                    #
            # VERY HIGH CERTAINTY OF ERRONOUS RECOGNISED characters #
            # WHEN MULTIPLE NORMALIZED CHARACTERS ARE FED           #
            #########################################################

    return list()
