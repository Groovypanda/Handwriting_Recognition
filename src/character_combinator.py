import character_normalizer as normalizer
import character_recognition as cr
import character_utils
import cv2

def normalize_and_combine_characters(split_characters_images, sessionargs_char_recognition):
    split_characters_list = list()

    max_iterations = min(8, len(split_characters_images))
    identifier = 1;

    for character_image in split_characters_images:
        base_percentage, max_percentage, identiefier_list = 0, 0, list()
        split_characters_list.append( (character_image, base_percentage, identiefier_list, max_percentage) )

    print (len (split_characters_list))
    for iteration_length in range(1, max_iterations+1):
        print("##########################")
        print(iteration_length)
        for startindex in range(0, len(split_characters_list) - iteration_length + 1):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(startindex)

            combined_character_splices = split_characters_images[startindex:startindex + iteration_length]

            normalized_combined_character_splices = normalizer.normalize_character(combined_character_splices)
            print("normalized")




            print ("CHARACTER")
            probabilities = cr.img_to_prob(normalized_combined_character_splices, sessionargs_char_recognition)
            print(cr.most_probable_chars(probabilities, 5))

            #cv2.imshow("c2", normalized_combined_character_splices)
            #cv2.waitKey(0)

    return list()
