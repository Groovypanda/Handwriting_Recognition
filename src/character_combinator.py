import character_normalizer as normalizer
import character_recognition as cr
import character_utils


def normalize_and_combine_characters(split_characters_images, sessionargs_char_recognition):
    split_characters_list = list()

    max_iterations = min(8, len(split_characters_images))
    identifier = 1;

    for character_image in split_characters_images:
        base_percentage, max_percentage, identiefier_list = 0, 0, list()
        split_characters_list.append( (character_image, base_percentage, identiefier_list, max_percentage) )

    for iteration_length in range(1, max_iterations):
        print("##########################")
        print(iteration_length)
        for startindex in range(len(split_characters_list) - 1 - iteration_length):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(startindex)

            combined_character_splices = split_characters_images[startindex:startindex + iteration_length]
            normalized_combined_character_splices = normalizer.normalize_character(combined_character_splices)

            for char in normalized_combined_character_splices:
                print ("CHARACTER")
                probabilities = cr.img_to_prob(char, sessionargs_char_recognition)
                #print(cr.most_probable_chars(probabilities, 5))
                #print(sorted([(character_utils.cls2str(i),x) for (i,x) in enumerate(probabilities)], key=lambda x:-x[1])[:5])

            identifier += 1;


    return list()
