
import word_extraction as word_extraction
import character_extraction_main as character_extraction_main
import character_normalizer as character_normalizer
import character_combinator as character_combinator
import cv2
import os

dir = os.path.dirname(__file__)

filepath = os.path.join(dir, '../data/texts/')

index1 = 0
for file in sorted(os.listdir(filepath)):

    img = cv2.imread(filepath + file, 0)

    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    words = word_extraction.preprocess_image(img, index1)


    index2 = 0

    for word in words:

        cv2.imshow("word", word)
        cv2.waitKey(0)

        characters = character_extraction_main.extract_characters(word, index2)

        #finalchars = character_combinator.evaluate_character_combinations(characters, None)

        for character in characters:
            cv2.imshow("character", character)
            cv2.waitKey(0)



        index2 += 1
    index1 += 1
