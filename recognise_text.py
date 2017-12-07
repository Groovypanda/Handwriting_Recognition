import Postprocessing.process as pro
import CharacterRecognition.character_recognition as c
import CharacterRecognition.preprocessing as pre
import sys


def recognise_character(file_name):
    return pro.sentence_img_to_text(pro.read_image(file_name))


def recognise_word(file_name):
    return pro.word_img_to_most_likely_words(pro.read_image(file_name))


def recognise_text(file_name):
    return c.img_to_text(pre.read_image(file_name))


def main(argv):
    if (len(argv) >= 2):
        conf = argv[0]
        file_name = argv[1]

        if conf == '--character' or '-c':
            print(recognise_character(file_name))
            return
        elif conf == '--word' or '-w':
            print(recognise_word(file_name))
            return
        elif conf == '--text' or '-t':
            print(recognise_text(file_name))
            return

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
    main(sys.argv[1:])
