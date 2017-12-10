import cv2
import definitions
import main
import character_extraction as chrext
import splitpoint_decision as toc
import character_recognition as cr
from character_preprocessing import augmentImage
from character_preprocessing import preprocess_image
from character_preprocessing import erodeImage
import numpy as np
import word_normalizer as wn
import vocabulary
import splitpoint_decision as sd

aug_demo = False
char_rec_demo = False
word_splitting_demo = True
word_rec_demo = True
data_creation_demo = False

def start_demo():
    char_file = definitions.PROJECT_PATH + 'Data/charset/Img/Sample056/img056-003.png'
    demo_chars = definitions.PROJECT_PATH + 'demo_data/char/{}.png'
    # word_file = definitions.PROJECT_PATH + 'Data/words/b04/b04-060/b04-060-00-07.png' # Government
    word_file = 'words/beer.png'
    word_image = main.read_image(word_file)
    char_image = main.read_image(char_file)
    tocsessionargs = toc.init_session()
    chrsessionargs = cr.init_session()

    if aug_demo:
        # input("Data augmentation demo")
        demo_augmentation(char_image)
    if char_rec_demo:
        # input("Character recognition demo")
        demo_char_recognition(demo_chars, chrsessionargs)
    if data_creation_demo:
        # input("Data creation demo")
        demo_data_creation()
    if word_splitting_demo:
        # input("Word splitting demo")
        demo_word_splitting(word_image, tocsessionargs)
    if word_rec_demo:
        # input("Word recognition demo")
        demo_word_recognition(word_image, chrsessionargs, tocsessionargs)


def resize_word(img):
    return cv2.resize(img, dsize=(800, 300))


def demo_data_creation():
    sd.start_data_creation()
    cv2.destroyAllWindows()


def demo_word_splitting(word_image, sessionargs):
    splitpoints = chrext.find_splits_img(word_image)
    cv2.imshow('Oversegmentation of word', resize_word(toc.show_splitpoints(word_image, splitpoints)))
    toc.show_splitpoints(word_image, splitpoints)
    splitpoint_decisions = toc.decide_splitpoints(word_image, splitpoints, sessionargs)
    splitpoints = [split for (is_split, split) in zip(splitpoint_decisions, splitpoints) if is_split]
    cv2.imshow('Splitpoints after neural network', resize_word(toc.show_splitpoints(word_image, splitpoints)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_augmentation(img):
    img = preprocess_image(img, inverse=True)
    add_noise = True
    add_rotations = True
    add_translations = True
    add_scales = True
    add_shearing = True
    images = augmentImage(img, add_noise=add_noise, add_rotations=add_rotations, add_translations=add_translations,
                          add_scales=add_scales, add_shearing=add_shearing)
    images.append(erodeImage(img))
    for augimg in images:
        cv2.imshow("Augmented image", cv2.resize(np.multiply(255, np.subtract(1, augimg)), dsize=(300, 300)))
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_char_recognition(demo_chars, sessionargs):
    for ordinal in range(ord('a'), ord('z')+1):
        char = chr(ordinal)
        img = main.read_image(demo_chars.format(char))
        cv2.imshow("Character recognition of the letter {}".format(char), img)
        print(cr.img_to_text(img, sessionargs, n=3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def demo_word_recognition(img, chrsessionargs, tocsessionargs):
    cv2.imshow("Word recognition", img)
    cv2.waitKey(0)
    normalized_word_image = wn.normalize_word(img)
    char_imgs = chrext.extract_characters(normalized_word_image, sessionargs=tocsessionargs, postprocess=False)
    char_probabilities = cr.imgs_to_text(char_imgs, chrsessionargs, n=3)
    print(max(vocabulary.possible_written_characters(char_probabilities), key=lambda x: x[1]))
    print("Without neural net: " + str(main.recognise_possible_words(img, chrsessionargs, tocsessionargs, postprocess=False)))
    print("With neural net: " + str(main.recognise_possible_words(img, chrsessionargs, tocsessionargs, postprocess=True, verbose=True)))
    cv2.destroyAllWindows()



if __name__ == "__main__":
    start_demo()
