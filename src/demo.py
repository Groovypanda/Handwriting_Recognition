import cv2
import definitions
import main
import character_extraction as chrext
import splitpoint_decision as toc
import character_recognition as cr
from character_preprocessing import augmentImage
from character_preprocessing import preprocess_image
from character_preprocessing import erodeImage
from character_utils import index2str
import numpy as np

aug_demo = False
char_rec_demo = True
word_splitting_demo = False


def start_demo():
    char_file = definitions.PROJECT_PATH + 'Data/charset/Img/Sample056/img056-003.png'
    word_file = definitions.PROJECT_PATH + 'Data/words/d01/d01-104/d01-104-00-02.png'
    word_image = main.read_image(word_file)
    char_image = main.read_image(char_file)
    tocsessionargs = toc.init_session()
    chrsessionargs = cr.init_session()

    if aug_demo:
        input("Data augmentation demo")
        demo_augmentation(char_image)
    if char_rec_demo:
        input("Character recognition demo")
        print(cr.img_to_text(char_image, chrsessionargs, n=3))
    if word_splitting_demo:
        input("Word splitting demo")
        demo_word_splitting(word_image, tocsessionargs)


def resize_word(img):
    return cv2.resize(img, dsize=(800, 300))


def demo_word_splitting(word_image, sessionargs):
    splitpoints = chrext.find_splits_img(word_image)
    cv2.imshow('Oversegmentation of word', resize_word(toc.show_splitpoints(word_image, splitpoints)))
    toc.show_splitpoints(word_image, splitpoints)
    splitpoint_decisions = toc.decide_splitpoints(word_image, splitpoints, sessionargs)
    splitpoints = [split for (is_split, split) in zip(splitpoint_decisions, splitpoints) if is_split]
    cv2.imshow('Splitpoints after neural network', resize_word(toc.show_splitpoints(word_image, splitpoints)))
    cv2.waitKey(0)


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


if __name__ == "__main__":
    start_demo()
