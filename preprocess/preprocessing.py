# __author__=   'Sargam Modak'

import cv2
import random
import numpy as np
from skimage.transform import resize


def resize_image(img, size):
    # print img.shape
    height, width, _ = img.shape
    new_width = size
    scale = new_width * 1. / width
    new_height = int(height * scale)

    if height < width:
        new_height = size
        scale = new_height * 1. / height
        new_width = int(width * scale)

    return resize(image=img,
                  output_shape=(new_height, new_width),
                  preserve_range=True)


def random_crop_image(img, size):
    height, width, _ = img.shape
    hoff = 0
    woff = 0

    if height != size:
        hoff = random.randint(a=0,
                              b=height-size-1)

    if width != size:
        woff = random.randint(a=0,
                              b=width-size-1)

    return img[hoff:hoff+size, woff:woff+size, :]


def change_brightness(img, scale):
    new_img = img + scale
    new_img[new_img > 255.] = 255.
    new_img[new_img < 0.] = 0.
    return new_img


def change_contrast(img, scale):
    new_img = (img - 128.) * scale + 128.
    new_img[new_img > 255.] = 255.
    new_img[new_img < 0.] = 0.
    return new_img


def rotate_image(img, angle):
    rotated_matrix = cv2.getRotationMatrix2D(center=img.shape[1::-1],
                                             angle=angle,
                                             scale=1.)
    img = cv2.warpAffine(src=img,
                         M=rotated_matrix,
                         dsize=img.shape[1::-1])
    return img


def translate_image(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(src=img,
                         M=M,
                         dsize=img.shape[1::-1])
    return img


def blur_image(img, ksize):
    img = cv2.blur(src=img, ksize=ksize)
    return img


def add_noise(img):
    noise = np.random.normal(loc=0.5, scale=1.0, size=img.shape)
    img+=noise
    return img
