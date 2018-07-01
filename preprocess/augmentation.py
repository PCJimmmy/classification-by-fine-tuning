# __author__= 'Sargam Modak'

from preprocessing import *


def augment(img, horizontal_flip=False, vertical_flip=False, brightness=False, contrast=False, rotation=False,
            translation=False, blur=False, noise=False):

    if random.random() > 0.5:

        if horizontal_flip and random.random() > 0.5:
            img = cv2.flip(img, 1)

        if vertical_flip and random.random() > 0.5:
            img = cv2.flip(img, 0)

        if brightness and random.random() > 0.5:
            scale = random.randint(a=-10,
                                   b=10)
            img = change_brightness(img=img,
                                    scale=scale)

        if contrast and random.random() > 0.5:
            scale = random.randint(1, 2)
            img = change_contrast(img=img,
                                  scale=scale)

        if rotation and random.random() > 0.5:
            angle = random.randint(a=-30,
                                   b=30)
            img = rotate_image(img=img, angle=angle)

        if translation and random.random() > 0.5:
            x = random.randint(a=0,
                               b=50)
            y = random.randint(a=0,
                               b=50)
            img = translate_image(img=img,
                                  x=x,
                                  y=y)

        if blur and random.random() > 0.5:
            x = random.randint(a=1,
                               b=3)
            y = random.randint(a=1,
                               b=3)
            img = blur_image(img=img,
                             ksize=(x, y))

        if noise and random.random() > 0.5:
            img = add_noise(img)

    return img
