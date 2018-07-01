# __author__=   'Sargam Modak'

import os
import json
import logging
import numpy as np
from skimage.io import imread
from keras.utils import to_categorical
from keras.engine.training import _make_batches

from config import load_config
from augmentation import augment
from preprocessing import resize_image, random_crop_image


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)

class Generator:

    def __init__(self, training=True):

        cfg = load_config()
        self.training = training

        if training:
            images_path_file = cfg['training_images_path']['val']
        else:
            images_path_file = cfg['validation_images_path']['val']

        assert os.path.exists(images_path_file), '{} file does not exist. Make sure training and validation file has ' \
                                                 'been created'.format(images_path_file)
        image_paths = []
        with open(images_path_file) as f:
            image_paths.append(f.readlines())
        self.image_paths = [img_path[:-1] for img_path in image_paths[0]]
        self.total_images = len(self.image_paths)
        self.image_paths = np.array(self.image_paths)

        self.batch_size = cfg['batch_size']['val']
        self.input_size = cfg['img_size']['val']
        classes_file = cfg['classes_file']['val']

        assert os.path.exists(classes_file), "{} file does not exist or path is wrong".format(classes_file)

        self.classes = []
        with open(classes_file) as cls_file:
            self.classes.append(cls_file.readlines())
        self.classes = [cls[:-1] for cls in self.classes[0]]
        self.total_classes = len(self.classes)

        self.data_path = cfg['data_path']['val']
        cls2id_file = cfg['cls2id']['val']
        id2cls_file = cfg['id2cls']['val']

        if training:
            self.cls2id = {}
            self.id2cls = {}
            for idx, cls in enumerate(self.classes):
                self.cls2id[cls] = idx
                self.id2cls[idx] = cls
            json.dump(obj=self.cls2id,
                      fp=open(cls2id_file, 'w'))
            json.dump(obj=self.id2cls,
                      fp=open(id2cls_file, 'w'))
        else:
            self.cls2id = json.load(open(cls2id_file))
            self.id2cls = json.load(open(id2cls_file))

    def generator(self):

        while True:
            batches = _make_batches(size=self.total_images,
                                    batch_size=self.batch_size)
            for start, end in batches:
                arr = []
                labels = []
                cur_batch = self.image_paths[start:end]

                for image_path in cur_batch:
                    # print image_path
                    img = imread(fname=os.path.join(self.data_path, image_path))

                    # if channels are not 3
                    ndim = len(img.shape)

                    if ndim == 2:
                        img = img[..., np.newaxis]
                        img = np.tile(A=img,
                                      reps=(1, 1, 3))

                    if ndim == 4:
                        img = img[..., :3]

                    # resizing image maintaining aspect ratio
                    img = resize_image(img=img,
                                       size=self.input_size)

                    if self.training:
                        # random cropping while training
                        img = random_crop_image(img=img,
                                                size=self.input_size)
                        img = augment(img=img,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      brightness=True,
                                      contrast=True,
                                      rotation=True,
                                      translation=True,
                                      blur=True,
                                      noise=True)
                    else:
                        # center cropping
                        h, w, c = img.shape
                        center_h = h / 2
                        center_w = w / 2
                        center_new_img = self.input_size/2
                        new_x1 = center_w - center_new_img
                        new_y1 = center_h - center_new_img
                        new_x2 = center_w + center_new_img
                        new_y2 = center_h + center_new_img
                        if self.input_size % 2 == 1:
                            new_x2 += 1
                            new_y2 += 1
                        img = img[new_y1:new_y2, new_x1:new_x2]

                    arr.append(img)
                    cls = image_path.split('/')[0]
                    id_for_cls = self.cls2id[cls]
                    labels.append(id_for_cls)
                    
                arr = np.array(arr)
                arr.astype('float32')

                # making mean of data 0 with standard deviation 1
                arr /= 255.
                arr -= 0.5
                arr *= 2.

                # one hot encoding
                labels = to_categorical(y=labels,
                                        num_classes=self.total_classes)
                yield (arr, labels)
