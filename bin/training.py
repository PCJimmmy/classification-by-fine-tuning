# __author__=   'Sargam Modak'

import tensorflow as tf
import os
from keras.optimizers import Adamax
import logging
import sys

# to add parent directory path in system path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from config import load_config
from preprocess import Generator
from models import create_model, create_callback

# to stop program from using full gpu resources
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def training(input_shape, nclasses, epochs, batch_size, training_images_path, validation_images_path):
    
    assert os.path.exists(training_images_path), '{} file does not exists.'.format(training_images_path)
    assert os.path.exists(validation_images_path), '{} file does not exists.'.format(validation_images_path)

    training_images_arr = []
    validation_images_arr = []

    with open(training_images_path) as train_f:
        training_images_arr.append(train_f.readlines())
    training_images_arr = training_images_arr[0]
    logger.info(msg="loaded training images.")

    with open(validation_images_path) as valid_f:
        validation_images_arr.append(valid_f.readlines())
    validation_images_arr = validation_images_arr[0]
    logger.info(msg="loaded validation images.")

    train_steps_per_epoch = len(training_images_arr) / batch_size
    validation_steps_per_epoch = len(validation_images_arr) / batch_size
    
    model = create_model(input_shape=input_shape,
                         nclasses=nclasses,
                         fine_tune=True)
    logger.info("Model created.")

    model_id = cfg['model_id']['val']
    model_path = os.path.join('./util_files', model_id+'_model.hdf5')
    if os.path.exists(model_path):
        model.load_weights(model_path)
        logger.info("loaded weights successfully.")

    # print model.summary()
    
    model.compile(optimizer=Adamax(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    training_generator = Generator()
    validation_generator = Generator(training=False)

    logger.info('training started..')
    model.fit_generator(generator=training_generator.generator(),
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=1,
                        callbacks=create_callback(),
                        validation_data=validation_generator.generator(),
                        validation_steps=validation_steps_per_epoch,
                        workers=8,
                        use_multiprocessing=True,
                        max_queue_size=20)
    
    logger.info(msg="Training completed successfully.")


if __name__ == '__main__':
    cfg = load_config()
    input_shape = cfg['img_size']['val']
    classes_file = cfg['classes_file']['val']

    assert os.path.exists(classes_file), '{} file does not exists.'.format(classes_file)

    classes = []
    with open(classes_file) as f:
        classes.append(f.readlines())
    classes = [cls[:-1] for cls in classes[0]]

    total_classes = len(classes)
    batch_size = cfg['batch_size']['val']
    epochs = cfg['epochs']['val']

    training_images_path = cfg['training_images_path']['val']
    validation_images_path = cfg['validation_images_path']['val']

    training(input_shape=input_shape,
             nclasses=total_classes,
             epochs=epochs,
             batch_size=batch_size,
             training_images_path=training_images_path,
             validation_images_path=validation_images_path)
    logger.info(msg='Exitting program.')
