# __author__=   'Sargam Modak'

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation


def create_model(input_shape, nclasses, fine_tune=True):
    inception_model = InceptionV3(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_shape, input_shape, 3),
                                  classes=nclasses)
    
    for layer in inception_model.layers:
        # unfreeze last block
        if fine_tune and layer.name == 'conv2d_90':
            break
        layer.trainable = False
    
    out = inception_model.output
    out = Conv2D(filters=nclasses,
                 kernel_size=(1, 1),
                 padding='SAME',
                 activation='tanh')(out)
    out = GlobalAveragePooling2D()(out)
    out = Activation('softmax')(out)
    
    model = Model(inputs=inception_model.input,
                  outputs=out)
    
    return model
