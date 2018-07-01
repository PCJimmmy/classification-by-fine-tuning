# __author__=   'Sargam Modak'

import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from config import load_config


def create_callback(model_checkpoint=True, early_stopping=True, tensorboard=True):

    cfg = load_config()
    callbacks = []

    if model_checkpoint:
        model_id = cfg['model_id']['val']
        filepath = os.path.join('./util_files', model_id + '_model.hdf5')
        ckpt = ModelCheckpoint(filepath=filepath,
                               verbose=1,
                               save_best_only=True)
        callbacks.append(ckpt)

    if early_stopping:
        estopping = EarlyStopping(patience=5,
                                  verbose=1)
        callbacks.append(estopping)

    if tensorboard:
        tnsrbrd = TensorBoard(log_dir='./tensorboard_logs')
        callbacks.append(tnsrbrd)

    return callbacks
