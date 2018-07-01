# __author__=   'Sargam Modak'

import os
from random import shuffle
from sklearn.model_selection import train_test_split
import sys

# adding parent directory to system path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from config import load_config


def create_data(data_path, training_file_path, validation_file_path):

    assert os.path.exists(data_path), "'{}' path does not exist.".format(data_path)

    dirs = os.listdir(data_path)
    training_path_file = []
    validation_path_file = []

    for cur_dir in dirs:
        arr = os.listdir(os.path.join(data_path, cur_dir))
        training_paths, validation_paths = train_test_split(arr, train_size=0.8)

        for training_path in training_paths:
            training_path_file.append(cur_dir + '/' + training_path)

        for validation_path in validation_paths:
            validation_path_file.append(cur_dir + '/' + validation_path)

    # shuffling the data
    shuffle(training_path_file)
    shuffle(validation_path_file)

    with open(training_file_path, 'w') as f:
        for val in training_path_file:
            f.writelines(val+'\n')

    with open(validation_file_path, 'w') as f:
        for val in validation_path_file:
            f.writelines(val+'\n')


if __name__ == '__main__':

    cfg = load_config()
    data_path = cfg['data_path']['val']

    assert os.path.exists(data_path), "'{}' path does not exist.".format(data_path)

    training_file_path = cfg['training_images_path']['val']
    validation_file_path = cfg['validation_images_path']['val']

    create_data(data_path=data_path,
                training_file_path=training_file_path,
                validation_file_path=validation_file_path)
    print "Data Created Successfully."
