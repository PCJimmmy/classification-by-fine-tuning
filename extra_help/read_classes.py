# __author__=   'Sargam Modak'

import os
import logging
import sys

# adding parent directory to system path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from config import load_config


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=__name__)


def read_classes(data_path=None, classes_file=None):

    if data_path is None:
        raise Exception("Path cannot be left None.")

    if classes_file is None:
        raise Exception("Classes file cannot be left None.")

    if not os.path.exists(data_path):
        raise Exception("{} data path does not exist.".format(data_path))

    dirs = os.listdir(data_path)
    with open(classes_file, 'w') as cls_file:
        for val in dirs:
            cls_file.writelines(val+'\n')
            
            
if __name__ == '__main__':
    cfg = load_config()
    data_path = cfg['data_path']['val']
    classes_file = cfg['classes_file']['val']

    read_classes(data_path=data_path,
                 classes_file=classes_file)

    logger.info("Classes file created.")
