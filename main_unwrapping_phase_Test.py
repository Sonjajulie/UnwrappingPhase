import numpy as np

from kerasdeeplabv3plus.model import Deeplabv3
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
import os
from sklearn.model_selection import train_test_split

def main(cl_parser: Config, cl_config: dict):
    """
    * check whether  model should be trained or not
    * check whether train data path is given
    * call train class/method
        . call encoder
        . decoder
    * fit data
    * save network weights

    * read network weights
    * read input image
    * output
    """
    # load data
    # function to separate data into test and train data (0.66, 0.33)
    # train (0.9 train, 0.1 validation data)
    filepath_wrap = "/media/sonja/Elements/Projects/Unwrapping_phase/Training/add_noise_2/wrap/"
    filepath_true = "/media/sonja/Elements/Projects/Unwrapping_phase/Training/add_noise_2/k/"
    list_of_files_wrap = sorted([os.path.join(filepath_wrap, item) for item in os.listdir(filepath_wrap)
                          if os.path.isfile(os.path.join(filepath_wrap, item))])
    list_of_files_true = sorted([os.path.join(filepath_true, item) for item in os.listdir(filepath_true)
                          if os.path.isfile(os.path.join(filepath_true, item))])
    length_files = 10 #  len(list_of_files_true)
    dim_ = 256
    images_wrap = np.zeros((length_files, dim_, dim_))
    images_true = np.zeros((length_files, dim_, dim_))


    for i in range(length_files):  #length_files
        images_wrap[i] = np.loadtxt(list_of_files_wrap[i])
        images_true[i] = np.loadtxt(list_of_files_true[i])
    images_wrap = np.expand_dims(images_wrap, axis=3)
    images_true = np.expand_dims(images_true, axis=3)
    #
    X_train, X_test, y_train, y_test = train_test_split(images_wrap, images_true, test_size=0.33, random_state=0)
    logger.info("Initialize model...")
    # unwrapping_model = UnwrappingModel(X_train,y_train)
    deeplab_model = Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(256, 256, 1), classes=21, backbone='xception',
              OS=16, alpha=1., activation=None)
    #unwrapping_model = DeepLabV3(X_train,y_train)

if __name__ == '__main__':
    import logging.config
    parser = ClusteringParser()
    config = Config('logfile.log')
    logger = logging.getLogger(__name__)

    # read config log file from classes.Config
    logging.config.dictConfig(config.config_dict)
    logger.info("Start unwrapping program")
    main(parser, config.config_dict)
