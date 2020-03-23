import numpy as np
from classes.UnwrappingModel import UnwrappingModel
from classes.DeepLabV3 import DeepLabV3
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
import os
import h5py
from sklearn.model_selection import train_test_split

def import_hdf5(filename):
	f = h5py.File(filename, 'r')
	a_group_key = list(f.keys())[0]
	data = list(f[a_group_key])
	return np.array(data)

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

    # where is data?
    filepath_wrap = "/home/jan/Desktop/UnwrappingPhase/Training/add_noise_2/wrap/"
    filepath_true = "/home/jan/Desktop/UnwrappingPhase/Training/add_noise_2/k/"




    list_of_files_wrap = sorted([os.path.join(filepath_wrap, item) for item in os.listdir(filepath_wrap)
                          if os.path.isfile(os.path.join(filepath_wrap, item))])
    list_of_files_true = sorted([os.path.join(filepath_true, item) for item in os.listdir(filepath_true)
                          if os.path.isfile(os.path.join(filepath_true, item))])
    length_files = len(list_of_files_true)  # len(list_of_files_true)
    n_xy = 256
    n_classes = 20
    images_wrap = np.zeros((length_files, n_xy, n_xy))
    images_true_2d = np.zeros((n_xy, n_xy))
    images_true = np.zeros((length_files, n_xy, n_xy, n_classes))


    for i in range(length_files):
        images_wrap[i] = import_hdf5(list_of_files_wrap[i])
        images_true_2d = import_hdf5(list_of_files_true[i])
        # convert 2d array in 3d array with classes
        k_offset = np.amin(images_true_2d)
        for x in range(n_xy):
            for y in range(n_xy):
                images_true[i, x, y, k_offset + images_true_2d[x, y]] = 1

    images_wrap = np.expand_dims(images_wrap, axis=3)
    #
    X_train, X_test, y_train, y_test = train_test_split(images_wrap, images_true, test_size=0.33, random_state=0)
    logger.info("Initialize model...")
    # unwrapping_model = UnwrappingModel(X_train,y_train)
    unwrapping_model = DeepLabV3(X_train,y_train, X_test, y_test)

if __name__ == '__main__':
    import logging.config
    parser = ClusteringParser()
    config = Config('logfile.log')
    logger = logging.getLogger(__name__)

    # read config log file from classes.Config
    logging.config.dictConfig(config.config_dict)
    logger.info("Start unwrapping program")
    main(parser, config.config_dict)
