import numpy as np
import cv2  # used for resize
from matplotlib import pyplot as pylab
from classes.ClusteringParser import ClusteringParser
from classes.Config import Config
from classes.DeepLabV3 import DeepLabV3


def main(cl_parser: Config, cl_config: dict):
    # os.chdir('keras-deeplab-v3-plus-master') # go to keras-deeplab-v3-plusmaster

    deeplab_model = DeepLabV3(input_tensor=None, os=16, input_shape=(512, 512, 3), classes=21, alpha=1.,
                              activation=None, weights_path="https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/",
                              weightname="deeplabv3_xception_tf_dim_ordering_tf_kernels.h5", weights="given")

    pathIn = 'input'  # path for the input image
    pathOut = 'output'  # output path for the segmented image
    img = pylab.imread(pathIn + "/image1.jpg")
    w, h, _ = img.shape
    ratio = 512. / np.max([w, h])
    resized = cv2.resize(img, (int(ratio * h), int(ratio * w)))
    resized = resized / 127.5 - 1.
    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)), mode='constant')
    res = deeplab_model.model.predict(np.expand_dims(resized2, 0))
    labels = np.argmax(res.squeeze(), -1)
    pylab.figure(figsize=(20, 20))
    pylab.imshow(labels[:-pad_x], cmap='inferno'), pylab.axis('off'), pylab.colorbar()
    # pylab.show()
    pylab.savefig(pathOut + "/segmented.png", bbox_inches='tight', pad_inches=0)
    pylab.close()
    # os.chdir('..')

if __name__ == '__main__':
    import logging.config
    parser = ClusteringParser()
    config = Config('logfile.log')
    logger = logging.getLogger(__name__)

    # read config log file from classes.Config
    logging.config.dictConfig(config.config_dict)
    logger.info("Start unwrapping program")
    main(parser, config.config_dict)
