---
Unwrapping phase
---

This repository attempts to reproduce the results of the paper:
"Rapid and robust two-dimensional phase unwrapping via deep learning"
written by Zhang et al. (2019).

###  Create training dataset:

Traning dataset is created by using the mathematica script "Create_training_dataset.nb". At the moment, we implemented additive noise only.
We use the same approach as explained in the paper Zhang et al. (2019). 
In "pthout" one can specify the path, where all training files are saved. Three different folder will be created in this path: k, wrap and unwrap
and all dat-files will be saved in the folders respectively.
The parameter "nDatasets" specifies the number of files. Currently 2000 files will be created.


###  Create model:
With Keras create in on my own!


Is this the correct one?
https://github.com/tensorflow/models/tree/master/research/deeplab

According to the blog:
http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/
1.) Fork the Project-DeepLab: https://github.com/tensorflow/models/tree/master/official
2.) Install and run https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md
3.) DeepLab on Cityscapes: finish running deeplab on Cityscapes: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md
Good explanations for atrous convolutions:
https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74
https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
https://medium.com/coinmonks/review-sppnet-1st-runner-up-object-detection-2nd-runner-up-image-classification-in-ilsvrc-906da3753679


https://www.analyticsvidhya.com/blog/2019/02/tutorial-semantic-segmentation-google-deeplab/
