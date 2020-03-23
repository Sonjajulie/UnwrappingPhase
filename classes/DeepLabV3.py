# import numpy as np

# Tensor flow libraries
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Concatenate, Input, DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D, GlobalAveragePooling2D
# from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.models import load_model, SeparableConv2D, UpSampling2D,
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Lambda, Dropout

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils.data_utils import get_file
import os

class DeepLabV3:
    def __init__(self, k_data=None, k_data_labels=None, k_data_test=None, k_data_labels_test=None, input_tensor=None, os=16, input_shape=(256, 256, 1), classes=20, alpha=1., activation=None,
                 weights_path=None, weightname=None, weights=None):
        """ Instantiates the Deeplabv3+ architecture
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc),
                'cityscapes' (pre-trained on cityscape) or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
            classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
                If number of classes not aligned with the weights used, last layer is initialized randomly
            backbone: backbone to use. one of {'xception','mobilenetv2'}
            activation: optional activation to add to the top of the network.
                One of 'softmax', 'sigmoid' or None
            alpha: controls the width of the MobileNetV2 network. This is known as the
                width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`
        """
        self.activation = activation
        self.alpha = alpha
        self.atrous_rates = (6, 12, 18)
        self.b0 = None
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.classes = classes
        self.dec_skip1 = None
        self.entry_block3_stride = 2
        self.exit_block_rates = (1, 2)
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.img_input = None
        self.inputs = None
        self.last_layer_name = 'custom_logits_semantic'
        self.middle_block_rate = 1
        self.model = None
        self.os = os
        self.shape_before = None
        self.size_before = None
        self.size_before2 = None
        self.size_before3 = None
        self.skip1 = None
        self.weights_path = weights_path
        self.weights = weights
        self.weightname = weightname
        self.x = None

        # initialize deeplab3v+ NN - architecture
        if k_data is not None:
            k_shape = k_data.shape[1:]

        # create keras - model
        self.model_architecture()
        self.model = Model(self.inputs, self.x, name='deeplabv3plus')


        # load weights
        if self.weights == 'given':
            self.weights_model = get_file(self.weightname,
                                    self.weights_path+self.weightname,
                                    cache_subdir='models')

            self.model.load_weights(self.weights_model, by_name=True)
        else:

            # what parameters did they use?
            # Furthermore, during training, we used the stochastic gradient descent (SGD) with momentum
            # 0.9 for 200000 iterations, base learning rate 0.005, mini-batch size 8 and learning power 0.9.
            # epochs=2e5
            epochs=3000
            sgd = SGD(learning_rate=0.005, nesterov=True, momentum=0.9, decay=0.9/epochs)
            self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
            self.model.fit([k_data], [k_data_labels], batch_size=8,epochs=epochs)
            # self.model.save('unwrapping_phase_model.h5')  # creates a HDF5 file 'my_model.h5'
            # Save the weights
            self.model.save_weights('/home/jan/Desktop/UnwrappingPhase/Training/add_noise_2/my_checkpoint')

            # Evaluate the model
            print(self.model.metrics_names)
            loss, acc = self.model.evaluate(k_data_test, k_data_labels_test, verbose=2)
            print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


        # Display the model's architecture
        self.model.summary()


        # self.parallel_model = multi_gpu_model(self.model, gpus=8)
        # # what parameters did they use?
        # self.parallel_model.compile(optimizer='adam', loss='binary_crossentropy', loss_weights=0.1)
        # # fit model with data, what batch size, and what number of epochs did they use?
        # if k_data_labels is not None:
        #     self.parallel_model.fit([k_data], [k_data_labels], epochs=50, batch_size=32)
        #     self.parallel_model.save('unwrapping_phase_model.h5')  # creates a HDF5 file 'my_model.h5'
        # else:
        #     # download weights!
        #     self.parallel_model.load_model('unwrapping_phase_model.h5')


    def model_architecture(self):
        if self.input_tensor is None:
            self.img_input = Input(shape=self.input_shape)
        else:
            self.img_input = self.input_tensor

        if self.os == 8:
            self.entry_block3_stride = 1
            self.middle_block_rate = 2  # ! Not mentioned in paper, but required
            self.exit_block_rates = (2, 4)
            self.atrous_rates = (12, 24, 36)
        else:
            self.entry_block3_stride = 2
            self.middle_block_rate = 1
            self.exit_block_rates = (1, 2)
            self.atrous_rates = (6, 12, 18)



        self.x = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(
            self.img_input)
        self.x = BatchNormalization(name='entry_flow_conv1_1_BN')(self.x)
        self.x = Activation(tf.nn.relu)(self.x)

        self.x = self._conv2d_same(self.x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        self.x = BatchNormalization(name='entry_flow_conv1_2_BN')(self.x)
        self.x = Activation(tf.nn.relu)(self.x)

        self.x = self._xception_block(self.x, [128, 128, 128], 'entry_flow_block1', skip_connection_type='conv',
                                      stride=2, depth_activation=False)
        self.x, self.skip1 = self._xception_block(self.x, [256, 256, 256], 'entry_flow_block2',
                                                  skip_connection_type='conv', stride=2, depth_activation=False,
                                                  return_skip=True)

        self.x = self._xception_block(self.x, [728, 728, 728], 'entry_flow_block3', skip_connection_type='conv',
                                      stride=self.entry_block3_stride, depth_activation=False)
        for i in range(16):
            self.x = self._xception_block(self.x, [728, 728, 728], f'middle_flow_unit_{i + 1}',
                                          skip_connection_type='sum', stride=1, rate=self.middle_block_rate,
                                          depth_activation=False)

        self.x = self._xception_block(self.x, [728, 1024, 1024], 'exit_flow_block1', skip_connection_type='conv',
                                      stride=1, rate=self.exit_block_rates[0], depth_activation=False)
        self.x = self._xception_block(self.x, [1536, 1536, 2048], 'exit_flow_block2', skip_connection_type='none',
                                      stride=1, rate=self.exit_block_rates[1], depth_activation=True)


        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        self.shape_before = tf.shape(self.x)
        self.b4 = GlobalAveragePooling2D()(self.x)
        # from (b_size, channels)->(b_size, 1, 1, channels)
        self.b4 = Lambda(lambda x: K.expand_dims(x, 1))(self.b4)
        self.b4 = Lambda(lambda x: K.expand_dims(x, 1))(self.b4)
        self.b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(self.b4)
        self.b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(self.b4)
        self.b4 = Activation(tf.nn.relu)(self.b4)
        # upsample. have to use compat because of the option align_corners
        self.size_before = tf.keras.backend.int_shape(self.x)
        self.b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, self.size_before[1:3],method='bilinear',
                                                             align_corners=True))(self.b4)

        # simple 1x1
        self.b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(self.x)
        self.b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(self.b0)
        self.b0 = Activation(tf.nn.relu, name='aspp0_activation')(self.b0)

        # rate = 6 (12)
        self.b1 = self.sep_conv_bn(self.x, 256, 'aspp1', rate=self.atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        self.b2 = self.sep_conv_bn(self.x, 256, 'aspp2', rate=self.atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        self.b3 = self.sep_conv_bn(self.x, 256, 'aspp3', rate=self.atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        self.x = Concatenate()([self.b4, self.b0, self.b1, self.b2, self.b3])


        self.x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(self.x)
        self.x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(self.x)
        self.x = Activation(tf.nn.relu)(self.x)
        self.x = Dropout(0.1)(self.x)
        # DeepLab v.3+ decoder


        # Feature projection
        # x4 (x2) block
        self.size_before2 = tf.keras.backend.int_shape(self.x)
        self.x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, self.skip1.shape[1:3], method='bilinear',
                                                             align_corners=True))(self.x)

        self.dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(self.skip1)
        self.dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(self.dec_skip1)
        self.dec_skip1 = Activation(tf.nn.relu)(self.dec_skip1)
        self.x = Concatenate()([self.x, self.dec_skip1])
        self.x = self.sep_conv_bn(self.x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
        self.x = self.sep_conv_bn(self.x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

        # you can use it with arbitary number of classes
        if (self.weights == 'given' and self.classes == 21) or (self.weights == 'given'  and self.classes == 19):
            self.last_layer_name = 'logits_semantic'
        else:
            self.last_layer_name = 'custom_logits_semantic'

        # you can use it with arbitary number of classes
        self.x = Conv2D(self.classes, (1, 1), padding='same', name=self.last_layer_name)(self.x)
        self.size_before3 = tf.keras.backend.int_shape(self.img_input)
        self.x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, self.size_before3[1:3], method='bilinear',
                                                             align_corners=True))(self.x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if self.input_tensor is not None:
            self.inputs = get_source_inputs(self.input_tensor)
        else:
            self.inputs = self.img_input

        if self.activation in {'softmax', 'sigmoid'}:
            self.x = tf.keras.layers.Activation(self.activation)(self.x)




    @staticmethod
    def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
        """Implements right 'same' padding for even kernel sizes
            Without this there is a 1 pixel drift when stride = 2
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
        """
        if stride == 1:
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='same', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='valid', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)

    def _xception_block(self, inputs, depth_list, prefix, skip_connection_type, stride,
                        rate=1, depth_activation=False, return_skip=False):
        """ Basic building block of modified Xception network
            Args:
                inputs: input tensor
                depth_list: number of filters in each SepConv layer. len(depth_list) == 3
                prefix: prefix before name
                skip_connection_type: one of {'conv','sum','none'}
                stride: stride at last depthwise conv
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                return_skip: flag to return additional tensor after 2 SepConvs for decoder
                """
        residual = inputs
        for i in range(3):
            residual = self.sep_conv_bn(residual, depth_list[i], f"{prefix}_separable_conv{i + 1}",
                                        stride=stride if i == 2 else 1, rate=rate, depth_activation=depth_activation)
            if i == 1:
                skip = residual

        if skip_connection_type == 'conv':
            shortcut = self._conv2d_same(inputs, depth_list[-1], f"{prefix}_shortcut", kernel_size=1, stride=stride)
            shortcut = BatchNormalization(name=f"{prefix}_shortcut_BN")(shortcut)
            outputs = layers.add([residual, shortcut])
        elif skip_connection_type == 'sum':
            outputs = layers.add([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        if return_skip:
            return outputs, skip
        else:
            return outputs




    @staticmethod
    def sep_conv_bn(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & poinwise convs
                epsilon: epsilon to use in BN layer
        """

        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = Activation(tf.nn.relu)(x)
        x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=f"{prefix}_depthwise")(x)
        x = BatchNormalization(name=f"{prefix}_depthwise_BN", epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(tf.nn.relu)(x)
        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=f"{prefix}_pointwise")(x)
        x = BatchNormalization(name=f"{prefix}_pointwise_BN", epsilon=epsilon)(x)
        if depth_activation:
            x = Activation(tf.nn.relu)(x)
        return x
