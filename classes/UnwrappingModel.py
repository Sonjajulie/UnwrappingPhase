# import numpy as np

# Tensor flow libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization  # , Dropout, Activation
from tensorflow.keras.layers import Conv2D, Concatenate, Input, DepthwiseConv2D
from tensorflow.keras.layers import SeparableConv2D, UpSampling2D
from tensorflow.keras.utils import multi_gpu_model
# from tensorflow.keras.models import load_model


class UnwrappingModel:
    def __init__(self, k_data, k_data_labels=None):
        # initialize deeplab3v+ NN - architecture

        k_shape = k_data.shape[1:]
        output, low_feat_layer = self.encoder(k_shape)
        self.output_decoder = self.decoder(output, low_feat_layer)
        # create keras - model
        self.model = Model(inputs=[self.input_encoder], outputs=[self.output_decoder])
        self.parallel_model = multi_gpu_model(self.model, gpus=8)
        # what parameters did they use?
        self.parallel_model.compile(optimizer='adam', loss='binary_crossentropy', loss_weights=0.1)
        # fit model with data, what batch size, and what number of epochs did they use?
        if k_data_labels is not None:
            self.parallel_model.fit([k_data], [k_data_labels], epochs=50, batch_size=32)
            self.parallel_model.save('unwrapping_phase_model.h5')  # creates a HDF5 file 'my_model.h5'
        else:
            # download weights!
            self.parallel_model.load_model('unwrapping_phase_model.h5')


    def encoder(self, k_shape):
        output_dcnn, low_feat_layer = self.encoder_DCNN(k_shape)
        output_aspp = self.encoder_ASPP(output_dcnn)
        return output_aspp, low_feat_layer

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    def encoder_DCNN(self, k_shape):
        """
        Encoder 1 - Deep convolutional neural network (DCNN)
        according to the paper Zhang et al.(2019)
        """

        # Entry flow
        # 1. Conv 32, 3x3, stride 2
        # 2. Conv 64, 3x3
        # 3. Sep Conv 128, 3x3      |   Conv 128, 1x1 Stride 2
        # 4. Sep Conv 128, 3x3
        # 5. Sep Conv 128, 3x3 Stride 2
        # 6. Concat
        # 7. Sep Conv 256, 3x3       |   Conv 256, 1x1 Stride 2
        # 8. Sep Conv 256, 3x3
        # 9. Depthwise Sep Conv 256, 3x3
        # 10. Concat
        # 11. Sep Conv 728, 3x3       |   Conv 728, 1x1 Stride 2
        # 12. Sep Conv 728, 3x3
        # 13. Atrous Conv 728, 3x3, rate=2
        # 14. Concat

        self.input_encoder = Input(shape=k_shape, name='input_dcnn')
        self._l1 = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', activation='relu')(self.input_encoder)
        self._l1b = BatchNormalization()(self._l1)


        self._l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(self._l1b)
        self._l2b = BatchNormalization()(self._l2)

        self._l3 = SeparableConv2D(128, (3, 3), padding='same', activation='relu')(self._l2b)
        self._l3b = BatchNormalization()(self._l3)
        self._l4 = SeparableConv2D(128, (3, 3), padding='same', activation='relu')(self._l3b)
        self._l4b = BatchNormalization()(self._l4)
        self._l5 = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(self._l4b)
        self._l5b = BatchNormalization()(self._l5)

        self._ll1 = Conv2D(128, (1, 1), strides=(2,2), padding='valid', activation='relu')(self._l2b)
        self._ll1b = BatchNormalization()(self._ll1)

        self._l6 = Concatenate(axis=1)([self._l5b, self._ll1b])
        self._l7 = SeparableConv2D(256, (3, 3), padding='same', activation='relu')(self._l6)
        self._l7b = BatchNormalization()(self._l7)
        self._l8 = SeparableConv2D(256, (3, 3), padding='same', activation='relu')(self._l7b)
        self._l8b = BatchNormalization()(self._l8) #  depth_multiplier=256,
        self._l9 = SeparableConv2D(256,kernel_size=(3, 3), strides=2, padding='same', activation='relu')(self._l8b)
        self._l9b = BatchNormalization()(self._l9)

        self._ll2 = Conv2D(256, (1, 1), 2, padding='valid', activation='relu')(self._ll1b)
        self._ll2b = BatchNormalization()(self._ll2)

        self._l10 = Concatenate(axis=1)([self._l9b, self._ll2b])
        self._l11 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l10)
        self._l11b = BatchNormalization()(self._l11)
        self._l12 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l11b)
        self._l12b = BatchNormalization()(self._l12)
        self._l13 = Conv2D(728, (3, 3), dilation_rate=(2, 2), padding='valid', activation='relu')(self._l12b)
        self._l13b = BatchNormalization()(self._l13)
        # self._l13 = SeparableConv2D(728, kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(self._l12b)
        # self._l13b = BatchNormalization()(self._l13)

        self._ll3 = Conv2D(728, (1, 1), strides=2, padding='valid', activation='relu')(self._ll2b)
        self._ll3b = BatchNormalization()(self._ll3)
        # 92 X 28 X 728 = valid
        # 92 X 32 X 728 = same
        # 16 X 16 X 728 = ll2b
        # 48 X 16 X 728 = _l10
        self._l14 = Concatenate(axis=1)([self._l13b, self._ll3b])

        # Middle flow:
        # 15. Sep Conv 728, 3x3     |     Input
        # 16. Sep Conv 728, 3x3
        # 17. Sep Conv 728, 3x3
        # 18. Concat

        for i in range(16):
            self._l15 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l14)
            self._l15b = BatchNormalization()(self._l15)
            self._l16 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l15b)
            self._l16b = BatchNormalization()(self._l16)
            self._l17 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l16b)
            self._l17b = BatchNormalization()(self._l17)
            self._l18 = Concatenate(axis=1)([self._l14, self._l17b])
            self._l14 = self._l18

        # Exit flow:
        # 19. Sep Conv 728, 3x3     |   Conv 1024, 1x1 Stride 2
        # 20. Sep Conv 1024, 3x3
        # 21. Atrous Conv 1024, 3x3, rate=4
        # 22. Concat
        # 23. Sep Conv 1536, 3x3
        # 24. Depthwise Sep Conv 1536, 3x3
        # 25. Depthwise Sep Conv 2048, 3x3

        self._l19 = SeparableConv2D(728, (3, 3), padding='same', activation='relu')(self._l14)
        self._l19b = BatchNormalization()(self._l19)
        self._l20 = SeparableConv2D(1024, (3, 3), padding='same', activation='relu')(self._l19b)
        self._l20b = BatchNormalization()(self._l20)
        # self._l21 = SeparableConv2D(1024, (3, 3), padding='same', dilation_rate=4, activation='relu')(self._l20b)
        self._l21 = SeparableConv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='relu')(self._l20b)
        self._l21b = BatchNormalization()(self._l21)

        self._ll5 = Conv2D(1024, (1, 1), 2, padding='valid', activation='relu')(self._l14)
        self._ll5b = BatchNormalization()(self._ll5)

        self._l22 = Concatenate(axis=1)([self._l21b, self._ll5b])
        self._l23 = SeparableConv2D(1536, (3, 3), padding='same', activation='relu')(self._l22)
        self._l23b = BatchNormalization()(self._l23)
        self._l24 = SeparableConv2D(1536, (3, 3), padding='same', activation='relu')(self._l23b)
        self._l24b = BatchNormalization()(self._l24)
        self._l25 = SeparableConv2D(2048, (3, 3), padding='same', activation='relu')(self._l24b)
        self._l25b = BatchNormalization()(self._l25)

        # final layer + low level features
        return self._l25b, self._ll1b

    # noinspection PyPep8Naming,PyAttributeOutsideInit
    def encoder_ASPP(self, output_dcnn):
        """
        Encoder 2 - atrous spatial pyramid pooling (ASPP):
        according to Zhang et al.(2019)
        """

        # 1. 1x1 conv   |  3x3 conv, rate 12  |  3x3 conv, rate 24   |  3x3 conv, rate 36  |  Image pooling
        # Merge
        # 6. 1x1 conv

        # input_img = Input(shape=(2,), name='input_aspp)
        self._p1 = Conv2D(256, (1, 1), padding='same')(output_dcnn)
        self._p2 = Conv2D(256, (3, 3), dilation_rate=12, padding='same')(output_dcnn)
        self._p3 = Conv2D(256, (3, 3), dilation_rate=24, padding='same')(output_dcnn)
        self._p4 = Conv2D(256, (3, 3), dilation_rate=36, padding='same')(output_dcnn)
        self._p5 = Conv2D(256, (3, 3), dilation_rate=36, padding='same')(output_dcnn)

        self._p6 = Concatenate(axis=1)([self._p1, self._p2, self._p3, self._p4, self._p5])
        self._p7 = Conv2D(256, (1, 1), padding='same')(self._p6)  # final layer
        return self._p7

    # noinspection PyAttributeOutsideInit
    def decoder(self, output_aspp, output_dcnn):
        """
        decoder of the Deeplabv3+ architecture
        """

        # Input - DCNN      |    Input - ASPP
        # 1. 1x1 conv       | Upsampling by 2

        # noinspection PyPep8Naming
        # input_DCNN = Input(shape=(2,))
        self._d1 = Conv2D(256, (1, 1), padding='same')(output_dcnn)

        # noinspection PyPep8Naming
        # input_ASPP = Input(shape=(2,))
        self._dd1 = UpSampling2D(size=4, interpolation='bilinear')(output_aspp)

        #  2. Concat
        self._d2 = Concatenate(axis=1)([self._d1, self._dd1])

        # 3. 3x3 conv
        self._d3 = Conv2D(256, (3, 3), padding='same')(self._d2)

        # 4. Upsampling by 4
        self._d4 = UpSampling2D(size=4, interpolation='bilinear')(self._d3)

        return self._d4  # final layer
