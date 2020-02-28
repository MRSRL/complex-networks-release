import os
import sys

import numpy as np
import theano.tensor as T
from keras import backend as K
from keras.engine.topology import Layer

from .tf_util import getImag, getReal

sys.path.append("/home/ekcole/Workspace/recon_unrolled/mri_util")


class modReLU(Layer):
    def get_realpart(self, x):
        image_format = K.image_data_format()
        ndim = K.ndim(x)
        input_shape = K.shape(x)

        if (image_format == "channels_first" and ndim != 3) or ndim == 2:
            input_dim = input_shape[1] // 2
            return x[:, :input_dim]

        input_dim = input_shape[-1] // 2
        if ndim == 3:
            return x[:, :, :input_dim]
        elif ndim == 4:
            return x[:, :, :, :input_dim]
        elif ndim == 5:
            return x[:, :, :, :, :input_dim]

    def get_imagpart(self, x):
        image_format = K.image_data_format()
        ndim = K.ndim(x)
        input_shape = K.shape(x)

        if (image_format == "channels_first" and ndim != 3) or ndim == 2:
            input_dim = input_shape[1] // 2
            return x[:, input_dim:]

        input_dim = input_shape[-1] // 2
        if ndim == 3:
            return x[:, :, input_dim:]
        elif ndim == 4:
            return x[:, :, :, input_dim:]
        elif ndim == 5:
            return x[:, :, :, :, input_dim:]

    def get_abs(self, x):
        real = getReal(x, self.data_format)
        imag = getImag(x, self.data_format)

        return K.sqrt(real * real + imag * imag)

    def __init__(self, data_format, output_dim, **kwargs):
        self.data_format = data_format
        self.output_dim = output_dim
        super(modReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        #         shape = list(input_shape)
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(
            name="b", shape=(self.output_dim), initializer="uniform", trainable=True
        )
        super(modReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        real = getReal(x, self.data_format)
        imag = getImag(x, self.data_format)
        act = real
        mag = self.get_abs(x)
        # ang = self.get_angle(x)

        comp_num = real + 1j * imag

        z_norm = mag + 0.00001
        step1 = z_norm + self.b
        step2 = K.relu(step1)

        real_act = (real / mag) * step2
        imag_act = (imag / mag) * step2

        act = tf.concat([real_act, imag_act], axis=1)
        # activation = K.cast_to_floatx(act)

        return act

    def compute_output_shape(self, input_shape):
        return input_shape


# Implementation of ModRelu
# Arjovsky et al., 2015
# How do we make b a learnable parameter?
# class modrelu(Layer):
#     def __init__(self, **kwargs):
#         super(modReLU, self).__init__(**kwargs)

#     def build(self, input_shape):
#         #Add b as a trainable parameter
#         #Shape is one parameter
#         #Random - zero mean, unit variance
#         self.b = self.add_weight(name='b',
#                                   shape= ,
#                                   initializer=,
#                                   trainable=True)
#         super(modrelu, self).build(input_shape)

#     def call(self, z):
#         z_norm = tf.abs(z)
#         zone = z_norm + self.b
#         step1 = tf.complex(tf.nn.relu(zone), tf.zeros_like(z_norm))
#         step2 = z / tf.complex(z_norm, tf.zeros_like(z_norm))
#         tf_output = tf.multiply(step2, step1)
#     return tf_output

#     def compute_output_shape(self, input_shape):
#     return (input_shape)`
