import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer


class modReLU(Layer):
    def __init__(self, **kwargs):
        super(modReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.b = self.add_weight(
            name="b",
            shape=(input_shape[1] / 2, input_shape[2], input_shape[3]),
            initializer="uniform",
            trainable=True,
        )
        super(modReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, z, bias):
        # relu(|z|+b) * (z / |z|)
        norm = tf.abs(z)
        scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
        scaled = tf.complex(tf.real(z) * scale, tf.imag(z) * scale)
        return scaled

    def compute_output_shape(self, input_shape):
        return input_shape
