from math import pi

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

from mri_util import tf_util


def complex_conv(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last"
):
    num_features = num_features // 2
    check = tf_input.dtype is not tf.complex64 and tf_input.dtype is not tf.complex128
    tf_real = tf_util.getReal(tf_input, data_format)
    tf_imag = tf_util.getImag(tf_input, data_format)

    with tf.variable_scope(None, default_name="complex_conv2d"):
        tf_real_real = tf.layers.conv2d(
            tf_real,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="real_conv",
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_imag_real = tf.layers.conv2d(
            tf_imag,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="real_conv",
            reuse=True,
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_real_imag = tf.layers.conv2d(
            tf_real,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="imag_conv",
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_imag_imag = tf.layers.conv2d(
            tf_imag,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="imag_conv",
            reuse=True,
            data_format=data_format,
            strides=[stride, stride],
        )
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.concat([real_out, imag_out], axis=-1)
    return tf_output


def complex_conv_transpose(
    tf_input, num_features, kernel_size, stride, data_format="channels_last"
):
    num_features = num_features // 2
    check = tf_input.dtype is not tf.complex64 and tf_input.dtype is not tf.complex128
    tf_real = getReal(tf_input, data_format)
    tf_imag = getImag(tf_input, data_format)

    with tf.variable_scope(None, default_name="complex_conv2d"):
        tf_real_real = tf.layers.conv2d_transpose(
            tf_real,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="real_conv",
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_imag_real = tf.layers.conv2d_transpose(
            tf_imag,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="real_conv",
            reuse=True,
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_real_imag = tf.layers.conv2d_transpose(
            tf_real,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="imag_conv",
            data_format=data_format,
            strides=[stride, stride],
        )
        tf_imag_imag = tf.layers.conv2d_transpose(
            tf_imag,
            num_features,
            kernel_size,
            padding="same",
            use_bias=False,
            name="imag_conv",
            reuse=True,
            data_format=data_format,
            strides=[stride, stride],
        )
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.concat([real_out, imag_out], axis=-1)
    return tf_output


def zrelu(z, data_format):
    real = tf_util.getReal(z, data_format)
    imag = tf_util.getImag(z, data_format)
    comp = tf.complex(real, imag)
    ang = tf.angle(comp)

    zero = tf.constant(False, dtype=tf.bool)

    # Check whether phase <= pi/2
    le = tf.less_equal(ang, pi / 2)
    # if phase <= pi/2, then le = True
    # if phase > pi/2, then le = False
    x = comp
    y = tf.zeros_like(comp)
    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    comp = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.greater_equal(ang, 0)
    # if phase >= 0, then ge = True
    # if phase < 0, then ge = False

    # x is updated comp with pi/2 comparison values
    x = comp

    # if phase >= 0, keep it in comp
    # if phase < 0, throw it away and set comp equal to 0
    comp = tf.where(ge, x, y)

    realOut = tf.real(comp)
    imagOut = tf.imag(comp)

    output_shape = tf.shape(z)

    if data_format == "channels_last":
        tf_output = tf.concat([realOut, imagOut], 2)
    else:
        tf_output = tf.concat([realOut, imagOut], 0)
    tf_output = tf.reshape(tf_output, output_shape)

    return tf_output


def modrelu(tf_output, data_format="channels_last"):
    input_shape = tf.shape(tf_output)
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    # Reshape into complex number
    real = tf_util.getReal(tf_output, data_format)
    imag = tf_util.getImag(tf_output, data_format)
    tf_output = tf.complex(real, imag)
    # Channel size
    shape_c = tf_output.shape[axis_c]

    with tf.name_scope("bias") as scope:
        if data_format == "channels_last":
            bias_shape = (1, 1, 1, shape_c)
        else:
            bias_shape = (1, shape_c, 1, 1)
        bias = tf.get_variable(name=scope,
                               shape=bias_shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    # relu(|z|+b) * (z / |z|)
    norm = tf.abs(tf_output)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    tf_output = tf.complex(tf.real(tf_output) * scale,
                           tf.imag(tf_output) * scale)

    realOut = tf.real(tf_output)
    imagOut = tf.imag(tf_output)

    # Interleave
    if data_format == "channels_last":
        tf_output = tf.concat([realOut, imagOut], 2)
    else:
        tf_output = tf.concat([realOut, imagOut], 0)

    tf_output = tf.reshape(tf_output, input_shape)

    return tf_output

def cardioid(z):
    phase = tf.angle(z)
    scale = 0.5 * (1 + tf.cos(phase))
    output = tf.complex(tf.real(z) * scale, tf.imag(z) * scale)
    # output = 0.5*(1+tf.cos(phase))*z
    return output