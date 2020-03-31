from math import pi

import numpy as np
import tensorflow as tf


def complex_conv(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last"
):
    num_features = num_features // 2

    tf_real = tf.real(tf_input)
    tf_imag = tf.imag(tf_input)

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
    tf_output = tf.complex(real_out, imag_out)

    return tf_output


def complex_conv_transpose(
    tf_input, num_features, kernel_size, stride, data_format="channels_last"
):
    num_features = num_features // 2

    tf_real = tf.real(tf_input)
    tf_imag = tf.imag(tf_input)

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
    tf_output = tf.complex(real_out, imag_out)

    return tf_output


def zrelu(x):
    # x and tf_output are complex-valued
    phase = tf.angle(x)

    # Check whether phase <= pi/2
    le = tf.less_equal(phase, pi / 2)

    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    y = tf.zeros_like(x)
    x = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.greater_equal(phase, 0)

    # if phase >= 0, keep it
    # if phase < 0, throw it away and set output equal to 0
    output = tf.where(ge, x, y)

    return output


def modrelu(x, data_format="channels_last"):
    input_shape = tf.shape(x)
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    # Channel size
    shape_c = x.shape[axis_c]

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
    norm = tf.abs(x)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    output = tf.complex(tf.real(x) * scale,
                        tf.imag(x) * scale)

    return output


def cardioid(x):
    phase = tf.angle(x)
    scale = 0.5 * (1 + tf.cos(phase))
    output = tf.complex(tf.real(x) * scale, tf.imag(x) * scale)
    # output = 0.5*(1+tf.cos(phase))*z

    return output
