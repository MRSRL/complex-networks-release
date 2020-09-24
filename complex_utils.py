from math import pi

import numpy as np
import tensorflow as tf


def complex_conv(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last", dilation_rate=(1, 1), use_bias=True,
    kernel_initializer=None, kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True
):
    # allocate half the features to real, half to imaginary
    num_features = num_features // 2

    tf_real = tf.real(tf_input)
    tf_imag = tf.imag(tf_input)

    with tf.variable_scope(None, default_name="complex_conv2d"):
        tf_real_real = tf.layers.conv2d(
            inputs=tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
        )
        tf_imag_real = tf.layers.conv2d(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
            reuse=True,
        )
        tf_real_imag = tf.layers.conv2d(
            tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
        )
        tf_imag_imag = tf.layers.conv2d(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
            reuse=True,
        )
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)

    return tf_output


def complex_conv_transpose(tf_input, num_features, kernel_size, stride, data_format="channels_last", use_bias=True,
                           kernel_initializer=None, kernel_regularizer=None, bias_regularizer=None,
                           activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True
                           ):
    # allocate half the features to real, half to imaginary
    # num_features = num_features // 2

    tf_real = tf.real(tf_input)
    tf_imag = tf.imag(tf_input)

    with tf.variable_scope(None, default_name="complex_conv2d"):
        tf_real_real = tf.layers.conv2d_transpose(
            inputs=tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
        )
        tf_imag_real = tf.layers.conv2d_transpose(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
            reuse=True,
        )
        tf_real_imag = tf.layers.conv2d_transpose(
            tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
        )
        tf_imag_imag = tf.layers.conv2d_transpose(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=[stride, stride],
            padding="same",
            data_format=data_format,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
            reuse=True,
        )
    real_out = tf_real_real - tf_imag_imag
    imag_out = tf_imag_real + tf_real_imag
    tf_output = tf.complex(real_out, imag_out)

    return tf_output


def complex_conv1d(
    tf_input, num_features, kernel_size, stride=1, data_format="channels_last", dilation_rate=(1), use_bias=True,
    kernel_initializer=None, kernel_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True
):
    # allocate half the features to real, half to imaginary
    num_features = num_features // 2

    tf_real = tf.real(tf_input)
    tf_imag = tf.imag(tf_input)

    with tf.variable_scope(None, default_name="complex_conv1d"):
        tf_real_real = tf.layers.conv1d(
            inputs=tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
        )
        tf_imag_real = tf.layers.conv1d(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="real_conv",
            reuse=True,
        )
        tf_real_imag = tf.layers.conv1d(
            tf_real,
            filters=num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
        )
        tf_imag_imag = tf.layers.conv1d(
            tf_imag,
            filters=num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="imag_conv",
            reuse=True,
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
