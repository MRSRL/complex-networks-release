import tensorflow as tf
import complex_utils

tf_input = tf.ones(
    shape=[1, 256, 1], dtype=tf.dtypes.complex64, name=None)

num_features = 100
kernel_size = 3

tf_output = complex_utils.complex_conv1d(
    tf_input, num_features=num_features, kernel_size=kernel_size)

print(tf_output)
