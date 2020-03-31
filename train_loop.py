from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
import types
import warnings

import numpy as np
import tensorflow as tf

import mri_data
import mri_model
from mri_util import fftc, metrics, tf_util

tf.app.flags.DEFINE_string("gpu", "single", "Single or multi GPU Mode")
tf.app.flags.DEFINE_string("conv", "real", "Real or complex convolution")
tf.app.flags.DEFINE_boolean("do_conjugate", "False", "Complex conjugate")
# Data dimensions
tf.app.flags.DEFINE_integer("feat_map", 128, "Number of feature maps")

tf.app.flags.DEFINE_integer("shape_y", 180, "Image shape in Y")
tf.app.flags.DEFINE_integer("shape_z", 80, "Image shape in Z")
tf.app.flags.DEFINE_integer(
    "num_channels", 8, "Number of channels for input datasets.")
tf.app.flags.DEFINE_integer(
    "num_emaps", 1, "Number of eigen maps for input sensitivity maps."
)

# For logging
tf.app.flags.DEFINE_integer("print_level", 1, "Print out level.")
tf.app.flags.DEFINE_string(
    "log_root", "summary", "Root directory where logs are written to."
)
tf.app.flags.DEFINE_string(
    "train_dir", "train", "Directory for checkpoints and event logs."
)
tf.app.flags.DEFINE_integer(
    "num_summary_image", 4, "Number of images for summary output"
)
tf.app.flags.DEFINE_integer(
    "log_every_n_steps", 10, "The frequency with which logs are print."
)
tf.app.flags.DEFINE_integer(
    "save_summaries_secs",
    10,
    "The frequency with which summaries are saved, " + "in seconds.",
)

tf.app.flags.DEFINE_integer(
    "save_interval_secs",
    10,
    "The frequency with which the model is saved, " + "in seconds.",
)

tf.app.flags.DEFINE_integer(
    "random_seed", 1000, "Seed to initialize random number generators."
)

# For model
tf.app.flags.DEFINE_integer(
    "num_grad_steps", 2, "Number of grad steps for unrolled algorithms"
)
tf.app.flags.DEFINE_boolean(
    "do_hard_proj", True, "Turn on/off hard data projection at the end"
)

# Optimization Flags
tf.app.flags.DEFINE_string("device", "0", "GPU device to use.")
tf.app.flags.DEFINE_integer(
    "batch_size", 4, "The number of samples in each batch.")

tf.app.flags.DEFINE_float(
    "adam_beta1", 0.9, "The exponential decay rate for the 1st moment estimates."
)
tf.app.flags.DEFINE_float(
    "adam_beta2", 0.999, "The exponential decay rate for the 2nd moment estimates."
)
tf.app.flags.DEFINE_float(
    "opt_epsilon", 1.0, "Epsilon term for the optimizer.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
tf.app.flags.DEFINE_integer(
    "max_steps", None, "The maximum number of training steps.")

# Dataset Flags
tf.app.flags.DEFINE_string(
    "mask_path", "masks", "Directory where masks are located.")
tf.app.flags.DEFINE_string(
    "train_path", "train", "Sub directory where training data are located."
)
tf.app.flags.DEFINE_string(
    "dataset_dir", "dataset", "The directory where the dataset files are stored."
)

tf.app.flags.DEFINE_boolean(
    "do_validation", True, "Turn on/off validation during training"
)

tf.app.flags.DEFINE_string(
    "mode", "train_validate", "Train_validate, train, or predict"
)

tf.app.flags.DEFINE_string(
    "activation", "relu", "The activation function used")
# If not defined will loop through entire test directory
tf.app.flags.DEFINE_integer("num_cases", None, "The number of inference files")

# plot middle layer weights in frequency domain
tf.app.flags.DEFINE_integer("layer_num", 0, "The number layer to plot")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # path where model checkpoints and summaries will be saved
    model_dir = os.path.join(FLAGS.log_root, FLAGS.train_dir)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True,
                              log_device_placement=True)
    ) as sess:
        """Execute main function."""
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

        if not FLAGS.dataset_dir:
            raise ValueError(
                "You must supply the dataset directory with " + "--dataset_dir"
            )

        if FLAGS.random_seed >= 0:
            random.seed(FLAGS.random_seed)
            np.random.seed(FLAGS.random_seed)

        tf.logging.set_verbosity(tf.logging.INFO)

        print("Preparing dataset...")
        out_shape = [FLAGS.shape_z, FLAGS.shape_y]
        train_dataset, num_files = mri_data.create_dataset(
            os.path.join(FLAGS.dataset_dir, "train"),
            FLAGS.mask_path,
            num_channels=FLAGS.num_channels,
            num_emaps=FLAGS.num_emaps,
            batch_size=FLAGS.batch_size,
            out_shape=out_shape,
        )

        # channels last format: batch, z, y, channels
        # placeholders
        ks_shape = [None, FLAGS.shape_z, FLAGS.shape_y, FLAGS.num_channels]
        ks_place = tf.placeholder(tf.complex64, ks_shape)
        sense_shape = [None, FLAGS.shape_z,
                       FLAGS.shape_y, 1, FLAGS.num_channels]
        sense_place = tf.placeholder(tf.complex64, sense_shape)
        im_shape = [None, FLAGS.shape_z, FLAGS.shape_y, 1]
        im_truth_place = tf.placeholder(tf.complex64, im_shape)

        # run through unrolled model
        im_out_place = mri_model.unroll_fista(
            ks_place,
            sense_place,
            is_training=True,
            verbose=True,
            do_hardproj=FLAGS.do_hard_proj,
            num_summary_image=FLAGS.num_summary_image,
            resblock_num_features=FLAGS.feat_map,
            num_grad_steps=FLAGS.num_grad_steps,
            conv=FLAGS.conv,
            do_conjugate=FLAGS.do_conjugate,
            activation=FLAGS.activation
        )

        # tensorboard summary function
        _create_summary(sense_place, ks_place, im_out_place, im_truth_place)

        # define L1 loss between output and ground truth
        loss = tf.reduce_mean(tf.abs(im_out_place - im_truth_place), name="l1")
        loss_sum = tf.summary.scalar("loss/l1", loss)

        # optimize using Adam
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate,
            name="opt",
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
        ).minimize(loss)

        # counter for saving checkpoints
        with tf.variable_scope("counter"):
            counter = tf.get_variable(
                "counter",
                shape=[1],
                initializer=tf.constant_initializer([0]),
                dtype=tf.int32,
            )
            update_counter = tf.assign(counter, tf.add(counter, 1))

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        # initialize model
        print("[*] initializing network...")
        if not load(model_dir, saver, sess):
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # calculate number of parameters in model
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total number of trainable parameters: %d" % total_parameters)
        tf.summary.scalar("parameters/parameters", total_parameters)

        # use iterator to go through TFrecord dataset
        train_iterator = train_dataset.make_one_shot_iterator()
        features, labels = train_iterator.get_next()

        ks_truth = labels  # ground truth kspace
        ks_in = features["ks_input"]  # input kspace
        sense_in = features["sensemap"]  # sensitivity maps
        mask_recon = features["mask_recon"]  # reconstruction mask

        # ground truth kspace to image domain
        im_truth = tf_util.model_transpose(ks_truth * mask_recon, sense_in)

        # gather summaries for tensorboard
        total_summary = tf.summary.merge_all()

        print("Start from step %d." % (sess.run(counter)))
        for step in range(int(sess.run(counter)), FLAGS.max_steps):
            # evaluate input kspace, sensitivity maps, ground truth image
            ks_in_run, sense_in_run, im_truth_run = sess.run(
                [ks_in, sense_in, im_truth]
            )
            # run optimizer and collect output image from model and tensorboard summary
            im_out, total_summary_run, _ = sess.run(
                [im_out_place, total_summary, optimizer],
                feed_dict={
                    ks_place: ks_in_run,
                    sense_place: sense_in_run,
                    im_truth_place: im_truth_run,
                },
            )
            print("step", step)
            # add summary to tensorboard
            summary_writer.add_summary(total_summary_run, step)

            # save checkpoint every 500 steps
            if step % 500 == 0:
                print("saving checkpoint")
                saver.save(sess, model_dir + "/model.ckpt")

            # update recorded step training is at
            sess.run(update_counter)
        print("End of training loop")


def load(log_dir, saver, sess):
    # search for and load a model
    print("[*] Reading Checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("[*] Model restored.")
        return True
    else:
        print("[*] Failed to find a checkpoint")
        return False


def _create_summary(sense_place, ks_place, im_out_place, im_truth_place):
    # tensorboard summary function
    sensemap = sense_place
    ks_input = ks_place
    image_output = im_out_place
    image_truth = im_truth_place

    image_input = tf_util.model_transpose(ks_input, sensemap)
    mask_input = tf_util.kspace_mask(ks_input, dtype=tf.complex64)
    ks_output = tf_util.model_forward(image_output, sensemap)
    ks_truth = tf_util.model_forward(image_truth, sensemap)

    with tf.name_scope("input-output-truth"):
        summary_input = tf_util.sumofsq(ks_input, keep_dims=True)
        summary_output = tf_util.sumofsq(ks_output, keep_dims=True)
        summary_truth = tf_util.sumofsq(ks_truth, keep_dims=True)
        summary_fft = tf.log(
            tf.concat((summary_input, summary_output,
                       summary_truth), axis=2) + 1e-6
        )
        tf.summary.image("kspace", summary_fft,
                         max_outputs=FLAGS.num_summary_image)
        summary_input = tf_util.sumofsq(image_input, keep_dims=True)
        summary_output = tf_util.sumofsq(image_output, keep_dims=True)
        summary_truth = tf_util.sumofsq(image_truth, keep_dims=True)
        summary_image = tf.concat(
            (summary_input, summary_output, summary_truth), axis=2
        )
        tf.summary.image("image", summary_image,
                         max_outputs=FLAGS.num_summary_image)

    with tf.name_scope("phase"):
        summary_input = tf.angle(image_input)
        summary_output = tf.angle(image_output)
        summary_truth = tf.angle(image_truth)
        summary_image = tf.concat(
            (summary_input, summary_output, summary_truth), axis=2
        )
        tf.summary.image("total", summary_image,
                         max_outputs=FLAGS.num_summary_image)
        tf.summary.image("input", summary_input,
                         max_outputs=FLAGS.num_summary_image)
        tf.summary.image("output", summary_output,
                         max_outputs=FLAGS.num_summary_image)

    with tf.name_scope("truth"):
        summary_truth_real = tf.reduce_sum(
            image_truth, axis=-1, keep_dims=True)
        summary_truth_real = tf.real(summary_truth_real)
        tf.summary.image(
            "image_real", summary_truth_real, max_outputs=FLAGS.num_summary_image
        )

    with tf.name_scope("mask"):
        summary_mask = tf_util.sumofsq(mask_input, keep_dims=True)
        tf.summary.image("mask", summary_mask,
                         max_outputs=FLAGS.num_summary_image)

    with tf.name_scope("sensemap"):
        summary_map = tf.slice(
            tf.abs(sensemap), [0, 0, 0, 0, 0], [-1, -1, -1, 1, -1])
        summary_map = tf.transpose(summary_map, [0, 1, 4, 2, 3])
        summary_map = tf.reshape(
            summary_map, [tf.shape(summary_map)[0],
                          tf.shape(summary_map)[1], -1]
        )
        summary_map = tf.expand_dims(summary_map, axis=-1)
        tf.summary.image("image", summary_map,
                         max_outputs=FLAGS.num_summary_image)


if __name__ == "__main__":
    tf.app.run()
