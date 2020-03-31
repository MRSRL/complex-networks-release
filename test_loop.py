"""Test loop that will calculate image metrics."""
from __future__ import absolute_import, division, print_function

import os
import random
import subprocess
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation

import mri_data
import mri_model
from mri_util import cfl, fftc, metrics, tf_util

BIN_BART = "bart"

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.app.flags.DEFINE_string("gpu", "single", "Single or multi GPU Mode")
tf.app.flags.DEFINE_string("conv", "complex", "Real or complex convolution")
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
    model_dir = os.path.join(FLAGS.log_root, FLAGS.train_dir)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    bart_dir = os.path.join(model_dir, "bart_recon")
    if not os.path.exists(bart_dir):
        os.makedirs(bart_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
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

        test_dataset, num_files = mri_data.create_dataset(
            os.path.join(FLAGS.dataset_dir, "test"),
            FLAGS.mask_path,
            num_channels=FLAGS.num_channels,
            num_emaps=FLAGS.num_emaps,
            batch_size=FLAGS.batch_size,
            out_shape=out_shape,
        )
        # channels first: (batch, channels, z, y)
        # placeholders
        ks_shape = [None, FLAGS.shape_z, FLAGS.shape_y, FLAGS.num_channels]
        ks_place = tf.placeholder(tf.complex64, ks_shape)
        sense_shape = [None, FLAGS.shape_z,
                       FLAGS.shape_y, 1, FLAGS.num_channels]
        sense_place = tf.placeholder(tf.complex64, sense_shape)
        im_shape = [None, FLAGS.shape_z, FLAGS.shape_y, 1]
        im_truth_place = tf.placeholder(tf.complex64, im_shape)
        # run through unrolled
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
        )

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

        # initialize model
        print("[*] initializing network...")
        if not load(model_dir, saver, sess):
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # See how many parameters are in model
        total_parameters = 0
        for variable in tf.trainable_variables():
            variable_parameters = 1
            for dim in variable.get_shape():
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total number of trainable parameters: %d" % total_parameters)

        test_iterator = test_dataset.make_one_shot_iterator()
        features, labels = test_iterator.get_next()

        ks_truth = labels
        ks_in = features["ks_input"]
        sense_in = features["sensemap"]
        mask_recon = features["mask_recon"]
        im_truth = tf_util.model_transpose(ks_truth * mask_recon, sense_in)

        total_summary = tf.summary.merge_all()

        output_psnr = []
        output_nrmse = []
        output_ssim = []
        cs_psnr = []
        cs_nrmse = []
        cs_ssim = []

        for test_file in range(num_files):
            ks_in_run, sense_in_run, im_truth_run = sess.run(
                [ks_in, sense_in, im_truth]
            )
            im_out, total_summary_run = sess.run(
                [im_out_place, total_summary],
                feed_dict={
                    ks_place: ks_in_run,
                    sense_place: sense_in_run,
                    im_truth_place: im_truth_run,
                },
            )

            # CS recon
            bart_test = bart_cs(bart_dir, ks_in_run, sense_in_run, l1=0.007)
            # bart_test = None

            # handle batch dimension
            for b in range(FLAGS.batch_size):
                truth = im_truth_run[b, :, :, :]
                out = im_out[b, :, :, :]
                psnr, nrmse, ssim = metrics.compute_all(
                    truth, out, sos_axis=-1)
                output_psnr.append(psnr)
                output_nrmse.append(nrmse)
                output_ssim.append(ssim)

            print("output mean +/ standard deviation psnr, nrmse, ssim")
            print(
                np.mean(output_psnr),
                np.std(output_psnr),
                np.mean(output_nrmse),
                np.std(output_nrmse),
                np.mean(output_ssim),
                np.std(output_ssim),
            )

            psnr, nrmse, ssim = metrics.compute_all(
                im_truth_run, bart_test, sos_axis=-1
            )
            cs_psnr.append(psnr)
            cs_nrmse.append(nrmse)
            cs_ssim.append(ssim)

            print("cs mean +/ standard deviation psnr, nrmse, ssim")
            print(
                np.mean(cs_psnr),
                np.std(cs_psnr),
                np.mean(cs_nrmse),
                np.std(cs_nrmse),
                np.mean(cs_ssim),
                np.std(cs_ssim),
            )
        print("End of testing loop")
        txt_path = os.path.join(model_dir, "metrics.txt")
        f = open(txt_path, "w")
        f.write(
            "parameters = "
            + str(total_parameters)
            + "\n"
            + "output psnr = "
            + str(np.mean(output_psnr))
            + " +\- "
            + str(np.std(output_psnr))
            + "\n"
            + "output nrmse = "
            + str(np.mean(output_nrmse))
            + " +\- "
            + str(np.std(output_nrmse))
            + "\n"
            + "output ssim = "
            + str(np.mean(output_ssim))
            + " +\- "
            + str(np.std(output_ssim))
            + "\n"
            "cs psnr = "
            + str(np.mean(cs_psnr))
            + " +\- "
            + str(np.std(cs_psnr))
            + "\n"
            + "output nrmse = "
            + str(np.mean(cs_nrmse))
            + " +\- "
            + str(np.std(cs_nrmse))
            + "\n"
            + "output ssim = "
            + str(np.mean(cs_ssim))
            + " +\- "
            + str(np.std(cs_ssim))
        )
        f.close()


def load(log_dir, saver, sess):
    print("[*] Reading Checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("[*] Model restored.")
        return True
    else:
        print("[*] Failed to find a checkpoint")
        return False


def bart_cs(bart_dir, ks, sensemap, l1=0.01):
    cfl_ks = np.squeeze(ks)
    cfl_ks = np.expand_dims(cfl_ks, -2)
    cfl_sensemap = np.squeeze(sensemap)
    cfl_sensemap = np.expand_dims(cfl_sensemap, -2)

    ks_dir = os.path.join(bart_dir, "file_ks")
    sense_dir = os.path.join(bart_dir, "file_sensemap")
    img_dir = os.path.join(bart_dir, "file_img")

    cfl.write(ks_dir, cfl_ks, "R")
    cfl.write(sense_dir, cfl_sensemap, "R")

    # L1-wavelet regularized
    cmd_flags = "-S -e -R W:3:0:%f -i 100" % l1

    cmd = "%s pics %s %s %s %s" % (
        BIN_BART, cmd_flags, ks_dir, sense_dir, img_dir,)
    subprocess.check_call(["bash", "-c", cmd])
    bart_recon = load_recon(img_dir, sense_dir)
    return bart_recon


def load_recon(file, file_sensemap):
    bart_recon = np.squeeze(cfl.read(file))
    if bart_recon.ndim == 2:
        bart_recon = np.transpose(bart_recon, [1, 0])
        bart_recon = np.expand_dims(bart_recon, axis=0)
        bart_recon = np.expand_dims(bart_recon, axis=-1)
    if bart_recon.ndim == 3:
        bart_recon = np.transpose(bart_recon, [2, 1, 0])
        bart_recon = np.expand_dims(bart_recon, axis=-1)

    return bart_recon


def calculate_metrics(output, bart_test, truth):
    cs_psnr = []
    cs_nrmse = []
    cs_ssim = []
    output_psnr = []
    output_nrmse = []
    output_ssim = []

    psnr, nrmse, ssim = metrics.compute_all(truth, output, sos_axis=-1)
    output_psnr.append(psnr)
    output_nrmse.append(nrmse)
    output_ssim.append(ssim)


def _create_summary(sense_place, ks_place, im_out_place, im_truth_place):
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
