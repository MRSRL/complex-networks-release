"""Data preparation for training."""
import os
import random
import shutil
import subprocess
import zipfile

import numpy as np
import tensorflow as tf
import wget

from mri_util import cfl, fftc, tf_util

tf.logging.set_verbosity(tf.logging.ERROR)

BIN_BART = "bart"


def download_dataset_knee(dir_out, dir_tmp="tmp", verbose=False, do_cleanup=True):
    """Download and unzip knee dataset from mridata.org."""
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)
    if os.path.isdir(dir_tmp):
        print("WARNING! Temporary folder exists (%s)" % dir_tmp)
    else:
        os.makedirs(dir_tmp)

    num_data = 20
    for i in range(num_data):
        if verbose:
            print("Processing data (%d)..." % i)

        url = "http://old.mridata.org/knees/fully_sampled/p%d/e1/s1/P%d.zip" % (
            i + 1,
            i + 1,
        )
        dir_name_i = os.path.join(dir_out, "data%02d" % i)

        if verbose:
            print("  dowloading from %s..." % url)
        if not os.path.isdir(dir_name_i):
            os.makedirs(dir_name_i)
        file_download = wget.download(url, out=dir_tmp)

        if verbose:
            print("  unzipping contents to %s..." % dir_name_i)
        with zipfile.ZipFile(file_download, "r") as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                file_src = zip_ref.open(member)
                file_dest = open(os.path.join(dir_name_i, filename), "wb")
                with file_src, file_dest:
                    shutil.copyfileobj(file_src, file_dest)

    if do_cleanup:
        if verbose:
            print("Cleanup...")
        shutil.rmtree(dir_tmp)

    if verbose:
        print("Done")


def create_masks(
    dir_out,
    shape_y=320,
    shape_z=256,
    verbose=False,
    acc_y=(1, 2, 3),
    acc_z=(1, 2, 3),
    shape_calib=1,
    variable_density=False,
    num_repeat=4,
):
    """Create sampling masks using BART."""
    flags = ""
    file_fmt = "mask_%0.1fx%0.1f_c%d_%02d"
    if variable_density:
        flags = flags + " -v "
        file_fmt = file_fmt + "_vd"

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    for a_y in acc_y:
        for a_z in acc_z:
            if a_y * a_z != 1:
                num_repeat_i = num_repeat
                if (a_y == acc_y[-1]) and (a_z == acc_z[-1]):
                    num_repeat_i = num_repeat_i * 2
                for i in range(num_repeat_i):
                    random_seed = 1e6 * random.random()
                    file_name = file_fmt % (a_y, a_z, shape_calib, i)
                    if verbose:
                        print("creating mask (%s)..." % file_name)
                    file_name = os.path.join(dir_out, file_name)
                    cmd = "%s poisson -C %d -Y %d -Z %d -y %d -z %d -s %d %s %s" % (
                        BIN_BART,
                        shape_calib,
                        shape_y,
                        shape_z,
                        a_y,
                        a_z,
                        random_seed,
                        flags,
                        file_name,
                    )
                    subprocess.check_output(["bash", "-c", cmd])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def setup_data_tfrecords(
    dir_in_root,
    dir_out,
    data_divide=(0.75, 0.05, 0.2),
    min_shape=[80, 180],
    num_maps=1,
    crop_maps=False,
    verbose=False,
):
    """Setups training data as tfrecords.

    prep_data.setup_data('/mnt/raid3/data/Studies_DCE/recon-ccomp6/',
        '/mnt/raid3/jycheng/Project/deepspirit/data/train/', verbose=True)
    """

    # Check for two echos in here
    # Use glob to find if have echo01

    if verbose:
        print("Directory names:")
        print("  Input root:  %s" % dir_in_root)
        print("  Output root: %s" % dir_out)

    file_kspace = "kspace"
    file_sensemap = "sensemap"

    case_list = os.listdir(dir_in_root)
    random.shuffle(case_list)
    num_cases = len(case_list)

    i_train_1 = np.round(data_divide[0] * num_cases).astype(int)
    i_validate_0 = i_train_1 + 1
    i_validate_1 = np.round(data_divide[1] * num_cases).astype(int) + i_validate_0

    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, "train")):
        os.mkdir(os.path.join(dir_out, "train"))
    if not os.path.exists(os.path.join(dir_out, "validate")):
        os.mkdir(os.path.join(dir_out, "validate"))
    if not os.path.exists(os.path.join(dir_out, "test")):
        os.mkdir(os.path.join(dir_out, "test"))

    i_case = 0
    for case_name in case_list:
        file_kspace_i = os.path.join(dir_in_root, case_name, file_kspace)
        file_sensemap_i = os.path.join(dir_in_root, case_name, file_sensemap)

        if i_case < i_train_1:
            dir_out_i = os.path.join(dir_out, "train")
        elif i_case < i_validate_1:
            dir_out_i = os.path.join(dir_out, "validate")
        else:
            dir_out_i = os.path.join(dir_out, "test")

        if verbose:
            print("Processing [%d] %s..." % (i_case, case_name))
        i_case = i_case + 1

        kspace = np.squeeze(cfl.read(file_kspace_i))
        if (min_shape is None) or (
            min_shape[0] <= kspace.shape[1] and min_shape[1] <= kspace.shape[2]
        ):
            if verbose:
                print("  Slice shape: (%d, %d)" % (kspace.shape[1], kspace.shape[2]))
                print("  Num channels: %d" % kspace.shape[0])
            shape_x = kspace.shape[-1]
            kspace = fftc.ifftc(kspace, axis=-1)
            kspace = kspace.astype(np.complex64)

            # if shape_c_out < shape_c:
            #     if verbose:
            #         print("  applying coil compression (%d -> %d)..." %
            #               (shape_c, shape_c_out))
            #     shape_cal = 24
            #     ks_cal = recon.crop(ks, [-1, shape_cal, shape_cal, -1])
            #     ks_cal = np.reshape(ks_cal, [shape_c,
            #                                  shape_cal*shape_cal,
            #                                  shape_x])
            #     cc_mat = coilcomp.calc_gcc_weights_c(ks_cal, shape_c_out)
            #     ks_cc = np.reshape(ks, [shape_c, -1, shape_x])
            #     ks_cc = coilcomp.apply_gcc_weights_c(ks_cc, cc_mat)
            #     ks = np.reshape(ks_cc, [shape_c_out, shape_z, shape_y, shape_x])

            cmd_flags = ""
            if crop_maps:
                cmd_flags = cmd_flags + " -c 1e-9"
            cmd_flags = cmd_flags + (" -m %d" % num_maps)
            cmd = "%s ecalib %s %s %s" % (
                BIN_BART,
                cmd_flags,
                file_kspace_i,
                file_sensemap_i,
            )
            if verbose:
                print("  Estimating sensitivity maps (bart espirit)...")
                print("    %s" % cmd)
            subprocess.check_call(["bash", "-c", cmd])
            sensemap = np.squeeze(cfl.read(file_sensemap_i))
            sensemap = np.expand_dims(sensemap, axis=0)
            sensemap = sensemap.astype(np.complex64)

            if verbose:
                print("  Creating tfrecords (%d)..." % shape_x)
            for i_x in range(shape_x):
                file_out = os.path.join(
                    dir_out_i, "%s_x%03d.tfrecords" % (case_name, i_x)
                )
                kspace_x = kspace[:, :, :, i_x]
                sensemap_x = sensemap[:, :, :, :, i_x]

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "name": _bytes_feature(str.encode(case_name)),
                            "xslice": _int64_feature(i_x),
                            "ks_shape_x": _int64_feature(kspace.shape[3]),
                            "ks_shape_y": _int64_feature(kspace.shape[2]),
                            "ks_shape_z": _int64_feature(kspace.shape[1]),
                            "ks_shape_c": _int64_feature(kspace.shape[0]),
                            "map_shape_x": _int64_feature(sensemap.shape[4]),
                            "map_shape_y": _int64_feature(sensemap.shape[3]),
                            "map_shape_z": _int64_feature(sensemap.shape[2]),
                            "map_shape_c": _int64_feature(sensemap.shape[1]),
                            "map_shape_m": _int64_feature(sensemap.shape[0]),
                            "ks": _bytes_feature(kspace_x.tostring()),
                            "map": _bytes_feature(sensemap_x.tostring()),
                        }
                    )
                )

                tf_writer = tf.python_io.TFRecordWriter(file_out)
                tf_writer.write(example.SerializeToString())
                tf_writer.close()


def process_tfrecord(example, num_channels=None, num_emaps=None):
    """Process TFRecord to actual tensors."""
    features = tf.parse_single_example(
        example,
        features={
            "name": tf.FixedLenFeature([], tf.string),
            "xslice": tf.FixedLenFeature([], tf.int64),
            "ks_shape_x": tf.FixedLenFeature([], tf.int64),
            "ks_shape_y": tf.FixedLenFeature([], tf.int64),
            "ks_shape_z": tf.FixedLenFeature([], tf.int64),
            "ks_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_x": tf.FixedLenFeature([], tf.int64),
            "map_shape_y": tf.FixedLenFeature([], tf.int64),
            "map_shape_z": tf.FixedLenFeature([], tf.int64),
            "map_shape_c": tf.FixedLenFeature([], tf.int64),
            "map_shape_m": tf.FixedLenFeature([], tf.int64),
            "ks": tf.FixedLenFeature([], tf.string),
            "map": tf.FixedLenFeature([], tf.string),
        },
    )

    name = features["name"]
    xslice = tf.cast(features["xslice"], dtype=tf.int32)
    # shape_x = tf.cast(features['shape_x'], dtype=tf.int32)
    ks_shape_y = tf.cast(features["ks_shape_y"], dtype=tf.int32)
    ks_shape_z = tf.cast(features["ks_shape_z"], dtype=tf.int32)
    if num_channels is None:
        ks_shape_c = tf.cast(features["ks_shape_c"], dtype=tf.int32)
    else:
        ks_shape_c = num_channels
    map_shape_y = tf.cast(features["map_shape_y"], dtype=tf.int32)
    map_shape_z = tf.cast(features["map_shape_z"], dtype=tf.int32)
    if num_channels is None:
        map_shape_c = tf.cast(features["map_shape_c"], dtype=tf.int32)
    else:
        map_shape_c = num_channels
    if num_emaps is None:
        map_shape_m = tf.cast(features["map_shape_m"], dtype=tf.int32)
    else:
        map_shape_m = num_emaps

    with tf.name_scope("kspace"):
        ks_record_bytes = tf.decode_raw(features["ks"], tf.float32)
        image_shape = [ks_shape_c, ks_shape_z, ks_shape_y]
        ks_x = tf.reshape(ks_record_bytes, image_shape + [2])
        ks_x = tf_util.channels_to_complex(ks_x)
        ks_x = tf.reshape(ks_x, image_shape)

    with tf.name_scope("sensemap"):
        map_record_bytes = tf.decode_raw(features["map"], tf.float32)
        map_shape = [map_shape_m * map_shape_c, map_shape_z, map_shape_y]
        map_x = tf.reshape(map_record_bytes, map_shape + [2])
        map_x = tf_util.channels_to_complex(map_x)
        map_x = tf.reshape(map_x, map_shape)

    return name, xslice, ks_x, map_x


def read_tfrecord_with_sess(tf_sess, filename_tfrecord):
    """Read TFRecord for debugging."""
    tf_reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename_tfrecord])
    _, serialized_example = tf_reader.read(filename_queue)
    name, xslice, ks_x, map_x = process_tfrecord(serialized_example)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=tf_sess, coord=coord)
    name, xslice, ks_x, map_x = tf_sess.run([name, xslice, ks_x, map_x])
    coord.request_stop()
    coord.join(threads)

    return {"name": name, "xslice": xslice, "ks": ks_x, "sensemap": map_x}


def read_tfrecord(filename_tfrecord):
    """Read TFRecord for debugging."""
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=session_config)
    data = read_tfrecord_with_sess(tf_sess, filename_tfrecord)
    tf_sess.close()
    return data
