"""Wraps BART functions."""
from __future__ import absolute_import, division, print_function

import os
import subprocess

import numpy as np

from packages.fileio import cfl
from packages.mrirecon import recon

BIN_BART = "bart"


def bart_generate_mask(
    shape,
    acc,
    variable_density=True,
    shape_calib=10,
    verbose=False,
    tmp_file="mask.tmp",
):
    """Use bart poisson to generate masks."""
    if verbose:
        print("Generating sampling mask...")
    flags = "-Z %d -Y %d -z %g -y %g" % (shape[0], shape[1], acc[0], acc[1])
    if shape_calib > 0:
        flags = flags + (" -C %d" % shape_calib)
    if variable_density:
        flags = flags + " -v"
    cmd = "%s poisson %s %s" % (BIN_BART, flags, tmp_file)
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])
    mask = np.abs(np.squeeze(cfl.read(tmp_file)))
    return mask


def bart_espirit(
    ks_input,
    shape=None,
    verbose=False,
    filename_ks_tmp="ks.tmp",
    filename_map_tmp="map.tmp",
):
    """Estimate sensitivity maps using BART ESPIRiT.

    ks_input dimensions: [emaps, channels, kz, ky, kx]
    """
    if verbose:
        print("Estimating sensitivity maps...")
    if shape is not None:
        ks_input = recon.crop(ks_input, [-1, -1, shape[0], shape[1], -1])
    cfl.write(filename_ks_tmp, ks_input)
    cmd = "%s ecalib  %s %s" % (BIN_BART, filename_ks_tmp, filename_map_tmp)
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])
    sensemap = cfl.read(filename_map_tmp)
    return sensemap


def bart_pics(
    ks_input,
    sensemap=None,
    verbose=False,
    do_l1=True,
    filename_ks_tmp="ks.tmp",
    filename_map_tmp="map.tmp",
    filename_im_tmp="im.tmp",
    filename_ks_out_tmp="ks_out.tmp",
):
    """BART PICS reconstruction."""
    if verbose:
        print("PICS (l1-ESPIRiT) reconstruction...")

    cfl.write(filename_ks_tmp, ks_input)
    if sensemap is None:
        cmd = "%s ecalib -c 1e-9 %s %s" % (BIN_BART, filename_ks_tmp, filename_map_tmp)
        if verbose:
            print("  %s" % cmd)
        subprocess.check_output(["bash", "-c", cmd])
    else:
        cfl.write(filename_map_tmp, sensemap)

    pics_flags = ""
    if do_l1:
        pics_flags = "-l1 -r 1e-1"
    else:
        pics_flags = "-l2 -r 1e-1"
    cmd = "%s pics %s -S %s %s %s" % (
        BIN_BART,
        pics_flags,
        filename_ks_tmp,
        filename_map_tmp,
        filename_im_tmp,
    )
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])

    cmd = "%s fakeksp -r %s %s %s %s" % (
        BIN_BART,
        filename_im_tmp,
        filename_ks_tmp,
        filename_map_tmp,
        filename_ks_out_tmp,
    )
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])
    ks_pics = np.squeeze(cfl.read(filename_ks_out_tmp))
    ks_pics = np.expand_dims(ks_pics, axis=0)

    return ks_pics


def recon_dataset(raw_input, raw_output, dir_tmp=".", tag=None):
    """Recon datasets."""
    if tag is None:
        tag = "%05d" % np.random.randint(0, 1e4)
    # raw_input = np.real(raw_input)
    # raw_input = raw_input[:, :, :, ::2] + 1j * raw_input[:, :, :, 1::2]

    # raw_output = np.real(raw_output)
    # raw_output = raw_output[:, :, :, ::2] + 1j * raw_output[:, :, :, 1::2]

    file_ksin = os.path.join(dir_tmp, "ksin." + tag)
    file_imout = os.path.join(dir_tmp, "imout." + tag)
    file_kscalib = os.path.join(dir_tmp, "kscalib." + tag)
    file_map = os.path.join(dir_tmp, "map." + tag)

    im_out = np.zeros((raw_input.shape[0], 2) + raw_input.shape[1:3], dtype=np.complex)
    acc_list = np.zeros((raw_input.shape[0], 1))
    print("ESPIRiT reconstruction...")
    for i in range(raw_input.shape[0]):
        raw_input_i = raw_input[i, :, :, :]
        raw_output_i = raw_output[i, :, :, :]

        acc = np.sum(raw_output_i != 0) / np.sum(raw_input_i != 0)
        acc_list[i] = acc
        print("  [%d] Acceleration = %g" % (i, acc))

        raw_output_i = np.transpose(raw_output_i, (2, 0, 1))
        raw_output_i = np.reshape(raw_output_i, (1,) + raw_output_i.shape + (1,))
        cfl.write(file_kscalib, raw_output_i)
        cmd = "%s ecalib -c 1e-9 %s %s" % (BIN_BART, file_kscalib, file_map)
        subprocess.check_output(["bash", "-c", cmd])

        raw_input_i = np.transpose(raw_input_i, (2, 0, 1))
        raw_input_i = np.reshape(raw_input_i, (1,) + raw_input_i.shape + (1,))
        cfl.write(file_ksin, raw_input_i)
        cmd = "%s pics -l2 %s %s %s" % (BIN_BART, file_ksin, file_map, file_imout)
        subprocess.check_output(["bash", "-c", cmd])
        im_out[i, :, :, :] = np.squeeze(cfl.read(file_imout))

    return im_out, acc_list
