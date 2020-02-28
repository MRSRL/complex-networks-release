"""Wraps BART functions."""
import random
import subprocess
from timeit import default_timer as timer

import numpy as np

from mri_util import cfl, recon


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
    random_seed = 1e6 * random.random()
    flags = "-Z %d -Y %d -z %g -y %g -C %d -s %d" % (
        shape[0],
        shape[1],
        acc[0],
        acc[1],
        shape_calib,
        random_seed,
    )
    if variable_density:
        flags = flags + " -v"
    cmd = "bart poisson %s %s" % (flags, tmp_file)
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])
    mask = np.abs(np.squeeze(cfl.read(tmp_file)))
    return mask


def bart_espirit(
    ks_input,
    shape=None,
    verbose=False,
    shape_e=2,
    crop_value=None,
    cal_size=None,
    smooth=False,
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
        ks_input = recon.zeropad(ks_input, [-1, -1, shape[0], shape[1], -1])

    flags = ""
    if crop_value is not None:
        flags = flags + "-c %f " % crop_value
    if cal_size is not None:
        flags = flags + "-r %d " % cal_size
    if smooth:
        flags = flags + "-S "

    cfl.write(filename_ks_tmp, ks_input)
    cmd = "bart ecalib -m %d %s %s %s" % (
        shape_e,
        flags,
        filename_ks_tmp,
        filename_map_tmp,
    )
    if verbose:
        print("  %s" % cmd)
    time_start = timer()
    subprocess.check_output(["bash", "-c", cmd])
    time_end = timer()
    sensemap = cfl.read(filename_map_tmp)
    return sensemap, time_end - time_start


def bart_pics(
    ks_input,
    verbose=False,
    sensemap=None,
    shape_e=2,
    do_cs=True,
    do_imag_reg=False,
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
        cmd = "bart ecalib -m %d -c 1e-9 %s %s" % (
            shape_e,
            filename_ks_tmp,
            filename_map_tmp,
        )
        if verbose:
            print("  %s" % cmd)
        subprocess.check_output(["bash", "-c", cmd])
    else:
        cfl.write(filename_map_tmp, sensemap)
    if do_cs:
        flags = "-l1 -r 1e-2"
    else:
        flags = "-l2 -r 1e-2"
    if do_imag_reg:
        flags = flags + " -R R1:7:1e-1"

    cmd = "bart pics %s -S %s %s %s" % (
        flags,
        filename_ks_tmp,
        filename_map_tmp,
        filename_im_tmp,
    )
    if verbose:
        print("  %s" % cmd)
    subprocess.check_output(["bash", "-c", cmd])

    cmd = "bart fakeksp -r %s %s %s %s" % (
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
