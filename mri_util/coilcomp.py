"""Coil compression.

Reference(s):
[1] Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for
    accelerated imaging with Cartesian sampling. Magn Reson Med 2013
    Mar 9;69:571-582
"""
import os
import sys

import numpy as np

from mri_util import fftc


def calc_gcc_weights_c(ks_calib, num_virtual_channels, correction=True):
    """Calculate coil compression weights.

    Input
      ks_calib -- raw k-space data of dimensions
                    (num_channels, num_readout, num_kx)
      num_virtual_channels -- number of virtual channels to compress to
      correction -- apply rotation correction (default: True)
    Output
      cc_mat -- coil compression matrix (use apply_gcc_weights)
    """
    me = "coilcomp.calc_gcc_weights_c"

    num_kx = ks_calib.shape[2]
    # num_readout = ks_calib.shape[1]
    num_channels = ks_calib.shape[0]

    if num_virtual_channels > num_channels:
        print(
            [
                "%s> Num of virtual channels (%d) is more than the actual "
                + " channels (%d)!"
            ]
            % (me, num_virtual_channels, num_channels)
        )
        return np.eye(num_channels, dtype=np.complex64)

    if num_kx > 1:
        # find max in readout
        tmp = np.sum(np.sum(np.power(np.abs(ks_calib), 2), axis=0), axis=1)
        i_xmax = np.argmax(tmp)
        # circ shift to move max to center (make copy to not touch original data)
        ks_calib_int = np.roll(ks_calib.copy(), int(num_kx / 2 - i_xmax), axis=-1)
        ks_calib_int = fftc.ifftc(ks_calib_int, axis=-1)
    else:
        ks_calib_int = ks_calib.copy()

    cc_mat = np.zeros((num_virtual_channels, num_channels, num_kx), dtype=np.complex64)
    for i_x in range(num_kx):
        ks_calib_x = np.squeeze(ks_calib_int[:, :, i_x])
        U, s, Vh = np.linalg.svd(ks_calib_x.T, full_matrices=False)
        V = Vh.conj()
        cc_mat[:, :, i_x] = V[0:num_virtual_channels, :]

    if correction:
        for i_x in range(int(num_kx / 2) - 2, -1, -1):
            V1 = cc_mat[:, :, i_x + 1]
            V2 = cc_mat[:, :, i_x]
            A = np.matmul(V1.conj(), V2.T)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj()
            cc_mat[:, :, i_x] = np.matmul(P, cc_mat[:, :, i_x])

        for i_x in range(int(num_kx / 2) - 1, num_kx, 1):
            V1 = cc_mat[:, :, i_x - 1]
            V2 = cc_mat[:, :, i_x]
            A = np.matmul(V1.conj(), V2.T)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj()
            cc_mat[:, :, i_x] = np.matmul(P, np.squeeze(cc_mat[:, :, i_x]))

    return cc_mat


def apply_gcc_weights_c(ks, cc_mat):
    """Apply coil compression weights.

    Input
      ks -- raw k-space data of dimensions (num_channels, num_readout, num_kx)
      cc_mat -- coil compression matrix calculated using calc_gcc_weights
    Output
      ks_out -- coil compresssed data
    """
    me = "coilcomp.apply_gcc_weights_c"

    num_channels = ks.shape[0]
    num_readout = ks.shape[1]
    num_kx = ks.shape[2]
    num_virtual_channels = cc_mat.shape[0]

    if num_channels != cc_mat.shape[1]:
        print("%s> ERROR! num channels does not match!" % me)
        print("%s>   ks: num channels = %d" % (me, num_channels))
        print("%s>   cc_mat: num channels = %d" % (me, cc_mat.shape[1]))

    ks_x = fftc.ifftc(ks, axis=-1)
    ks_out = np.zeros((num_virtual_channels, num_readout, num_kx), dtype=np.complex64)
    for i_channel in range(num_virtual_channels):
        cc_mat_i = np.reshape(cc_mat[i_channel, :, :], (num_channels, 1, num_kx))
        ks_out[i_channel, :, :] = np.sum(ks_x * cc_mat_i, axis=0)
    ks_out = fftc.fftc(ks_out, axis=-1)

    return ks_out


def calc_gcc_weights(ks_calib, num_virtual_channels, correction=True):
    """Calculate coil compression weights.

    Input
      ks_calib -- raw k-space data of dimensions (num_kx, num_readout, num_channels)
      num_virtual_channels -- number of virtual channels to compress to
      correction -- apply rotation correction (default: True)
    Output
      cc_mat -- coil compression matrix (use apply_gcc_weights)
    """

    me = "coilcomp.calc_gcc_weights"

    num_kx = ks_calib.shape[0]
    # num_readout = ks_calib.shape[1]
    num_channels = ks_calib.shape[2]

    if num_virtual_channels > num_channels:
        print(
            "%s> Num of virtual channels (%d) is more than the actual channels (%d)!"
            % (me, num_virtual_channels, num_channels)
        )
        return np.eye(num_channels, dtype=complex)

    # find max in readout
    tmp = np.sum(np.sum(np.power(np.abs(ks_calib), 2), axis=2), axis=1)
    i_xmax = np.argmax(tmp)
    # circ shift to move max to center (make copy to not touch original data)
    ks_calib_int = np.roll(ks_calib.copy(), int(num_kx / 2 - i_xmax), axis=0)
    ks_calib_int = fftc.ifftc(ks_calib_int, axis=0)

    cc_mat = np.zeros((num_kx, num_channels, num_virtual_channels), dtype=complex)
    for i_x in range(num_kx):
        ks_calib_x = np.squeeze(ks_calib_int[i_x, :, :])
        U, s, Vh = np.linalg.svd(ks_calib_x, full_matrices=False)
        V = Vh.conj().T
        cc_mat[i_x, :, :] = V[:, 0:num_virtual_channels]

    if correction:
        for i_x in range(int(num_kx / 2) - 2, -1, -1):
            V1 = cc_mat[i_x + 1, :, :]
            V2 = cc_mat[i_x, :, :]
            A = np.matmul(V1.conj().T, V2)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj().T
            cc_mat[i_x, :, :] = np.matmul(cc_mat[i_x, :, :], P)

        for i_x in range(int(num_kx / 2) - 1, num_kx, 1):
            V1 = cc_mat[i_x - 1, :, :]
            V2 = cc_mat[i_x, :, :]
            A = np.matmul(V1.conj().T, V2)
            Ua, sa, Vah = np.linalg.svd(A, full_matrices=False)
            P = np.matmul(Ua, Vah)
            P = P.conj().T
            cc_mat[i_x, :, :] = np.matmul(np.squeeze(cc_mat[i_x, :, :]), P)

    return cc_mat


def apply_gcc_weights(ks, cc_mat):
    """ Apply coil compression weights
    Input
      ks -- raw k-space data of dimensions (num_kx, num_readout, num_channels)
      cc_mat -- coil compression matrix calculated using calc_gcc_weights
    Output
      ks_out -- coil compresssed data
    """

    me = "coilcomp.apply_gcc_weights"

    if ks.shape[2] != cc_mat.shape[1]:
        print("%s> ERROR! num channels does not match!" % me)
        print("%s>   ks: num channels = %d" % (me, ks.shape[2]))
        print("%s>   cc_mat: num channels = %d" % (me, cc_mat.shape[1]))

    num_kx = ks.shape[0]
    num_readout = ks.shape[1]
    num_channels = ks.shape[2]
    num_virtual_channels = cc_mat.shape[2]

    ks_x = fftc.ifftc(ks, axis=0)
    ks_out = np.zeros((num_kx, num_readout, num_virtual_channels), dtype=complex)
    for i_channel in range(num_virtual_channels):
        cc_mat_i = np.reshape(cc_mat[:, :, i_channel], (num_kx, 1, num_channels))
        ks_out[:, :, i_channel] = np.sum(ks_x * cc_mat_i, axis=2)
    ks_out = fftc.fftc(ks_out, axis=0)

    return ks_out
