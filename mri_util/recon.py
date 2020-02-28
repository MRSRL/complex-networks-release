"""Basic MRI reconstruction functions."""
import numpy as np


def sumofsq(im, axis=0):
    """Compute square root of sum of squares.

    :param im: raw image
    """
    if axis < 0:
        axis = im.ndim - 1
    if axis > im.ndim:
        print("ERROR! Dimension %d invalid for given matrix" % axis)
        return -1

    out = np.sqrt(np.sum(im.real * im.real + im.imag * im.imag, axis=axis))

    return out


def phasecontrast(im, ref, axis=-1, coilaxis=-1):
    """Compute phase contrast."""
    if axis < 0:
        axis = im.ndim - 1

    out = np.conj(ref) * im
    if coilaxis >= 0:
        out = np.sum(out, axis=coilaxis)
    out = np.angle(out)

    return out


def fftmod(im, axis=-1):
    """Apply 1 -1 modulation along dimension specified by axis"""
    if axis < 0:
        axis = im.ndim - 1

    # generate modulation kernel
    dims = im.shape
    mod = np.ones(
        np.append(dims[axis], np.ones(len(dims) - 1, dtype=int)), dtype=im.dtype
    )
    mod[1 : dims[axis] : 2] = -1
    mod = np.transpose(mod, np.append(np.arange(1, len(dims)), 0))

    # apply kernel
    tpdims = np.concatenate(
        (np.arange(0, axis), np.arange(axis + 1, len(dims)), [axis])
    )
    out = np.transpose(im, tpdims)  # transpose for broadcasting
    out = out * mod
    tpdims = np.concatenate(
        (np.arange(0, axis), [len(dims) - 1], np.arange(axis, len(dims) - 1))
    )
    out = np.transpose(out, tpdims)  # transpose back to original dims

    return out


def crop_in_dim(im, shape, dim):
    """Centered crop of image."""
    if dim < 0 or dim >= im.ndim:
        print("ERROR! Invalid dimension specified!")
        return im
    if shape > im.shape[dim]:
        print("ERROR! Invalid shape specified!")
        return im

    im_shape = im.shape
    tmp_shape = [
        int(np.prod(im_shape[:dim])),
        im_shape[dim],
        int(np.prod(im_shape[(dim + 1) :])),
    ]
    im_out = np.reshape(im, tmp_shape)
    ind0 = (im_shape[dim] - shape) // 2
    ind1 = ind0 + shape
    im_out = im_out[:, ind0:ind1, :].copy()
    im_out = np.reshape(im_out, im_shape[:dim] + (shape,) + im_shape[(dim + 1) :])
    return im_out


def crop(im, out_shape, verbose=False):
    """Centered crop."""
    if im.ndim != np.size(out_shape):
        print("ERROR! Num dim of input image not same as desired shape")
        print("   %d != %d" % (im.ndim, np.size(out_shape)))
        return []

    im_out = im
    for i in range(np.size(out_shape)):
        if out_shape[i] > 0:
            if verbose:
                print("Crop [%d]: %d to %d" % (i, im_out.shape[i], out_shape[i]))
            im_out = crop_in_dim(im_out, out_shape[i], i)

    return im_out


def zeropad(im, out_shape):
    """Zeropad image."""
    if im.ndim != np.size(out_shape):
        print("ERROR! Num dim of input image not same as desired shape")
        print("   %d != %d" % (im.ndim, np.size(out_shape)))
        return im

    pad_shape = []
    for i in range(np.size(out_shape)):
        if out_shape[i] == -1:
            pad_shape_i = [0, 0]
        else:
            pad_start = int((out_shape[i] - im.shape[i]) / 2)
            pad_end = out_shape[i] - im.shape[i] - pad_start
            pad_shape_i = [pad_start, pad_end]

        pad_shape = pad_shape + [pad_shape_i]

    im_out = np.pad(im, pad_shape, "constant")

    return im_out
