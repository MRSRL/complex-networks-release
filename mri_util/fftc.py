try:
    import pyfftw.interfaces.numpy_fft as fft
except:
    from numpy import fft
import numpy as np


def ifftnc(x, axes):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.ifftn(tmp, axes=axes)
    return fft.ifftshift(tmp, axes=axes)


def fftnc(x, axes):
    tmp = fft.fftshift(x, axes=axes)
    tmp = fft.fftn(tmp, axes=axes)
    return fft.ifftshift(tmp, axes=axes)


def fftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return fftnc(x, (axis,)) / scale


def ifftc(x, axis=0, do_orthonorm=True):
    if do_orthonorm:
        scale = np.sqrt(x.shape[axis])
    else:
        scale = 1.0
    return ifftnc(x, (axis,)) * scale


def fft2c(x, order="C", do_orthonorm=True):
    if order == "C":
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return fftnc(x, (-2, -1)) / scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return fftnc(x, (0, 1)) / scale


def ifft2c(x, order="C", do_orthonorm=True):
    if order == "C":
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[-2:]))
        else:
            scale = 1.0
        return ifftnc(x, (-2, -1)) * scale
    else:
        if do_orthonorm:
            scale = np.sqrt(np.prod(x.shape[:2]))
        else:
            scale = 1.0
        return ifftnc(x, (0, 1)) * scale


def fft3c(x, order="C"):
    if order == "C":
        return fftnc(x, (-3, -2, -1))
    else:
        return fftnc(x, (0, 1, 2))


def ifft3c(x, order="C"):
    if order == "C":
        return ifftnc(x, (-3, -2, -1))
    else:
        return ifftnc(x, (0, 1, 2))
