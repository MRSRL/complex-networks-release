"""Metrics for testing."""
import numpy as np
import skimage.measure

from mri_util import recon


def compute_psnr(ref, x):
    """Compute peak to signal to noise ratio."""
    max_val = np.max(np.abs(ref))
    mse = np.mean(np.square(np.abs(x - ref)))
    psnr = 10 * np.log(np.square(max_val) / mse) / np.log(10)
    return psnr


def compute_nrmse(ref, x):
    """Compute normalized root mean square error.
    The norm of reference is used to normalize the metric.
    """
    mse = np.sqrt(np.mean(np.square(np.abs(ref - x))))
    norm = np.sqrt(np.mean(np.square(np.abs(ref))))

    return mse / norm


def compute_ssim(ref, x, sos_axis=None):
    """Compute structural similarity index metric.
    The image is first converted to magnitude image and normalized
    before the metric is computed.
    """
    ref = ref.copy()
    x = x.copy()
    if sos_axis is not None:
        x = recon.sumofsq(x, axis=sos_axis)
        ref = recon.sumofsq(ref, axis=sos_axis)
    x = np.squeeze(x)
    ref = np.squeeze(ref)
    x /= np.mean(np.square(np.abs(x)))
    ref /= np.mean(np.square(np.abs(ref)))
    return skimage.measure.compare_ssim(ref, x, data_range=x.max() - x.min())


def compute_all(ref, x, sos_axis=None):
    psnr = compute_psnr(ref, x)
    nrmse = compute_nrmse(ref, x)
    ssim = compute_ssim(ref, x, sos_axis=sos_axis)

    return psnr, nrmse, ssim
