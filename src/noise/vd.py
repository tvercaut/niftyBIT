import numpy as np
from utils import helper
import math


def estimate_vd(residual, mask=None):
    # Make the data zero mean, unit variance.
    """
    Estimate the virtual decimation factor.
    :param residual: Residual image. Should be a numpy nd-array
    :param mask: Image mask. If none, whole image is used.
    """
    # Make the data zero mean, unit variance
    if mask is not None:
        md = mask > 0
        local_image = np.subtract(residual, np.mean(residual[md]))
        local_image = np.divide(local_image, np.std(local_image[md]))
    else:
        local_image = np.subtract(residual, np.mean(residual))
        local_image = np.divide(local_image, np.std(local_image))

    shape = local_image.shape
    shape_len = len(shape)
    covas = np.zeros(shape_len, dtype=np.float64)
    sigma = np.zeros(shape_len, dtype=np.float64)

    for i in range(shape_len):
        shape_temp = np.array(shape)
        shape_temp[i] -= 1
        p = helper.ndmesh(*[np.arange(0, shape_temp[l])
                          for l in range(shape_len)])

        shape_temp = np.zeros(shape_len, dtype=np.int)
        shape_temp[i] = 1
        c = helper.ndmesh(*[np.arange(shape_temp[l], shape[l])
                          for l in range(shape_len)])

        if mask is not None:
            covas[i] = np.sum(local_image[p] * local_image[c] * mask[c])
            sigma[i] = 0.5 * np.sum((local_image[p] * local_image[p] +
                                    local_image[c] * local_image[c])
                                    * mask[c])
        else:
            covas[i] = np.sum(local_image[p] * local_image[c])
            sigma[i] = 0.5 * np.sum((local_image[p] * local_image[p] +
                                    local_image[c] * local_image[c]))

    # resels = volume/FWHM
    # VD = (resels * sqrt(4 * log(2)/pi))/volume = sqrt(4 * log(2)/pi)/FWHM
    fac = 1.0
    fwhm = 1.0
    log2 = math.log(2)
    for i in range(shape_len):
        fwhm *= math.sqrt((-2.0 * log2)/math.log(covas[i]/sigma[i]))
        fac *= 0.9394  # sqrt(4 * log(2)/pi) = 0.9394

    return fac/fwhm