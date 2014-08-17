import numpy as np


def compute_ssd(target, warped, mask=None):

    """
    Compute the sum of square difference between two images.
    :rtype : float64
    :param target: The target image
    :param warped: The warped image. Should be in the same space as the target
    :param mask: The image mask. Can be none
    Only voxels under the mask are taken into account

    :return: The computed sum of square difference
    """
    warped_data = warped.data.astype(np.float64)
    target_data = target.data.astype(np.float64)

    if mask is not None:
        valid_voxels = np.count_nonzero(mask.data)
        md = mask.data > 0
        ssd = np.sum((warped_data[md] - target_data[md])**2)/valid_voxels
    else:
        ssd = np.sum((warped_data - target_data)**2)

    return ssd
