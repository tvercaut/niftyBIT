import numpy as np


def compute_ssd(target, warped, mask):

    """
    Compute the sum of square difference between two images.

    :rtype : float
    :param target: The target image
    :param warped: The warped image. Should be in the same space as the target
    :param mask:
    :return: The computed sum of square difference
    """
    warped_data = warped.data.astype(np.float32)
    target_data = target.data.astype(np.float32)
    masked_data = mask.data > 0
    ssd = np.sum(((warped_data - target_data)**2)*masked_data)
    return ssd
