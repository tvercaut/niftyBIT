import numpy as np
from utils.image import Image
from utils import helper


class Affine(object):
    """
    Class for operating on affine transformations
    """
    def __init__(self):
        """
        Inititalise to an identity transformation
        """
        self.mat = np.identity(4, dtype=np.float64)
        self.field = None

    def generate_deformation(self, im):
        """
        Generate a deformation field according to the current
        transformation using the geometry of the input image.
        The image transformation is updated as
        new_transformation = affine * image_transform
        :param im: The input image whose geometry is used.
        """
        self.field = helper.initialise_field(im, self.mat)
        helper.generate_identity_deformation(self.field)



