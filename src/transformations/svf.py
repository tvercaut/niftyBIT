__author__ = 'Pankaj Daga'

from utils.image import Image
from utils.helper import RegError
from utils import helper
from utils.resampler import DisplacementFieldComposer
import numpy as np


class SVF(object):
    """
    Python class for operations on stationary velocity fields.
    """

    def __init__(self):
        self.field = None

    def __do_init_check(self):
        """
        Checks if the underlying field object is initialised or not
        """
        if self.field is None:
            raise RegError('Field is not initialised. init_field() must'
                           'be called before the object can be used.')

    def save(self, filepath):
        """
        Save the field as a Nifti image
        :param filepath: Full path and filename for the saved file
        """
        self.__do_init_check()
        self.field.save(filepath)

    def init_field(self, target, affine=None):
        """
        Initialise velocity field image header from the specified target image.
        Sets the data to 0.

        Parameters:
        -----------
        :param target: The target image. Mandatory.
        :param affine: The initial affine transformation
        """
        self.field = helper.initialise_field(target, affine)

    def exponentiate(self):
        """
        Compute the exponential of this velocity field using the
        scaling and squaring approach.

        The velocity field is in the tangent space of the manifold and the
        displacement field is the actual manifold and the transformation
        between the velocity field and the displacement field is given by
        the exponential chart.

        :param disp_image: Displacement field image that will be
        updated with the exponentiated velocity field.
        """

        self.__do_init_check()
        data = self.field.data
        result_data = np.zeros(self.field.data.shape)
        result = Image.from_data(result_data, self.field.get_header())

        # Important: Need to specify which axes to use
        norm = np.linalg.norm(data, axis=data.ndim-1)
        max_norm = np.max(norm[:])

        if max_norm < 0:
            raise ValueError('Maximum norm is invalid.')
        if max_norm == 0:
            return result

        pix_dims = np.asarray(self.field.zooms)
        # ignore NULL dimensions
        min_size = np.min(pix_dims[pix_dims > 0])
        num_steps = max(0,np.ceil(np.log2(max_norm / (min_size / 2))).astype('int'))

        # Approximate the initial exponential
        init = 1 << num_steps
        result.data = data / init

        dfc = DisplacementFieldComposer()
        # Do the squaring step to perform the integration
        # The exponential is num_steps times recursive composition of
        # the field with itself, which is equivalant to integration over
        # the unit interval.
        for _ in range(0, num_steps):
            result = dfc.compose(result, result)

        return result
