from utils.image import Image
from utils.helper import RegError
from utils import helper
from utils.resampler import FieldsComposer
import numpy as np
import nibabel as nib
import math


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

    def exponentiate(self, disp_image):
        """
        Compute the exponential of this velocity field using the
        scaling and squaring approach.

        :param disp_image: Displacement field image that will be
        updated with the exponentiated velocity field.
        """

        self.__do_init_check()

        data = self.field.data
        header = self.field.get_header()

        norm = np.linalg.norm(data)
        max_norm = np.max(norm[:])

        if max_norm <= 0:
            raise ValueError('Maximum norm is invalid.')

        pix_dims = np.asarray(header.get_zooms())
        # ignore NULL dimensions
        min_size = np.min(pix_dims[pix_dims > 0])
        num_steps = np.ceil(np.log2(max_norm / (min_size / 2))).astype('int')

        # Approximate the initial exponential
        init = 1 << num_steps
        disp_image.data = data / init

        # Generate temporary array
        temp_data = np.zeros(disp_image.data.shape)
        temp_nim = nib.Nifti1Image(temp_data, affine=None,
                                   header=disp_image.header)

        dfc = FieldsComposer()
        loop_limit = math.log(num_steps, 2)
        # Do the squaring step to perform the integration
        for _ in range(0, loop_limit):
            dfc.compose(disp_image, disp_image, temp_nim)
            disp_image = temp_nim
