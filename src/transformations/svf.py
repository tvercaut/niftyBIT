from utils import *
import numpy as np
import nibabel as nib
import math


class SVF:
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
            raise RegError('Field is not initialised. init_field() must be called '
                           'before the object can be used.')

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
        vol_ext = target.vol_ext
        dims = list()
        dims.extend(vol_ext)
        while len(dims) < 4:
            dims.extend([1])
        dims.extend([len(vol_ext)])

        # Inititalise with zero displacement
        data = np.zeros(dims, dtype=np.float32)

        if affine is None:
            # No initial affine. Just use target header
            self.field = Image.from_data(data, target.get_header())
        else:
            if affine.shape != (4, 4):
                raise RegError('Input affine transformation should be a 4x4 matrix.')

            mod_hdr = target.get_header()

            [sform, code] = mod_hdr.get_sform(True)
            if code > 0:
                t = affine * sform
                mod_hdr.setSForm(t)
            else:
                t = affine * mod_hdr.get_qform()
                mod_hdr.set_sform(t, 1)

            # Inititalise the velocity field in the transformed space
            self.field = Image.from_data(data, header=mod_hdr)

    def exponentiate(self, disp_image):
        """
        Compute the exponential of this velocity field using the scaling and squaring approach.

        :param disp_image: Displacement field image that will be updated with the exponentiated
        velocity field.
        """

        self.__do_init_check()

        data = self.field.get_data()
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
        c = 1 << num_steps
        disp_image.data = data / c

        # Generate temporary array
        temp_data = np.zeros(disp_image.data.shape)
        temp_nim = nib.Nifti1Image(temp_data, affine=None, header=disp_image.header)
        dfc = FieldsComposer()

        loop_limit = math.log(num_steps, 2)
        # Do the squaring step to perform the integration
        for _ in range(0, loop_limit):
            dfc.compose(disp_image, disp_image, temp_nim)
            disp_image = temp_nim
