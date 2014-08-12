import numpy as np
from utils import image as img


class RegError(Exception):
    """
    Exception class override
    """
    def __init__(self, v):
        super(RegError, self).__init__(v)
        self.message = v

    def __str__(self):
        return repr(self.message)


def generate_identity_deformation(image, def_image):
    """
    Helper method to generate an identity transform

    :param image: The image whose geometry we will use
    :param def_image: The deformation field image. if it is none,
    an appropriate image file will be created and returned
    """

    # Matrix to go from voxel to world space
    voxel_2_xyz = image.voxel_2_mm
    vol_ext = image.vol_ext
    voxels = np.mgrid[[slice(i) for i in vol_ext]]
    voxels = [d.reshape(vol_ext, order='F') for d in voxels]
    mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
           for k in range(len(voxels)))
           for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

    data = np.squeeze(def_image.data)
    for i in range(data.shape[-1]):
        data[..., i] = mms[i]


def field_conversion_method(image, field_image, is_deformation=True):

    data = np.zeros_like(field_image.data, dtype=np.float32)
    field = img.Image.from_data(data, image.get_header())
    # Matrix to go from voxel to world space
    voxel_2_xyz = image.voxel_2_mm
    vol_ext = image.vol_ext
    voxels = np.mgrid[[slice(i) for i in vol_ext]]
    voxels = [d.reshape(vol_ext, order='F') for d in voxels]
    mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
           for k in range(len(voxels)))
           for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

    input_data = np.squeeze(field_image.data)
    field_data = np.squeeze(data)
    if is_deformation:
        for i in range(data.shape[-1]):
            # Output is the displacement field
            field_data[..., i] = input_data[..., i] - mms[i]
    else:
        for i in range(data.shape[-1]):
            # Output is the deformation/position field
            field_data[..., i] = input_data[..., i] + mms[i]

    return field


def generate_displacement_from_deformation(image, def_image):
    """
    Generate the displacement field image from deformation field image
    :rtype : Image
    :param image: Image whose geometry we will use.
    :param def_image: The input deformation field image
    :return: Displacement field image
    """

    return field_conversion_method(image, def_image)


def generate_deformation_from_displacement(image, disp_image):
    """
    Generate the deformation field image from displacement fild image
    :rtype : Image
    :param image: Image whose geometry we will use.
    :param disp_image: The input displacement field image
    :return: Deformation field image
    """

    return field_conversion_method(image, disp_image, False)


def read_affine_transformation(input_aff):
    """
    Read an affine transformation file.
    The function expects a 4x4 transformation matrix.

    :rtype : Numpy 4x4 matrix
    :param input_aff: File object or file path

    Returns a numpy affine transformation matrix.
    """

    if isinstance(input_aff, basestring):
        file_obj = open(input_aff, 'r')
        matrix = read_affine_transformation(file_obj)
        file_obj.close()
        return matrix

    elif isinstance(input_aff, file):
        file_content = input_aff.read().strip()
        file_content = file_content.replace('\r\n', ';')
        file_content = file_content.replace('\n', ';')
        file_content = file_content.replace('\r', ';')

        mat = np.matrix(file_content)
        if mat.shape != (4, 4):
            raise RegError('Input affine transformation '
                           'should be a 4x4 matrix.')
        return mat

    raise TypeError('Input must be a file object or a file name.')


def is_power2(num):
    """
    Check if a number is power of 2

    :param num: Input integer
    :rtype: Boolean

    Returns true if the input is a power of 2, else false
    """

    return num != 0 and ((num & (num - 1)) == 0)


def compute_variance(array):
    """
    Compute the variance of a numpy array. This can also be a masked array
    :param array: A numpy (masked) array
    :return: The computed variance
    """
    return np.ma.var(array)
