import numpy as np


class RegError(Exception):
    """
    Exception class override
    """
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return repr(self.value)


def generate_identity_deformation(image, def_image):
    """
    Helper method to generate an identity transform

    :param image: The image whose geometry we will use
    :param def_image: The field holding the transformation. Must have the correct header
    in terms of spacing and rigid/affine transformations
    """
    # Matrix to go from world to voxel space
    voxel_2_xyz = image.voxel_2_mm
    vol_ext = image.vol_ext

    voxels = np.mgrid[[slice(i) for i in vol_ext]]
    voxels = [d.reshape(vol_ext, order='F') for d in voxels]

    mms = [voxel_2_xyz[i][3] + sum(voxel_2_xyz[i][k] * voxels[k]
        for k in range(len(voxels))) for i in range(len(voxel_2_xyz) - (4 - len(vol_ext)))]

    data = np.squeeze(def_image.data)
    for i in range(data.shape[-1]):
        data[..., i] = mms[i]


def read_affine_transformation(f):
    """
    Read an affine transformation file. The function expects a 4x4 transformation
    matrix.

    :rtype : Numpy 4x4 matrix
    :param f: File object or file path

    Returns a numpy affine transformation matrix.
    """

    if isinstance(f, basestring):
        fo = open(f, 'r')
        matrix = read_affine_transformation(fo)
        fo.close()
        return matrix

    elif isinstance(f, file):
        file_content = f.read().strip()
        file_content = file_content.replace('\r\n', ';')
        file_content = file_content.replace('\n', ';')
        file_content = file_content.replace('\r', ';')

        mat = np.matrix(file_content)
        if mat.shape != (4, 4):
            raise RegError('Input affine transformation should be a 4x4 matrix.')
        return mat

    raise TypeError('f must be a file object or a file name.')


def is_power2(num):
    """
    Check if a number is power of 2

    :param num: Input integer
    :rtype: Boolean

    Returns true if the input is a power of 2, else false
    """
    return num != 0 and ((num & (num - 1)) == 0)
