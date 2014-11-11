__author__ = 'Pankaj Daga'

import nibabel as nib
from numpy import linalg as la
from os import path
import numpy as np


class Image(object):
    """
    A wrapper around the nibabel image implementation
    At the moment the assumption is that we only deal with nifti images
    """
    def __init__(self, image):
        """
        Load the image and set up attributes like transformation matrices
        as well as volume dimensions and time points
        :param image: nibabel nifti image object
        """
        self.__image = image
        self.__set_attributes()

    def save(self, filename):
        """
        Save the file
        :param filename: Full path and filename for the saved file
        """
        saved_image = Image.from_data(self.data, self.__image.get_header())
        name = path.expanduser(filename)
        saved_image.__image.set_filename(filename)
        nib.save(saved_image.__image, name)

    @classmethod
    def from_data(cls, data, header):
        """
        Create object from data and header
        :param data: The image data
        :param header: The image header to use
        """
        image = nib.Nifti1Image(data, affine=None, header=header)
        return cls(image)

    @classmethod
    def generate_default_image_from_data(cls, data):
        """
        Create object from data. Set transformations to identity
        :param data: The image data
        """
        image = nib.Nifti1Image(data, np.eye(4))
        return cls(image)

    @classmethod
    def from_file(cls, imagepath):
        """
        Create object from image file
        :param imagepath: The path to the image file
        """
        image = nib.load(imagepath)
        return cls(image)

    def __set_attributes(self):
        """
        Set other attributes like transformations, volume extents etc.
        All tied towards nifti images at the moment but we need to
        make it generic later for the supported formats
        """
        # nibabel designers in their infinite wisdom, have made this really
        # complicated. Note, that if you use the parameter code=True
        # (needed to get the sform code, which for some reason cannot be
        # queried independently), the returned affine will be None if the
        # sform_code is 0.
        [self.voxel_2_mm, code] = self.__image.get_sform(True)
        if code <= 0:
            # Do not call it with 'True', else you will not even have the
            # default matrix if qform is set to 0
            # (which should not happen but can happen, I guess)
            self.voxel_2_mm = self.__image.get_qform()

        self.mm_2_voxel = la.inv(self.voxel_2_mm)
        self.time_points = 1
        self.vol_ext = self.__image.shape

        if len(self.vol_ext) > 3:
            self.time_points = self.vol_ext[3]
            # Set it to the underlying volume extent
            self.vol_ext = self.vol_ext[:3]

        self.data = self.__image.get_data()
        self.zooms = self.__image.get_header().get_zooms()

    def get_header(self):
        """
        Get the nifti header associated with this image
        :return: Nifti header.
        """
        return self.__image.get_header()

    # pylint: disable=W0201
    def update_transformation(self, mat):
        """
        Update the transformation matrix
        :param mat: The new transformation matrix
        """
        header = self.__image.get_header()
        self.voxel_2_mm = mat
        self.mm_2_voxel = la.inv(self.voxel_2_mm)
        header.set_sform(mat, 1)
