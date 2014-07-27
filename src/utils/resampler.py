from scipy import ndimage
import numpy as np


class ImageResampler:
    """
    Class to resample image according to a deformation field
    """
    def __init__(self):
        """
        Linear interpolation by default.
        """
        self.order = 1
        self.prefilter = False

    def resample(self, source, deformation, warped):
        """
        Resample the source image in the space of target.

        Parameters:
        :param source: The source image.
        :param deformation: The deformation field
        :param warped: Warped image. Should be allocated in the space of target
        and must have the same number of time points as the source image.
        """

        assert(source.time_points == warped.time_points)
        # Matrix to go from world to voxel space
        ijk_2_voxel = source.mm_2_voxel
        vol_ext = warped.vol_ext

        def_data = [deformation.data[..., i].reshape(vol_ext, order='F')
                    for i in range(deformation.data.shape[-1])]

        def_data = [ijk_2_voxel[i][3] + sum(ijk_2_voxel[i][k] * def_data[k]
                    for k in range(len(def_data)))
                    for i in range(len(vol_ext))]



        if source.time_points == 1:
            ndimage.map_coordinates(source.data, def_data, warped.data,
                                    order=self.order, prefilter=self.prefilter)
        else:
            for i in range(source.time_points):
                ndimage.map_coordinates(source.data[i], def_data, warped.data[i],
                                        order=self.order, prefilter=self.prefilter)


class NearestNeighbourResampler(ImageResampler):
    """
    Set to nearest neighbourhood resampling
    """
    def __init__(self):
        self.order = 0


class CubicSplineResampler(ImageResampler):
    """
    Set to cubic spline resampling
    """
    def __init__(self):
        self.order = 2
        self.prefilter = True


class FieldsComposer:
    """
    Compose fields using linear interpolation.
    """

    def __init__(self):
        self.order = 1

    def compose(self, left, right, result):
        """
        Compose deformations.
        Parameters:
        -----------
        :param left: Outer field.
        :param right: Inner field.
        :param result: Resulting field after composition. Must be valid and allocated.

        Order of composition: left(right(x))
        """
        ijk_2_voxel = right.mm_2_voxel
        vol_ext = right.vol_ext[:right.data.shape[-1]]
        right_data = [right.data[..., i].reshape(vol_ext, order='F')
                      for i in range(right.data.shape[-1])]

        right_data = [ijk_2_voxel[i][3] + sum(ijk_2_voxel[i][k] * right_data[k]
                      for k in range(len(right_data)))
                      for i in range(len(vol_ext))]

        data = np.squeeze(result.data)
        for i in range(data.shape[-1]):
            ndimage.map_coordinates(np.squeeze(left.data[..., i]), right_data,
                                    data[..., i], mode='reflect', order=self.order, prefilter=False)
