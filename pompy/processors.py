# -*- coding: utf-8 -*-
"""
Helper classes to process outputs of models.
"""

from __future__ import division

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import math
import numpy as np


class ConcentrationValueCalculator(object):

    """
    Calculates odour concentration values at points in simulation region from
    puff property arrays.
    """

    def __init__(self, puff_molecular_amount):
        """
        Parameters
        ----------
        puff_mol_amount : float
            Molecular content of each puff (e.g. in moles or raw number of
            molecules). This is conserved as the puff is transported within
            the plume but the puff becomes increasingly diffuse as it's radius
            grows due to diffusion.
        """
        # precompute constant used to scale Gaussian amplitude
        self._ampl_const = puff_molecular_amount / (8 * np.pi**3)**0.5

    def _puff_conc_dist(self, x, y, z, px, py, pz, r_sq):
        # calculate Gaussian puff concentration distribution
        return (
            self._ampl_const / r_sq**1.5 *
            np.exp(-((x - px)**2 + (y - py)**2 + (z - pz)**2) / (2 * r_sq))
        )

    def calc_conc_point(self, puff_array, x, y, z=0):
        """
        Calculate concentration at a single point.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : float
            x-coordinate of point.
        y : float
            y-coordinate of point.
        z : float
            z-coordinate of point.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        return self._puff_conc_dist(x, y, z, px, py, pz, r_sq).sum(-1)

    def calc_conc_list(self, puff_array, x, y, z=0):
        """
        Calculate concentrations across a 1D list of points in a xy-plane.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : (np) numpy-array-like of floats
            1D array of x-coordinates of points.
        y : (np) numpy-array-like of floats
            1D array of y-coordinates of points.
        z : float
            z-coordinate (height) of plane.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        na = np.newaxis
        return self._puff_conc_dist(x[:, na], y[:, na], z, px[na, :],
                                    py[na, :], pz[na, :], r_sq[na, :]).sum(-1)

    def calc_conc_grid(self, puff_array, x, y, z=0):
        """
        Calculate concentrations across a 2D grid of points in a xy-plane.

        Parameters
        ----------
        puff_array : numpy-array-like of Puff objects
            Collection of currently alive puff instances at a particular
            time step which it is desired to calculate concentration field
            values from.
        x : (nx,ny) numpy-array-like of floats
            2D array of x-coordinates of grid points.
        y : (nx,ny) numpy-array-like of floats
            2D array of y-coordinates of grid points.
        z : float
            z-coordinate (height) of grid plane.
        """
        # filter for non-nan puff entries and separate properties for
        # convenience
        px, py, pz, r_sq = puff_array[~np.isnan(puff_array[:, 0]), :].T
        na = np.newaxis
        return self._puff_conc_dist(x[:, :, na], y[:, :, na], z,
                                    px[na, na, :], py[na, na, :],
                                    pz[na, na, :], r_sq[na, na, :]).sum(-1)


class ConcentrationArrayGenerator(object):

    """
    Produces odour concentration field arrays from puff property arrays.

    Instances of this class can take single or multiple arrays of puff
    properties outputted from a PlumeModel and process them to produce an
    array of the concentration values across the a specified region using
    a Gaussian model for the individual puff concentration distributions.

    Compared to the ConcentrationValueCalculator class, this class should be
    more efficient for calculating large concentration field arrays for
    real-time graphical display of odour concentration fields for example
    at the expense of (very) slightly less accurate values due to the
    truncation of spatial extent of each puff.

    Notes
    -----
    The returned array values correspond to the *point* concentration
    measurements across a regular grid of sampling points - i.e. the
    equivalent to convolving the true continuous concentration distribution
    with a regular 2D grid of Dirac delta / impulse functions. An improvement
    in some ways would be to instead calculate the integral of the
    concentration distribution over the (square) region around each grid point
    however this would be extremely computationally costly and due to the lack
    of a closed form solution for the integral of a Gaussian also potentially
    difficult to implement without introducing other numerical errors. An
    integrated field can be approximated with this class by generating an
    array at a higher resolution than required and then filtering with a
    suitable kernel and down-sampling.

    This implementation estimates the concentration distribution puff kernels
    with sub-grid resolution, giving improved accuracy at the cost of
    increased computational cost versus using a precomputed radial field
    aligned with the grid to compute kernel values or using a library of
    precomputed kernels.

    For cases where the array region cover the whole simulation region the
    computational cost could also be reduced by increasing the size of the
    region the array corresponds to outside of the simulation region such that
    when adding the puff concentration kernels to the concentration field
    array, checks do not need to be made to restrict to the overlapping region
    for puffs near the edges of the simulation region which have a
    concentration distribution which extends beyond its extents.
    """

    def __init__(self, array_xy_region, array_z, nx, ny, puff_mol_amount,
                 kernel_rad_mult=3):
        """
        Parameters
        ----------
        array_region : Rectangle
            Two-dimensional rectangular region defined in world coordinates
            over which to calculate the concentration field.
        array_z : float
            Height on the vertical z-axis at which to calculate the
            concentration field over.
        nx : integer
            Number of grid points to sample at across x-dimension.
        ny : integer
            Number of grid points to sample at across y-dimension.
        puff_mol_amount : float
            Molecular content of each puff (e.g. in moles or raw number of
            molecules). This is conserved as the puff is transported within
            the plume but the puff becomes increasingly diffuse as it's radius
            grows due to diffusion.
            (dimensionality:molecular amount)
        kernel_rad_mult : float
            Multiplier used to determine to within how many puff radii from
            the puff centre to truncate the concentration distribution
            kernel calculated to. The default value of 3 will truncate the
            Gaussian kernel at (or above) the point at which the concentration
            has dropped to 0.004 of the peak value at the puff centre.
        """
        self.array_xy_region = array_xy_region
        self.array_z = array_z
        self.nx = nx
        self.ny = ny
        self._dx = array_xy_region.w / nx  # calculate x grid point spacing
        self._dy = array_xy_region.h / ny  # calculate y grid point spacing
        # precompute constant used to scale Gaussian kernel amplitude
        self._ampl_const = puff_mol_amount / (8*np.pi**3)**0.5
        self.kernel_rad_mult = kernel_rad_mult

    def puff_kernel(self, shift_x, shift_y, z_offset, r_sq, even_w, even_h):
        # kernel is truncated to min +/- kernel_rad_mult * effective puff
        # radius from centre i.e. Gaussian kernel with >= kernel_rad_mult *
        # standard deviation span
        # (effective puff radius is (r_sq - (z_offset/k_r_mult)**2)**0.5 to
        # account for the cross sections of puffs with centres out of the
        # array plane being 'smaller')
        # the truncation will introduce some errors - an improvement would
        # be to use some form of windowing e.g. Hann or Hamming window
        shape = (2*(r_sq*self.kernel_rad_mult**2 - z_offset**2)**0.5 /
                 np.array([self._dx, self._dy]))
        # depending on whether centre is on grid points or grid centres
        # kernel dimensions will need to be forced to odd/even respectively
        shape[0] = self.round_up_to_next_even_or_odd(shape[0], even_w)
        shape[1] = self.round_up_to_next_even_or_odd(shape[1], even_h)
        # generate x and y grids with required shape
        [x_grid, y_grid] = 0.5 + np.mgrid[-shape[0]/2:shape[0]/2,
                                          -shape[1]/2:shape[1]/2]
        # apply shifts to correct for offset of true centre from nearest
        # grid-point / centre
        x_grid = x_grid * self._dx + shift_x
        y_grid = y_grid * self._dy + shift_y
        # compute square radial field
        r_sq_grid = x_grid**2 + y_grid**2 + z_offset**2
        # output scaled Gaussian kernel
        return self._ampl_const / r_sq**1.5 * np.exp(-r_sq_grid / (2 * r_sq))

    @staticmethod
    def round_up_to_next_even_or_odd(value, to_even):
        # Returns value rounded up to first even number >= value if
        # to_even==True and to first odd number >= value if to_even==False.
        value = math.ceil(value)
        if to_even:
            if value % 2 == 1:
                value += 1
        else:
            if value % 2 == 0:
                value += 1
        return value

    def generate_single_array(self, puff_array):
        """
        Generates a single concentration field array from an array of puff
        properties.
        """
        # initialise concentration array
        conc_array = np.zeros((self.nx, self.ny))
        # loop through all the puffs
        for (puff_x, puff_y, puff_z, puff_r_sq) in puff_array:
            # to begin with check this a real puff and not a placeholder nan
            # entry as puff arrays may have been pre-allocated with nan
            # at a fixed size for efficiency and as the number of puffs
            # existing at any time interval is variable some entries in the
            # array will be unallocated, placeholder entries should be
            # contiguous (i.e. all entries after the first placeholder will
            # also be placeholders) therefore break out of loop completely
            # if one is encountered
            if np.isnan(puff_x):
                break
            # check puff centre is within region array is being calculated
            # over otherwise skip
            if not self.array_xy_region.contains(puff_x, puff_y):
                continue
            # finally check that puff z-coordinate is within
            # kernel_rad_mult*r_sq of array evaluation height otherwise skip
            puff_z_offset = (self.array_z - puff_z)
            if abs(puff_z_offset) / puff_r_sq**0.5 > self.kernel_rad_mult:
                continue
            # calculate (float) row index corresponding to puff x coord
            p = (puff_x - self.array_xy_region.x_min) / self._dx
            # calculate (float) column index corresponding to puff y coord
            q = (puff_y - self.array_xy_region.y_min) / self._dy
            # calculate nearest integer or half-integer row index to p
            u = math.floor(2 * p + 0.5) / 2
            # calculate nearest integer or half-integer row index to q
            v = math.floor(2 * q + 0.5) / 2
            # generate puff kernel array of appropriate scale and taking
            # into account true centre offset from nearest half-grid
            # points (u,v)
            kernel = self.puff_kernel((p - u) * self._dx, (q - v) * self._dy,
                                      puff_z_offset, puff_r_sq,
                                      u % 1 == 0, v % 1 == 0)
            # compute row and column slices for source kernel array and
            # destination concentration array taking in to the account
            # the possibility of the kernel being partly outside the
            # extents of the destination array
            (w, h) = kernel.shape
            r_rng_arr = slice(max(0, u - w / 2.),
                              max(min(u + w / 2., self.nx), 0))
            c_rng_arr = slice(max(0, v - h / 2.),
                              max(min(v + h / 2., self.ny), 0))
            r_rng_knl = slice(max(0, -u + w / 2.),
                              min(-u + w / 2. + self.nx, w))
            c_rng_knl = slice(max(0, -v + h / 2.),
                              min(-v + h / 2. + self.ny, h))
            # add puff kernel values to concentration field array
            conc_array[r_rng_arr, c_rng_arr] += kernel[r_rng_knl, c_rng_knl]
        return conc_array

    def generate_multiple_arrays(self, puff_arrays):
        """
        Generates multiple concentration field arrays from a sequence of
        arrays of puff properties.
        """
        conc_arrays = []
        for puff_array in puff_arrays:
            conc_arrays.append(self.generate_single_frame(puff_array))
        return conc_arrays
