# -*- coding: utf-8 -*-
"""Implementations of puff-based plume model components."""

from __future__ import division

__authors__ = 'Matt Graham'
__license__ = 'MIT'


import numpy as np
import scipy.interpolate as interp


class SlottedIterable(object):
    """Base class for objects with slots which can be used as iterables.

    Allows instances of subclasses to be used as iterables and provides a
    human-readable string representation.
    """

    __slots__ = ()

    def __iter__(self):
        """Iterate through slot attributes in defined order."""
        for name in self.__slots__:
            yield getattr(self, name)

    def __repr__(self):
        """String representation of object."""
        return '{cls}({attr})'.format(
            cls=self.__class__.__name__,
            attr=', '.join(['{0}={1}'.format(
                name, getattr(self, name)) for name in self.__slots__]))


class Puff(SlottedIterable):
    """Container for the properties of a single odour puff."""

    __slots__ = ('x', 'y', 'z', 'r_sq')

    def __init__(self, x, y, z, r_sq):
        """
        Parameters
        ----------
        x : float
            x-coordinate of puff centre.
        y : float
            y-coordinate of puff centre.
        z : float
            z-coordinate of puff centre.
        r_sq : float
            Squared radius of puff.
        """
        assert r_sq >= 0., 'r_sq must be non-negative.'
        self.x = x
        self.y = y
        self.z = z
        self.r_sq = r_sq


class Rectangle(SlottedIterable):
    """Axis-aligned rectangular region.

    Rectangle is defined by two points `(x_min, y_min)` and `(x_max, y_max)`
    with it required that `x_max > x_min` and `y_max > y_min`.
    """

    __slots__ = ('x_min', 'x_max', 'y_min', 'y_max')

    def __init__(self, x_min, x_max, y_min, y_max):
        """
        Parameters
        ----------
        x_min : float
            x-coordinate of bottom-left corner of rectangle.
        y_min : float
            x-coordinate of bottom-right corner of rectangle.
        x_max : float
            x-coordinate of top-right corner of rectangle.
        y_max : float
            y-coordinate of top-right corner of rectangle.
        """
        assert x_min < x_max, 'Rectangle x_min must be < x_max.'
        assert y_min < y_max, 'Rectangle y_min must be < y_max.'
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @property
    def w(self):
        """Width of rectangle (i.e. distance covered on x-axis)."""
        return self.x_max - self.x_min

    @property
    def h(self):
        """Height of rectangle (i.e. distance covered on y-axis)."""
        return self.y_max - self.y_min

    def contains(self, x, y):
        """Whether `(x, y)`` position is contained within this rectangle.

        Tests whether the supplied position, an `(x,y)` pair, is contained
        within the region defined by this `Rectangle` object and returns `True`
        if so and `False` if not.

        Parameters
        ----------
        x : float
            x-coordinate of position to test.
        y : float
            y-coordinate of position to test.

        Returns
        -------
        contains : boolean
            `True` if `(x, y)` is within the rectangle and `False` otherwise.
        """
        return (x >= self.x_min and x <= self.x_max and
                y >= self.y_min and y <= self.y_max)


class PlumeModel(object):
    """Puff-based odour plume dispersion model from Farrell et. al. (2002).

    The odour plume is modelled as a series of odour puffs which are released
    from a fixed source position. The odour puffs are dispersed by a modelled
    2D wind velocity field plus a white noise process model of mid-scale puff
    mass diffusion relative to the plume centre line. The puffs also spread in
    size over time to model fine-scale diffusive processes.
    """

    def __init__(self, sim_region=None, source_pos=(5., 0., 0.),
                 wind_model=None, model_z_disp=True, centre_rel_diff_scale=2.,
                 puff_init_rad=0.0316, puff_spread_rate=0.001,
                 puff_release_rate=10, init_num_puffs=10, max_num_puffs=1000,
                 rng=None):
        """
        Parameters
        ----------
        sim_region : Rectangle
            2D rectangular region of space over which the simulation is
            conducted. This should be a subset of the simulation region defined
            for the wind model.
        source_pos : float iterable
            Coordinates of the fixed source position within the simulation
            region from which puffs are released. If a length 2 iterable is
            passed, the z coordinate will be set a default of 0
            (dimension: length).
        wind_model : WindModel
            Dynamic model of the large scale wind velocity field in the
            simulation region.
        model_z_disp : boolean
            Whether to model dispersion of puffs from plume centre-line in z
            direction. If set `True` then the puffs will be modelled as
            dispersing in the vertical direction by a random walk process (the
            wind model is limited to 2D hence the vertical wind speed is
            assumed to be zero), if set `False` the puff z-coordinates will not
            be updated from their initial value of 0.
        centre_rel_diff_scale : float or float iterable
            Scaling for the stochastic process used to model the centre-line
            relative diffusive transport of puffs. Either a single float value
            of isotropic diffusion in all directions, or one of a pair of
            values specifying different scales for the x and y directions
            respectively if `model_z_disp=False` or a triplet of values
            specifying different scales for x, y and z scales respectively if
            `model_z_disp=True` (dimension: length / time**0.5).
        puff_init_rad: float
            Initial radius of the puffs (dimension: length).
        puff_spread_rate : float
            Constant which determines the rate at which the odour puffs
            increase in size over time (dimension: length**2 / time).
        puff_release_rate : float
            Mean rate at which new puffs are released into the plume. Puff
            release is modelled as a stochastic Poisson process, with each puff
            released assumed to be independent and the mean release rate fixed
            (dimension: count/time).
        init_num_puffs : integer
            Initial number of puffs to release at the beginning of the
            simulation.
        max_num_puffs : integer
            Maximum number of puffs to permit to be in existence simultaneously
            within model, used to limit memory and processing requirements of
            model. This parameter needs to be set carefully in relation to the
            puff release rate and simulation region size as if too small it
            will lead to breaks in puff release when the number of puffs
            remaining in the simulation region reaches the limit.
        rng : RandomState
            Random number generator to use in generating input noise. Defaults
            to `numpy.random` global generator if set to `None` however a
            seeded `RandomState` object can be passed if it is desired to have
            reproducible output.
        """
        if sim_region is None:
            sim_region = Rectangle(0., 50., -12.5, 12.5)
        if rng is None:
            rng = np.random
        self.sim_region = sim_region
        if wind_model is None:
            wind_model = WindModel()
        self.wind_model = wind_model
        self.rng = rng
        self.model_z_disp = model_z_disp
        self._vel_dim = 3 if model_z_disp else 2
        if model_z_disp and hasattr(centre_rel_diff_scale, '__len__'):
            assert len(centre_rel_diff_scale) == 2, (
                'When model_z_disp=True, centre_rel_diff_scale must be a '
                'scalar or length 1 or 3 iterable.')
        self.centre_rel_diff_scale = centre_rel_diff_scale
        assert sim_region.contains(source_pos[0], source_pos[1]), (
            'Specified source position must be within simulation region.')
        # default to zero height source when source_pos is 2D
        source_z = 0 if len(source_pos) != 3 else source_pos[2]
        self._new_puff_params = (
            source_pos[0], source_pos[1], source_z, puff_init_rad**2)
        self.puff_spread_rate = puff_spread_rate
        self.puff_release_rate = puff_release_rate
        self.max_num_puffs = max_num_puffs
        # initialise puff list with specified number of new puffs
        self.puffs = [
            Puff(*self._new_puff_params) for i in range(init_num_puffs)]

    def update(self, dt):
        """Update plume puff objects by forward intgating one time-step.

        Performs a single time-step update of plume model using Euler
        integration scheme.

        Parameters
        ----------
        dt : float
            Simulation time-step (dimension: time).
        """
        # add more puffs (stochastically) if enough capacity
        if len(self.puffs) < self.max_num_puffs:
            # puff release modelled as Poisson process at fixed mean rate
            # with number to release clipped if it would otherwise exceed
            # the maximum allowed
            num_to_release = min(
                self.rng.poisson(self.puff_release_rate * dt),
                self.max_num_puffs - len(self.puffs))
            self.puffs += [
                Puff(*self._new_puff_params) for i in range(num_to_release)]
        # initialise empty list for puffs that have not left simulation area
        alive_puffs = []
        for puff in self.puffs:
            # interpolate wind velocity at Puff position from wind model grid
            # assuming zero wind speed in vertical direction if modelling
            # z direction dispersion
            wind_vel = np.zeros(self._vel_dim)
            wind_vel[:2] = self.wind_model.velocity_at_pos(puff.x, puff.y)
            # approximate centre-line relative puff transport velocity
            # component as being a (Gaussian) white noise process scaled by
            # constants
            filament_diff_vel = (self.rng.normal(size=self._vel_dim) *
                                 self.centre_rel_diff_scale)
            vel = wind_vel + filament_diff_vel
            # update puff position using Euler integration
            puff.x += vel[0] * dt
            puff.y += vel[1] * dt
            if self.model_z_disp:
                puff.z += vel[2] * dt
            # update puff size using Euler integration with second puff
            # growth model described in paper
            puff.r_sq += self.puff_spread_rate * dt
            # only keep puff alive if it is still in the simulated region
            if self.sim_region.contains(puff.x, puff.y):
                alive_puffs.append(puff)
        # store alive puffs only
        self.puffs = alive_puffs

    @property
    def puff_array(self):
        """NumPy array of the properties of the simulated puffs.

        Each row corresponds to one puff with the first column containing the
        puff position x-coordinate, the second the y-coordinate, the third the
        z-coordinate and the fourth the puff squared radius.
        """
        return np.array([tuple(puff) for puff in self.puffs])


class WindModel(object):
    """Wind velocity model to calculate advective transport of odour.

    A 2D approximation is used as described in the paper, with the wind
    velocities calculated over a regular 2D grid of points using a finite
    difference method. The boundary conditions at the edges of the simulated
    region are for both components of the velocity field constant mean values
    plus coloured noise. For each of the field components these are calculated
    for the four corners of the simulated region and then linearly interpolated
    over the edges.
    """

    def __init__(self, sim_region=None, n_x=21, n_y=21, u_av=1., v_av=0.,
                 k_x=20., k_y=20., noise_gain=2., noise_damp=0.1,
                 noise_bandwidth=0.2, use_original_noise_updates=False,
                 rng=None):
        """
        Parameters
        ----------
        sim_region : Rectangle
            Two-dimensional rectangular region over which to model wind
            velocity field.
        n_x : integer
            Number of grid points in x direction.
        n_y : integer
            Number of grid points in y direction.
        u_av : float
            Mean x-component of wind velocity (dimension: length / time).
        v_av : float
            Mean y-component of wind velocity (dimension: length / time).
        k_x : float or array_like
            Diffusivity constant in x direction. Either a single scalar value
            across the whole simulated region or an array of size `(n_x, n_y)`
            defining values for each grid point (dimension: length**2 / time).
        k_y : float or array_like
            Diffusivity constant in y direction. Either a single scalar value
            across the whole simulated region or an array of size `(n_x, n_y)`
            defining values for each grid point (dimension: length**2 / time).
        noise_gain : float
            Input gain constant for boundary condition noise generation
            (dimensionless).
        noise_damp : float
            Damping ratio for boundary condition noise generation
            (dimensionless).
        noise_bandwidth : float
            Bandwidth for boundary condition noise generation (dimension:
            angle / time).
        use_original_noise_updates : boolean
            Whether to use the original non-SDE based updates for the noise
            process as defined in Farrell et al. (2002), see notes in
            `ColouredNoiseGenerator` documentation.
        rng : RandomState
            Random number generator to use in generating input noise. Defaults
            to `numpy.random` global generator if set to `None` however a
            seeded `RandomState` object can be passed if it is desired to have
            reproducible output.
        """
        if sim_region is None:
            sim_region = Rectangle(0, 100, -50, 50)
        if rng is None:
            rng = np.random
        self.sim_region = sim_region
        self.u_av = u_av
        self.v_av = v_av
        self.n_x = n_x
        self.n_y = n_y
        self.k_x = k_x
        self.k_y = k_y
        # set coloured noise generator for applying boundary condition
        # need to generate coloured noise samples at four corners of boundary
        # for both components of the wind velocity field so (2,8) state
        # vector (2 as state includes first derivative)
        self.noise_gen = ColouredNoiseGenerator(
            np.zeros((2, 8)), noise_damp, noise_bandwidth, noise_gain,
            use_original_noise_updates, rng)
        # compute grid node spacing
        self.dx = sim_region.w / (n_x - 1)  # x grid point spacing
        self.dy = sim_region.h / (n_y - 1)  # y grid point spacing
        # initialise wind velocity field to mean values
        # +2s are to account for boundary grid points
        self._u = np.ones((n_x + 2, n_y + 2)) * u_av
        self._v = np.ones((n_x + 2, n_y + 2)) * v_av
        # create views on to field interiors (i.e. not including boundaries)
        # for notational ease - note this does not copy any data
        self._u_int = self._u[1:-1, 1:-1]
        self._v_int = self._v[1:-1, 1:-1]
        # preassign array of corner means values
        self._corner_means = np.array([u_av, v_av]).repeat(4)
        # precompute linear ramp arrays with size of boundary edges for
        # linear interpolation of corner values
        self._ramp_x = np.linspace(0., 1., n_x + 2)
        self._ramp_y = np.linspace(0., 1., n_y + 2)
        # set up cubic spline interpolator for calculating off-grid wind
        # velocity field values
        self._x_points = np.linspace(sim_region.x_min, sim_region.x_max, n_x)
        self._y_points = np.linspace(sim_region.y_min, sim_region.y_max, n_y)
        # initialise flag to indicate velocity field interpolators not set
        self._interp_set = True

    def _set_interpolators(self):
        """ Set spline interpolators using current velocity fields."""
        self._interp_u = interp.RectBivariateSpline(
            self.x_points, self.y_points, self._u_int)
        self._interp_v = interp.RectBivariateSpline(
            self.x_points, self.y_points, self._v_int)
        self._interp_set = True

    @property
    def x_points(self):
        """1D array of the range of x-coordinates of simulated grid points."""
        return self._x_points

    @property
    def y_points(self):
        """1D array of the range of y-coordinates of simulated grid points."""
        return self._y_points

    @property
    def velocity_field(self):
        """Current calculated velocity field across simulated grid points."""
        return np.dstack((self._u_int, self._v_int))

    def velocity_at_pos(self, x, y):
        """Calculate velocity at a position or positions.

        Calculates the components of the velocity field at arbitrary point(s)
        in the simulation region using a bivariate spline interpolation over
        the calculated grid point values.

        Parameters
        ----------
        x : float or array
            x-coordinate of the point(s) to calculate the velocity at
            (dimension: length).
        y : float or array
            y-coordinate of the point(s) to calculate the velocity at
            (dimension: length).

        Returns
        -------
        vel : array
            Velocity field (2D) values evaluated at specified point(s)
            (dimension: length / time).
        """
        if not self._interp_set:
            self._set_interpolators()
        return np.array([float(self._interp_u(x, y)),
                         float(self._interp_v(x, y))])

    def update(self, dt):
        """Update wind velocity field by forward integrating one time-step.

        Updates wind velocity field values using finite difference
        approximations for spatial derivatives and Euler integration for
        time-step update.

        Parameters
        ----------
        dt : float
            Simulation time-step (dimension: time).
        """
        # update boundary values
        self._apply_boundary_conditions(dt)
        # approximate spatial first derivatives with centred finite difference
        # equations for both components of wind field
        du_dx, du_dy = self._centred_first_diffs(self._u)
        dv_dx, dv_dy = self._centred_first_diffs(self._v)
        # approximate spatial second derivatives with centred finite difference
        # equations for both components of wind field
        d2u_dx2, d2u_dy2 = self._centred_second_diffs(self._u)
        d2v_dx2, d2v_dy2 = self._centred_second_diffs(self._v)
        # compute approximate time derivatives across simulation region
        # interior from defining PDEs
        #     du/dt = -(u*du/dx + v*du/dy) + 0.5*k_x*d2u/dx2 + 0.5*k_y*d2u/dy2
        #     dv/dt = -(u*dv/dx + v*dv/dy) + 0.5*k_x*d2v/dx2 + 0.5*k_y*d2v/dy2
        du_dt = (-self._u_int * du_dx - self._v_int * du_dy +
                 0.5 * self.k_x * d2u_dx2 + 0.5 * self.k_y * d2u_dy2)
        dv_dt = (-self._u_int * dv_dx - self._v_int * dv_dy +
                 0.5 * self.k_x * d2v_dx2 + 0.5 * self.k_y * d2v_dy2)
        # perform update with Euler integration
        self._u_int += du_dt * dt
        self._v_int += dv_dt * dt
        # set flag to indicate interpolators no longer valid as fields updated
        self._interp_set = False

    def _apply_boundary_conditions(self, dt):
        """Applies boundary conditions to wind velocity field."""
        # update coloured noise generator
        self.noise_gen.update(dt)
        # extract four corner values for each of u and v fields as component
        # mean plus current noise generator output
        (u_tl, u_tr, u_bl, u_br, v_tl, v_tr, v_bl, v_br) = (
            self.noise_gen.output + self._corner_means)
        # linearly interpolate along edges
        self._u[:, 0] = u_tl + self._ramp_x * (u_tr - u_tl)  # u top edge
        self._u[:, -1] = u_bl + self._ramp_x * (u_br - u_bl)  # u bottom edge
        self._u[0, :] = u_tl + self._ramp_y * (u_bl - u_tl)  # u left edge
        self._u[-1, :] = u_tr + self._ramp_y * (u_br - u_tr)  # u right edge
        self._v[:, 0] = v_tl + self._ramp_x * (v_tr - v_tl)  # v top edge
        self._v[:, -1] = v_bl + self._ramp_x * (v_br - v_bl)  # v bottom edge
        self._v[0, :] = v_tl + self._ramp_y * (v_bl - v_tl)  # v left edge
        self._v[-1, :] = v_tr + self._ramp_y * (v_br - v_tr)  # v right edge

    def _centred_first_diffs(self, f):
        """Calculates centred first-order finite differences."""
        return ((f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * self.dx),
                (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * self.dy))

    def _centred_second_diffs(self, f):
        """Calculates centred second-order finite differences."""
        return (
            (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[0:-2, 1:-1]) / self.dx**2,
            (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, 0:-2]) / self.dy**2)


class ColouredNoiseGenerator(object):
    """Generator of coloured (correlated) Gaussian noise process.

    Generates a coloured noise output via numerical integration of a stochastic
    differential equation formulation. The system is assumed to be defined by
    the system of SDEs::

        dx_0  = x_1 * dt
        dx_1  = -(a * x_0 + b * x_1) * dt + c * dn

    where `a = bandwidth**2` and `b = 2 * damping * bandwidth`,
    `c = gain * bandwidth**2` and `dn` is a standard Gaussian white noise
    process. This is numerically integrated using an Euler-Maruyama scheme::

        for t in range(n_timestep):
            x[t+1,0] = x[t,0] + dt * x[t,1]
            x[t+1,1] = x[t,1] - dt * (a*x[t,0] + b*x[t,1]) + dt**0.5 * c * n[t]

    where `x` is an array of shape `(n_timestep, 2)` and `n` is an array of
    shape `(n_timestep)` filled with random standard normal draws.

    This differs from the code accompanying Farrell et al. (2002) which applies
    an Euler integration scheme to a state space formulation of a second-order
    linear system with Gaussian noise input at each time step, resulting in
    updates of the form::

        for t in range(n_timestep):
            x[t+1,0] = x[t,0] + dt * x[t,1]
            x[t+1,1] = x[t,1] + dt * (-a * x[t,0] - b * x[t,1] + c * n[t])

    This differs from the scheme implemented here by scaling the noise input
    by the timestep `dt` rather than its square root `dt**0.5`. This introduces
    an implicit dependence of the amplitude of the process on `dt`, in
    particular that the amplitude scales roughly as `dt**0.5`.

    Updates consistent with the Farrell et al. (2002) implementation can be
    achieved by setting the `use_original_updates` flat to `True`.
    """

    def __init__(self, init_state, damping=0.1, bandwidth=0.2, gain=1.,
                 use_original_updates=False,  rng=None):
        """
        Parameters
        ----------
        init_state : array_like
            The initial state of system, must be of shape `(2,n)` where `n` is
            the size of the noise vector to be produced. The first row
            sets the initial values and the second the initial first
            derivatives.
        damping : float
            Damping ratio for the system, affects system damping, values of
            < 1 give an underdamped system, = 1 a critically damped system and
            > 1 an overdamped system (dimensionless).
        bandwidth : float
            Bandwidth or equivalently undamped natural frequency of system,
            affects system reponsiveness to variations in (noise) input
            (dimension = 1 / time).
        gain : float
            Input gain of system, affects scaling of (noise) input.
        rng : RandomState
            Random number generator to use in generating input noise. Defaults
            to `numpy.random` global generator if set to `None` however a
            seeded `RandomState` object can be passed if it is desired to have
            reproducible output.
        use_original_updates : boolean
            Whether to use the original non-SDE based updates for the noise
            process as defined in Farrell et al. (2002), see above notes.
        """
        if rng is None:
            rng = np.random
        # set up state space matrices
        self.a_mtx = np.array([
            [0., 1.], [-bandwidth**2, -2. * damping * bandwidth]])
        self.b_mtx = np.array([[0.], [gain * bandwidth**2]])
        # initialise state
        self.state = init_state
        self.rng = rng
        self.use_original_updates = use_original_updates

    @property
    def output(self):
        """Coloured noise output."""
        return self.state[0, :]

    def update(self, dt):
        """Update state of noise generator.

        Parameters
        ----------
        dt : float
            Integrator time step.
        """
        # get normal random input
        n = self.rng.normal(size=(1, self.state.shape[1]))
        if self.use_original_updates:
            # apply Farrell et al. (2002) update
            self.state += dt * (self.a_mtx.dot(self.state) + self.b_mtx.dot(n))
        else:
            # apply update with Euler-Maruyama integration
            self.state += (
                dt * self.a_mtx.dot(self.state) + self.b_mtx.dot(n) * dt**0.5)
