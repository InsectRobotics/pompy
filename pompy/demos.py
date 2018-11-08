# -*- coding: utf-8 -*-
"""Demonstrations of setting up models and visualising outputs."""

from __future__ import division

__authors__ = 'Matt Graham'
__license__ = 'MIT'


import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import numpy as np
from pompy import models, processors


DEFAULT_SEED = 20181108


def set_up_figure(fig_size=(10, 5)):
    """Set up Matplotlib figure with simulation time title text.

    Parameters
    ----------
    title_text : string
        Text to set figure title to.
    fig_size : tuple
        Figure dimensions in inches in order `(width, height)`.
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    title = ax.set_title('Simulation time = ---- seconds')
    return fig, ax, title


def update_decorator(dt, title, steps_per_frame, models):
    """Decorator for animation update methods."""
    def inner_decorator(update_function):
        def wrapped_update(i):
            for j in range(steps_per_frame):
                for model in models:
                    model.update(dt)
            t = i * steps_per_frame * dt
            title.set_text('Simulation time = {0:.3f} seconds'.format(t))
            return [title] + update_function(i)
        return wrapped_update
    return inner_decorator


def wind_model_demo(dt=0.01, t_max=100, steps_per_frame=20, seed=DEFAULT_SEED):
    """Set up wind model and animate velocity field with quiver plot.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    t_max : float
        End time to simulate to.
    steps_per_frame: integer
        Number of simulation time steps to perform between animation frames.
    seed : integer
        Seed for random number generator.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : AxesSubplot
        Matplotlib axis object.
    anim : FuncAnimation
        Matplotlib animation object.
    """
    rng = np.random.RandomState(seed)
    # define simulation region
    wind_region = models.Rectangle(x_min=0., x_max=100., y_min=-25., y_max=25.)
    # set up wind model
    wind_model = models.WindModel(wind_region, 21, 11, rng=rng)
    # let simulation run for 10s to equilibrate wind model
    for t in np.arange(0, 10, dt):
        wind_model.update(dt)
    # generate figure and attach close event
    fig, ax, title = set_up_figure()
    # create quiver plot of initial velocity field
    vf_plot = ax.quiver(wind_model.x_points, wind_model.y_points,
                        wind_model.velocity_field.T[0],
                        wind_model.velocity_field.T[1], width=0.003)
    # expand axis limits to make vectors at boundary of field visible
    ax.axis(ax.axis() + np.array([-0.25, 0.25, -0.25, 0.25]))
    ax.set_xlabel('x-coordinate / m')
    ax.set_ylabel('y-coordinate / m')
    ax.set_aspect(1)
    fig.tight_layout()

    # define update function
    @update_decorator(dt, title, steps_per_frame, [wind_model])
    def update(i):
        vf_plot.set_UVC(
            wind_model.velocity_field.T[0], wind_model.velocity_field.T[1])
        return [vf_plot]

    # create animation object
    n_frame = int(t_max / (dt * steps_per_frame) + 0.5)
    anim = FuncAnimation(fig, update, n_frame, blit=True)
    return fig, ax, anim


def plume_model_demo(dt=0.01, t_max=100, steps_per_frame=200,
                     seed=DEFAULT_SEED):
    """Set up plume model and animate puffs overlayed over velocity field.

    Puff positions displayed using Matplotlib `scatter` plot function and
    velocity field displayed using `quiver` plot function.
    plot and quiver functions.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    t_max : float
        End time to simulate to.
    steps_per_frame: integer
        Number of simulation time steps to perform between animation frames.
    seed : integer
        Seed for random number generator.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : AxesSubplot
        Matplotlib axis object.
    anim : FuncAnimation
        Matplotlib animation object.
    """
    rng = np.random.RandomState(seed)
    # define simulation region
    sim_region = models.Rectangle(x_min=0., x_max=100, y_min=-25., y_max=25.)
    # set up wind model
    wind_model = models.WindModel(sim_region, 21, 11, rng=rng)
    # let simulation run for 10s to equilibrate wind model
    for t in np.arange(0, 10, dt):
        wind_model.update(dt)
    # set up plume model
    plume_model = models.PlumeModel(
        sim_region, (5., 0., 0.), wind_model, rng=rng)
    # set up figure window
    fig, ax, title = set_up_figure()
    # create quiver plot of initial velocity field
    # quiver expects first array dimension (rows) to correspond to y-axis
    # therefore need to transpose
    vf_plot = plt.quiver(
        wind_model.x_points, wind_model.y_points,
        wind_model.velocity_field.T[0], wind_model.velocity_field.T[1],
        width=0.003)
    # expand axis limits to make vectors at boundary of field visible
    ax.axis(ax.axis() + np.array([-0.25, 0.25, -0.25, 0.25]))
    # draw initial puff positions with scatter plot
    radius_mult = 200
    pp_plot = plt.scatter(
        plume_model.puff_array[:, 0], plume_model.puff_array[:, 1],
        radius_mult * plume_model.puff_array[:, 3]**0.5, c='r',
        edgecolors='none')
    ax.set_xlabel('x-coordinate / m')
    ax.set_ylabel('y-coordinate / m')
    ax.set_aspect(1)
    fig.tight_layout()

    # define update function
    @update_decorator(dt, title, steps_per_frame, [wind_model, plume_model])
    def update(i):
        # update velocity field quiver plot data
        vf_plot.set_UVC(wind_model.velocity_field[:, :, 0].T,
                        wind_model.velocity_field[:, :, 1].T)
        # update puff position scatter plot positions and sizes
        pp_plot.set_offsets(plume_model.puff_array[:, :2])
        pp_plot._sizes = radius_mult * plume_model.puff_array[:, 3]**0.5
        return [vf_plot, pp_plot]

    # create animation object
    n_frame = int(t_max / (dt * steps_per_frame) + 0.5)
    anim = FuncAnimation(fig, update, frames=n_frame, blit=True)
    return fig, ax, anim


def conc_point_val_demo(dt=0.01, t_max=5, steps_per_frame=1, x=10., y=0.0,
                        seed=DEFAULT_SEED):
    """Set up plume model and animate concentration at a point as time series.

    Demonstration of setting up plume model and processing the outputted
    puff arrays with the ConcentrationPointValueCalculator class, the
    resulting concentration time course at a point in the odour plume being
    displayed with the Matplotlib `plot` function.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    t_max : float
        End time to simulate to.
    steps_per_frame: integer
        Number of simulation time steps to perform between animation frames.
    x : float
        x-coordinate of point to measure concentration at.
    y : float
        y-coordinate of point to measure concentration at.
    seed : integer
        Seed for random number generator.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : AxesSubplot
        Matplotlib axis object.
    anim : FuncAnimation
        Matplotlib animation object.
    """
    rng = np.random.RandomState(seed)
    # define simulation region
    sim_region = models.Rectangle(x_min=0., x_max=100, y_min=-25., y_max=25.)
    # set up wind model
    wind_model = models.WindModel(sim_region, 21, 11, rng=rng)
    # set up plume model
    plume_model = models.PlumeModel(
        sim_region, (5., 0., 0.), wind_model, rng=rng)
    # let simulation run for 10s to initialise models
    for t in np.arange(0, 10, dt):
        wind_model.update(dt)
        plume_model.update(dt)
    # set up concentration point value calculator
    val_calc = processors.ConcentrationValueCalculator(1.)
    conc_vals = []
    conc_vals.append(val_calc.calc_conc_point(plume_model.puff_array, x, y))
    ts = [0.]
    # set up figure
    fig, ax, title = set_up_figure()
    # display initial concentration field as image
    conc_line, = plt.plot(ts, conc_vals)
    ax.set_xlim(0., t_max)
    ax.set_ylim(0., 150.)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Normalised concentration')
    ax.grid(True)
    fig.tight_layout()

    # define update function
    @update_decorator(dt, title, steps_per_frame, [wind_model, plume_model])
    def update(i):
        ts.append(dt * i * steps_per_frame)
        conc_vals.append(
            val_calc.calc_conc_point(plume_model.puff_array, x, y))
        conc_line.set_data(ts, conc_vals)
        return [conc_line]

    # create animation object
    n_frame = int(t_max / (dt * steps_per_frame) + 0.5)
    anim = FuncAnimation(fig, update, frames=n_frame, blit=True)
    return fig, ax, anim


def concentration_array_demo(dt=0.01, t_max=100, steps_per_frame=50,
                             seed=DEFAULT_SEED):
    """Set up plume model and animate concentration fields.

    Demonstration of setting up plume model and processing the outputted
    puff arrays with the `ConcentrationArrayGenerator` class, the resulting
    arrays being displayed with the Matplotlib `imshow` function.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    t_max : float
        End time to simulate to.
    steps_per_frame: integer
        Number of simulation time steps to perform between animation frames.
    seed : integer
        Seed for random number generator.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : AxesSubplot
        Matplotlib axis object.
    anim : FuncAnimation
        Matplotlib animation object.
    """
    rng = np.random.RandomState(seed)
    # define simulation region
    sim_region = models.Rectangle(x_min=0., x_max=100, y_min=-25., y_max=25.)
    # set up wind model
    wind_model = models.WindModel(sim_region, 21, 11, rng=rng)
    # set up plume model
    plume_model = models.PlumeModel(
        sim_region, (5., 0., 0.), wind_model, rng=rng)
    # let simulation run for 10s to initialise models
    for t in np.arange(0, 10, dt):
        wind_model.update(dt)
        plume_model.update(dt)
    # set up concentration array generator
    array_gen = processors.ConcentrationArrayGenerator(
        sim_region, 0.01, 500, 250, 1.)
    # set up figure
    fig, ax, title = set_up_figure()
    # display initial concentration field as image
    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    conc_im = plt.imshow(conc_array.T, extent=sim_region, cmap='Reds',
                         vmin=0., vmax=1.)
    ax.set_xlabel('x-coordinate / m')
    ax.set_ylabel('y-coordinate / m')
    ax.set_aspect(1)
    fig.tight_layout()

    # define update function
    @update_decorator(dt, title, steps_per_frame, [wind_model, plume_model])
    def update(i):
        conc_im.set_data(
            array_gen.generate_single_array(plume_model.puff_array).T)
        return [conc_im]

    # create animation object
    n_frame = int(t_max / (dt * steps_per_frame) + 0.5)
    anim = FuncAnimation(fig, update, frames=n_frame, blit=True)
    return fig, ax, anim
