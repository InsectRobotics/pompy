## PomPy - *p*uff-based *o*dour plume *m*odel in *Py*thon

[![Documentation Status](https://readthedocs.org/projects/pompy-docs/badge/?version=latest)](https://pompy-docs.readthedocs.io/en/latest/?badge=latest)

PomPy is a NumPy based implementation of the puff-based odour plume model described in [*Filament-Based Atmospheric Dispersion Model to Achieve Short Time-Scale Structure of Odor Plumes*](http://link.springer.com/article/10.1023%2FA%3A1016283702837#page-1) by Farrell et al. (2002).

![Plume model animation](plume.gif "Plume model animation example.")

### What is this repository for?

This Python package allows simulation of dynamic 2D odour concentration fields which show some of the key characteristics of real chemical plumes in turbulent flows including short term intermittency, diffusive effects and longer term variations in spatial extent and location, while being significantly cheaper to run than a full fluid dynamics simulation.

### Installation and requirements

PomPy was developed in Python 2.7 though it may be compatible with newer Python versions. For basic usage of the models the two dependencies are NumPy and SciPy. For the demonstrations in the `pompy.demo` module Matplotlib is also required. The [`requirements.txt`](requirements.txt) file lists versions of NumPy, SciPy and Matplotlib known to work with `pompy`. A Jupyter notebook server installation will also be required to run the example notebooks locally. 

To install PomPy in the current Python environment run

```
python setup.py install
```

and to install the Python requirements using `pip` run

```
pip install -r requirements.txt
```

### Documentation

Documentation of the `pompy` API is available at [Read the Docs](https://pompy-docs.readthedocs.io/en/latest/).

Two Jupyter notebooks showing examples of using the package are included in the repository root directory. The [`Farrell et al. (2002) example.ipynb`](Farrell%20et%20al.%20%282002%29%20example.ipynb) notebook file illustrates an example of using PomPy to generate an animation of an odour plume using the simulation parameters described in Farrell et al. (2002) and includes keys to match the symbols used to define the parameter in the paper with the relevant `pompy` class attributes. The [`Demonstration.ipynb`](Demonstrations.ipynb) and accompanying module `pompy.demos` give several other examples of setting up plume models using `pompy` and visualising the simulations.

The below script will generate a 20 second MP4 animation (see above for animated GIF version) of a generated plume with model parameters consistent with those proposed in the Farrell et al. (2002) paper.

```python
from pompy import models, processors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Seed random number generator
seed = 20180517
rng = np.random.RandomState(seed)

# Define wind model simulation region
wind_region = models.Rectangle(x_min=0., x_max=100., y_min=-50., y_max=50.)

# Define wind model parameters
wind_model_params = { 
    'n_x': 21,
    'n_y': 21,
    'u_av': 1.,
    'v_av': 0.,
    'k_x': 10.,
    'k_y': 10.,
    'noise_gain': 20.,
    'noise_damp': 0.1,
    'noise_bandwidth': 0.2,
    'use_original_noise_updates': True
}

# Create wind model object
wind_model = models.WindModel(wind_region, rng=rng, **wind_model_params)

# Define plume simulation region
# This is a subset of the wind simulation region
sim_region = models.Rectangle(x_min=0., x_max=50., y_min=-12.5, y_max=12.5)

# Define plume model parameters
plume_model_params = {
    'source_pos': (5., 0., 0.),
    'centre_rel_diff_scale': 2.,
    'puff_release_rate': 10,
    'puff_init_rad': 0.001**0.5,
    'puff_spread_rate': 0.001,
    'init_num_puffs': 10,
    'max_num_puffs': 1000,
    'model_z_disp': True,
}

# Create plume model object
plume_model = models.PlumeModel(
    rng=rng, sim_region=sim_region, wind_model=wind_model, **plume_model_params)

# Define concentration array (image) generator parameters
array_gen_params = {
    'array_z': 0.,
    'n_x': 500,
    'n_y': 250,
    'puff_mol_amount': 8.3e8
}

# Create concentration array generator object
array_gen = processors.ConcentrationArrayGenerator(
    array_xy_region=sim_region, **array_gen_params)
    
# Set up figure
fig = plt.figure(figsize=(5, 2.5))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')

# Display initial concentration field as image
conc_array = array_gen.generate_single_array(plume_model.puff_array)
conc_im = ax.imshow(
    conc_array.T, extent=sim_region, vmin=0., vmax=1e10, cmap='Reds')

# Simulation timestep
dt = 0.01

# Run wind model forward to equilibrate
for k in range(2000):
    wind_model.update(dt)

# Define animation update function
def update(i):
    # Do 10 time steps per frame update
    for k in range(10):
        wind_model.update(dt)
        plume_model.update(dt)
    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    conc_im.set_data(conc_array.T)
    return [conc_im]

# Animate plume concentration and save as MP4
anim = FuncAnimation(fig, update, frames=400, repeat=False)
anim.save('plume.mp4', dpi=100, fps=20, extra_args=['-vcodec', 'libx264'])
```
