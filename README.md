## `pompy` - *p*uff-based *o*dour plume *m*odel in *Py*thon

`pompy` is a NumPy based implementation of the puff-based odour plume model described in [*Filament-Based Atmospheric Dispersion Model to Achieve Short Time-Scale Structure of Odor Plumes*](http://link.springer.com/article/10.1023%2FA%3A1016283702837#page-1) by Farrell et al. (2002).

### Introduction

It was developed by [Matt Graham](http://matt-graham.github.io/) as part of a MSc by Research project in the Insect Robotics group of in the School of Informatics, University of Edinburgh, into whether and how insects are able to use olfactory landmarks to aid in spatial navigation. This project was motivated by the work of Kathrin Steck and colleagues who discovered that *Cataglyphis fortis*, a Saharan desert ant species, are able to use odour sources in their environment as 'landmarks' to help when navigating back to their nest.

### What is this repository for?

This `python` package allows simulation of dynamic 2D odour concentration fields which show some of the key characteristics of real chemical plumes in turbulent flows including short term intermittency, diffusive effects and longer term variations in spatial extent and location, while being significantly cheaper to run than a full fluid dynamics simulation.

### Installation and requirements

`pompy` was developed in Python 2.7. For basic usage of the models the two dependencies are NumPy and SciPy. For the demonstrations Matplotlib is also required. To install in the current Python environment run

```
python setup.py install
```

### Example usage

See also `pompy/demos.py` module.

```python
from pompy import models, processors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define simulation region
wind_region = models.Rectangle(0., -2., 10., 2.)
sim_region = models.Rectangle(0., -0.5, 2., 0.5)

# Set up wind model
wind_grid_dim_x = 21
wind_grid_dim_y = 11
wind_vel_x_av = 2.
wind_vel_y_av = 0.
wind_model = models.WindModel(
    wind_region, wind_grid_dim_x, wind_grid_dim_y, 
    wind_vel_x_av, wind_vel_y_av)
    
# Set up plume model
source_pos = (0.1, 0., 0.)
centre_rel_diff_scale = 1.5
puff_release_rate = 500
puff_init_rad = 0.001
plume_model = models.PlumeModel(
    sim_region, source_pos, wind_model,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate, 
    puff_init_rad=puff_init_rad)

# Create a concentration array generator
array_z = 0.01
array_dim_x = 500
array_dim_y = 500
puff_mol_amount = 1.
array_gen = processors.ConcentrationArrayGenerator(
    sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)

# Set up figure
fig = plt.figure(figsize=(6, 3))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')

# Display initial concentration field as image
conc_array = array_gen.generate_single_array(plume_model.puff_array)
im_extents = (sim_region.x_min, sim_region.x_max,
              sim_region.y_min, sim_region.y_max)
conc_im = ax.imshow(conc_array.T, extent=im_extents, vmin=0., vmax=5e4, cmap='Reds')

# Define animation update function
def update(i):
    dt = 0.005
    wind_model.update(dt)
    plume_model.update(dt)
    conc_array = array_gen.generate_single_array(plume_model.puff_array)
    conc_im.set_data(conc_array.T)
    return [conc_im]

# Animate plume concentration and save as fig
anim = FuncAnimation(fig, update, frames=100, interval=100, repeat=False)
anim.save('plume.gif', dpi=100, writer='imagemagick')
```
