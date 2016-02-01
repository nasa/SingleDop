SingleDop README
----------------
SingleDop is a software module, written in the Python programming language, that will retrieve two-dimensional low-level winds from either real or simulated Doppler radar data. It mimics the functionality of the algorithm described in the following reference:
- Xu et al., 2006: Background error covariance functions for vector wind analyses using Doppler-radar radial-velocity observations. Q. J. R. Meteorol. Soc., 132, 2887-2904.  

The interface is simplified to a single line of code in the end user's Python scripts, making implementation of the algorithm in their research analyses very easy. The software package also interfaces well with other open source radar packages, such as the [Python ARM Radar Toolkit (Py-ART)](https://github.com/ARM-DOE/pyart). Simple visualization (including vector and contour plots) and save/load routines (to preserve analysis results) are also provided.

SingleDop Installation
----------------------
SingleDop works under Python 2.7 and 3.4 on most Mac/Linux setups. Windows installation and other Python versions are currently untested.

To install:  
`python setup.py install`

The following dependencies need to be installed first:

- A robust version of Python 2.7 or 3.4 w/ most standard scientific packages (e.g., numpy, matplotlib, scipy, etc.) - Get one for free here: https://store.continuum.io/cshop/anaconda/
- [The Python Atmospheric Radiation Measurement (ARM) Radar Toolkit (Py-ART)](https://github.com/ARM-DOE/pyart)
- [Python Turbulence Detection Algorithm (PyTDA)](https://github.com/nasa/PyTDA)
- [xray/xarray - optional](http://xarray.pydata.org/en/stable/)

Specific import calls in the SingleDop source code:
```
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import scipy
import math
import time
import warnings
import pickle
import pyart
from pytda import get_sweep_data, get_sweep_azimuths, get_sweep_elevations, \
                  flatten_and_reduce_data_array
from .common import radar_coords_to_cart
from .cmap_map import lighten_cmap
try:
    import xarray as xray
except ImportError:
    try:
        import xray
    except ImportError:
        warnings.warn(
            'xray/xarray not installed, save using SaveFile (pickle)')  
```

Using SingleDop
---------------
To access everything:
```
import singledop
```

A demonstration notebook is in the notebooks directory.
