#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A susbset of io and utils functions of PTYPY (github.com/ptycho/ptypy)
"""
try:
    import mpi4py
    __has_mpi4py__= True
    del mpi4py
except ImportError('Message Passaging for Python (mpi4py) not found.\n\
CPU-parallelization disabled.\n\
Install python-mpi4py via the package repositories or with `pip install --user mpi4py`'): 
    __has_mpi4py__= False

try:
    import matplotlib
    __has_matplotlib__= True
    del matplotlib
except ImportError('Plotting for Python (matplotlib) not found.\n\
Plotting disabled.\n\
Install python-matplotlib via the package repositories or with `pip install --user matplotlib`'):
    __has_matplotlib__= False
       
# Initialize MPI (eventually GPU)
from .utils import parallel

# Logging
from .utils import verbose
#verbose.set_level(2)

from . import utils
# Import core modules
from . import io 

