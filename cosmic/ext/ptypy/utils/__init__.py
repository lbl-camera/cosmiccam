# -*- coding: utf-8 -*-
"""
Util sub-package

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .misc import *
from .math_utils import *
from .array_utils import *
from .scripts import *
from .parameters import *
from .verbose import *

"""
from .. import __has_matplotlib__ as hmpl
if hmpl:
    from plot_utils import *
del hmpl
"""

from . import parallel

