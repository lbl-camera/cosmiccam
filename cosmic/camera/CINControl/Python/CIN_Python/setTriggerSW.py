#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

cin_functions.WriteReg (cin_register_map.REG_TRIGGERMASK_REG, "0000", 0 )
