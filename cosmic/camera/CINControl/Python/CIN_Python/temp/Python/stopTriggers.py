#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_functions
import cin_register_map

# Clear the Focus bit
cin_functions.clearFocusBit()

# Set Number of Exposures to value = 0001
cin_functions.WriteReg(cin_register_map.REG_NUMBEROFEXPOSURE_REG, "0001", 1)
