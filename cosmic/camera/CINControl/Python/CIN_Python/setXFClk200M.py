#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

#def setXFCLK200M():
print "**** Set CIN FCLK to 200MHz"
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "7000", 0)
print "  "

# Set Clock&Bias Time Contant
cin_functions.WriteReg( cin_register_map.REG_CCDFCLKSELECT_REG, "0002", 0)


