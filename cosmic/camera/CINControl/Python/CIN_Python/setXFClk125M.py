#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

#def setXFCLK125M():
print "**** Set CIN FCLK to 125MHz"
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "B000", 0)
print "  "

# Set Clock&Bias Time Contant
cin_functions.WriteReg( cin_register_map.REG_CCDFCLKSELECT_REG, "0000", 0)


