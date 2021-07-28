#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_functions
import cin_register_map

cin_functions.clearFocusBit()

# Convert Decimal to Hex string
num_exp = str(hex(int(sys.argv[1])).lstrip("0x")).zfill(4)
#print num_exp
# Write Number of Exposures to CIN
cin_functions.WriteReg(cin_register_map.REG_NUMBEROFEXPOSURE_REG, num_exp, 1)
#Debug...
#reg_val = cin_functions.ReadReg(cin_register_map.REG_NUMBEROFEXPOSURE_REG)
#print reg_val
if (int(sys.argv[1]) > 1):
	cin_functions.setFocusBit()

