#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_functions
import cin_register_map

# Convert us to Hex - 1 count = 1us
time_d = int(float(sys.argv[1]))
#print time_d
time_h = str(hex(time_d)).lstrip("0x").zfill(8)
#print time_h
#print time_h[4:]
#print time_h[0:4]

# Write Number of Exposure Time MSB
cin_functions.WriteReg(cin_register_map.REG_DELAYTOSHUTTERMSB_REG, time_h[0:4], 1)
# Write Number of Exposure Time LSB
cin_functions.WriteReg(cin_register_map.REG_DELAYTOSHUTTERLSB_REG, time_h[4:], 1)

