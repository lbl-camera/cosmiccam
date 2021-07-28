#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions

# Mask Triggers & turn off Bias
#import setTriggerSW
#cin_functions.setCameraOff()

# Write Gain x8
cin_functions.WriteReg ("821D", "9E00", 0 )
cin_functions.WriteReg ("821E", "0000", 0 )
cin_functions.WriteReg ("821F", "0000", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )


cin_functions.WriteReg ("8211", "FFFF", 0 )
cin_functions.WriteReg ("8212", "FFFF", 0 )
cin_functions.WriteReg ("8213", "FFFF", 0 )
