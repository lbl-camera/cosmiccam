#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions

# Mask Triggers & turn off Bias
#import setTriggerSW
#cin_functions.setCameraOff()

# Write Gain x8
cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0086", 0 )
cin_functions.WriteReg ("821F", "0003", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

# Bias On & allow Ext Triggers
#cin_functions.setCameraOn()
#import setTrigger0

