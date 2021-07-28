#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_functions
import cin_register_map

# Get Value from Trigger Select Register
reg_val = cin_functions.ReadReg(cin_register_map.REG_TRIGGERSELECT_REG)
#print reg_val[7:]
#print reg_val[6:7]
#print reg_val[5:6]
#print reg_val[4:5]

str_val = reg_val[4:5]+reg_val[5:6]+"D"+reg_val[7:] 
#print str_val

cin_functions.WriteReg(cin_register_map.REG_TRIGGERSELECT_REG, str_val, 1)

