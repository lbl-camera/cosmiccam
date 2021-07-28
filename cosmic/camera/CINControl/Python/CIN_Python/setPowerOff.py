#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time


#def setPowerOff():
print " "
print "Powering Off CIN Board ........  "
if cin_functions.WriteReg( cin_register_map.REG_PS_ENABLE, "0000", 1) != 1:
	print 'Write register could not be verified. Aborting.'
	sys.exit(1)
cin_functions.WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 0)
time.sleep(2)


