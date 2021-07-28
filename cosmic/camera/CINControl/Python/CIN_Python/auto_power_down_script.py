#! /usr/bin/python
# -*- coding: utf-8 -*-

import cin_constants
import cin_register_map
import cin_functions
import time

print "\n  Shutting down CCD Bias and Clocks"
cin_functions.setCameraOff()

#print "\n  Shutting down Camera Power Supply"
#import setMainPS1_Off

print "\n  Shutting down Camera Interface Node Blade & FO Interface Boards"
import set_FOPS_Off
cin_functions.CINPowerDown()
#
#print " "
#raw_input("(Press Any Key to Exit)")
