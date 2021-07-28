#! /usr/bin/python
# -*- coding: utf-8 -*-

import cin_constants
import cin_register_map
import cin_functions
import time

import setMainPS1_On
time.sleep(2)  # Wait to allow visual check

cin_functions.loadCameraConfigFile("/home/user/CameraControl/CIN_Python/FCRIC-Test/fcrics_config_x8_125MHz.txt")

import setFCRIC_Clamped
import setFCRIC_Normal

# Mask 1 half of the sensor lines FCRIC to CIN 
#cin_functions.WriteReg("8211", "FFFF", 1) # 
#cin_functions.WriteReg("8212", "FF00", 1) #
#cin_functions.WriteReg("8213", "0000", 1) #
#raw_input("\nConfiguration Data sent to all fCRICs  (Press Enter Key to Exit)")

