#! /usr/bin/python
# -*- coding: utf-8 -*-

import cin_constants
import cin_register_map
import cin_functions
import time
from optparse import OptionParser

cin_functions.CINPowerDown()
cin_functions.CINPowerUp()

import getCfgFPGAStatus

# Baseline CIN FW and Clock Configuration
cin_functions.loadFrmFirmware('/global/groups/cosmic/CINControl/firmware/CIN_1kFSCCD/CIN_1kFSCCD_ISE14_7_top_302E.bit')
import setFClk125M

# Set External Trigger on both input ports
import setTriggerOR

#import getFClkStatus
#import getFrmFPGAStatus

import setFPPowerOn
import set_FOPS_On
import getPowerStatus

print "Loading WAVEFORM config file..."
cin_functions.loadCameraConfigFile("/global/groups/cosmic/CINControl/CINController/config/20180511_125MHz_fCCD_Timing_FS_2OS_xper.txt")
#cin_functions.loadCameraConfigFile("/global/groups/cosmic/CINControl/CINController/config/20170913_125MHz_fCCD_Timing_2OS_xper.txt")

parser = OptionParser()
parser.add_option("--test", dest = 'testOnly', default = 0, action = "store_true", help = "Generate test pattern only")

##Get the parser args and convert to a dict
(options, args) = parser.parse_args()
options = vars(options)

if not(options['testOnly']):
	print "Loading BIAS and FCRIC config files..."
	cin_functions.loadCameraConfigFile("/home/nsptycho/CINControl/QT/CINController/config/bias_setting_lbl_gold4_8000.txt")
	cin_functions.loadCameraConfigFile("/home/nsptycho/CINControl/QT/CINController/config/ARRA_fcrics_config_x1_11112011_8000.txt")
	import sendConnect

