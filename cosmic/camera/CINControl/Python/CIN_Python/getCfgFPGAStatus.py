#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

# ---------------------------------------------< Configuration FPGA functions >
#def getCfgFpgaStat():
# get Status Registers
print "****  CFG FPGA Status Registers  **** \n"
reg_val = cin_functions.ReadReg( cin_register_map.REG_BOARD_ID)
print " CIN Board ID     :  " + reg_val[4:]
reg_val = cin_functions.ReadReg( cin_register_map.REG_HW_SERIAL_NUM )
print " HW Serial Number :  " + reg_val[4:]
reg_val = cin_functions.ReadReg( cin_register_map.REG_FPGA_VERSION )
print " CFG FPGA Version :  " + reg_val[4:] + "\n"
reg_val = cin_functions.ReadReg( cin_register_map.REG_FPGA_STATUS )
print " CFG FPGA Status  :  " + reg_val[4:]
# FPGA Status
# 15 == FRM DONE
# 14 == NOT FRM BUSY
# 13 == NOT FRM INIT B
# 12 >> 4 == 0
# 3 >>0 == FP Config Control 3 == PS Interlock
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_FPGA_STATUS)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
if (int(stats_vec[-16]) == 1):
	print "  ** Frame FPGA Configuration Done"
else:
	print "  ** Frame FPGA NOT Configured"
if (int(stats_vec[-4]) == 1):
	print "  ** FP Power Supply Unlocked"
else:
	print "  ** FP Power Supply Locked Off"
reg_val = cin_functions.ReadReg( cin_register_map.REG_DCM_STATUS )
print "\n CFG DCM Status   : " + reg_val[4:]
# DCM Status
# 15 == 0
# 14 >> 8 == CONF SW
# 7 == ATCA 48V Alarm
# 6 == tx2 src ready
# 5 == tx1 src ready
# 4 == DCM STATUS2
# 3 == DCM STATUS1
# 2 == DCM STATUS0
# 1 == DCM PSDONE
# 0 == DCM LOCKED
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_DCM_STATUS)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
if (int(stats_vec[-8]) == 1):
	print "  ** ATCA 48V Alarm"
else:
	print "  ** ATCA 48V OK"
if (int(stats_vec[-1]) == 1):
	print "  ** CFG Clock DCM Locked"
else:
	print "  ** CFG Clock DCM NOT Locked"
if (int(stats_vec[-12]) == 0):
	print "  ** FP Power Supply Interlock Overide Enable"

