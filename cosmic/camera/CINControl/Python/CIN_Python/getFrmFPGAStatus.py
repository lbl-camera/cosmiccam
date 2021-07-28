#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time


#def getFrmFpgaStat():
# get Status Registers
print "**** Frame FPGA Status Registers ****\n "
# Test if FRAME FPGA is configured
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_FPGA_STATUS)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
if (int(stats_vec[-16]) == 1):
#	reg_val = cin_functions.ReadReg( cin_register_map.REG_FRM_BOARD_ID )
#	print "  CIN Board ID     : " + reg_val[4:]
	reg_val = cin_functions.ReadReg( cin_register_map.REG_FRM_FPGA_VERSION )
	print "  FRM FPGA Version : " + reg_val[4:]
	reg_val = cin_functions.ReadReg( cin_register_map.REG_FRM_FPGA_STATUS )
	print "  FPGA Status      : " + reg_val[4:]
#	print "    frm_10gbe_port_sel_i     : b15"
#	print "    pixel_clk_sel            : b0"
	reg_val = cin_functions.ReadReg( cin_register_map.REG_FRM_DCM_STATUS )
	print "  DCM Status       : " + reg_val[4:]
#	print "    xaui_align_status_fab2   : b12"
#	print "    xaui_txlock_fab2         : b11"
#	print "    mac_tx_ll_dst_rdy_n_fab2 : b10"
#	print "    mac_rx_dcm_locked_fab2   : b09"
#	print "    tx_mmcm_locked_fab2      : b08"
#	print " "
#	print "    xaui_align_status_fab1   : b04"
#	print "    xaui_txlock_fab1         : b03"
#	print "    mac_tx_ll_dst_rdy_n_fab1 : b02"
#	print "    mac_rx_dcm_locked_fab1   : b01"
#	print "    tx_mmcm_locked_fab1;     : b00"
#	print " "
else:
	print "  Frame FPGA NOT Configured"

