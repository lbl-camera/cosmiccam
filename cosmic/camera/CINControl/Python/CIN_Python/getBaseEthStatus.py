#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time


#def getBaseEthStat():
print "**** CFG FPGA - Base (1GbE) Ethernet Status Registers "
print " "
# Get PHY1 Status register
cin_functions.WriteReg( cin_register_map.REG_PHY1_MDIO_CMD, "C001", 1)
time.sleep(0.1)
cin_functions.WriteReg( cin_register_map.REG_PHY1_MDIO_CMD, "0000", 1)
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_PHY1_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
print "  1GbE PHY1 Status Register : "
print stats_vec[-9] + " : Extended Status"
print stats_vec[-7] + " : MGMT Frame Preamble Suppression"
print stats_vec[-6] + " : Copper auto-negotiation complete"
print stats_vec[-5] + " : Copper remote fault detect"
print stats_vec[-4] + " : Auto-negotiation Enabled"
print stats_vec[-3] + " : Link Up"
print stats_vec[-2] + " : Jabber Detected"
print stats_vec[-1] + " : Extended Capability"
print " "

# Get PHY1 Extended Status register
cin_functions.WriteReg( cin_register_map.REG_PHY1_MDIO_CMD, "C00F", 1)
time.sleep(0.1)
cin_functions.WriteReg( cin_register_map.REG_PHY1_MDIO_CMD, "0000", 1)
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_PHY1_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
print "  1GbE PHY1 Extended Status Register : "
print stats_vec[-14] + " : Full Duplex 1000 Base"
print stats_vec[-13] + " : Half Duplex 1000 Base"
print " "
time.sleep(0.1)

# Get PHY2 Status register
cin_functions.WriteReg( cin_register_map.REG_PHY2_MDIO_CMD, "C001", 1)
time.sleep(0.1)
cin_functions.WriteReg( cin_register_map.REG_PHY2_MDIO_CMD, "0000", 1)
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_PHY2_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
print "  1GbE PHY2 Status Register : "
print stats_vec[-9] + " : Extended Status"
print stats_vec[-7] + " : MGMT Frame Preamble Suppression"
print stats_vec[-6] + " : Copper auto-negotiation complete"
print stats_vec[-5] + " : Copper remote fault detect"
print stats_vec[-4] + " : Auto-negotiation Enabled"
print stats_vec[-3] + " : Link Up"
print stats_vec[-2] + " : Jabber Detected"
print stats_vec[-1] + " : Extended Capability"
print " "

# Get PHY2 Extended Status register
cin_functions.WriteReg( cin_register_map.REG_PHY2_MDIO_CMD, "C00F", 1)
time.sleep(0.1)
cin_functions.WriteReg( cin_register_map.REG_PHY2_MDIO_CMD, "0000", 1)
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_PHY2_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
print "  1GbE PHY2 Extended Status Register : "
print stats_vec[-14] + " : Full Duplex 1000 Base"
print stats_vec[-13] + " : Half Duplex 1000 Base"
print " "
time.sleep(0.1)

#def DecodePHYStatus():
#	Main Status Bit Map
#	Bit 15 == 0
#	Bit 14 == 1 Full Duplex 100 Base
#	Bit 13 == 1 Half Duplex 100 Base
#	Bit 12 == 1 Full Duplex 10 Base
#	Bit 11 == 1 Half Duplex 10 Base
#	Bit 10 == 0
#	Bit  9 == 0
#	Bit  8 == 1 Extended Status
#	Bit  7 == 0
#	Bit  6 == 1 MGMT Frame Preamble Suppression
#	Bit  5 == 1 Copper auto-negotiation complete
#	Bit  4 == 1 Copper remote fault detect
#	Bit  3 == 1 Auto-negotiation Enabled
#	Bit  2 == 1 Link Up
#	Bit  1 == 1 Jabber Detected
#	Bit  0 == 1 Extended Capability

#	Extended Status Bit Map
#	Bit 15 == 0
#	Bit 14 == 0
#	Bit 13 == 1 Full Duplex 1000 Base
#	Bit 12 == 1 Half Duplex 1000 Base
#	Bit 11 >> 0 == 0x0


