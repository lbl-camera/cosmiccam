#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time


#def getFabricEthStatus():
print " "
print "*******************  10GbE Fabric Ethernet Status  ***************** "
print " "
print "--- MAC Configuration: "

reg_val = cin_functions.ReadReg( cin_register_map.REG_IF_MAC_FAB1B2)[4:8]
reg_val = reg_val + cin_functions.ReadReg( cin_register_map.REG_IF_MAC_FAB1B1)[4:8]
reg_val = reg_val + cin_functions.ReadReg( cin_register_map.REG_IF_MAC_FAB1B0)[4:8]

print "MAC Address: " + reg_val[0:2] + ":" + reg_val[2:4] + ":" + reg_val[4:6] + ":" + reg_val[6:8] + ":" + reg_val[8:10] + ":" + reg_val[10:12]

reg_val = cin_functions.ReadReg( cin_register_map.REG_IF_IP_FAB1B1)[4:8]
reg_val = reg_val + cin_functions.ReadReg( cin_register_map.REG_IF_IP_FAB1B0)[4:8]

print "IP Address: " + str(int(reg_val[0:2],16)) + "." + str(int(reg_val[2:4],16)) + "." + str(int(reg_val[4:6],16)) + "." + str(int(reg_val[6:8],16))

reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_CONFIG_VEC_FAB1B1)[4:8],16)))[2:].zfill(16)
reg_val = reg_val + bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_CONFIG_VEC_FAB1B0)[4:8],16)))[2:].zfill(16)

mac_config_vec = reg_val[:]

print "MAC Configuration: " + mac_config_vec
print mac_config_vec[-21] + " : Control Frame Length Check Disable"
print mac_config_vec[-20] + " : Receiver Length/Type Error Disable"
print mac_config_vec[-19] + " : Receiver Preserve Preamble Enable"
print mac_config_vec[-18] + " : Transmitter Preserve Preamble Enable"
print mac_config_vec[-17] + " : Reconciliator Sublayer Fault Inhibit"
print mac_config_vec[-16] + " : Reserved"
print mac_config_vec[-15] + " : Deficite Idle Cont Enable"
print mac_config_vec[-14] + " : TX Flow Control"
print mac_config_vec[-13] + " : RX Flow Control"
print mac_config_vec[-12] + " : TX Reset"
print mac_config_vec[-11] + " : TX Jumbo Enable"
print mac_config_vec[-10] + " : TX FCS Enable"
print mac_config_vec[ -9] + " : TX Enable"
print mac_config_vec[ -8] + " : TX VLAN Enable"
print mac_config_vec[ -7] + " : Adjustable Frame Gaps"
print mac_config_vec[ -6] + " : Large Frame Gaps"
print mac_config_vec[ -5] + " : RX Reset"
print mac_config_vec[ -4] + " : RX Jumbo Enable"
print mac_config_vec[ -3] + " : Pass FCS Enable"
print mac_config_vec[ -2] + " : RX Enable"
print mac_config_vec[ -1] + " : RX VLAN Enable"
 
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_STATS1_FAB1B1)[4:8],16)))[2:].zfill(16)
reg_val = reg_val + bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_STATS1_FAB1B0)[4:8],16)))[2:].zfill(16)

stats1_vec = reg_val[:]

print " "
print "MAC TX Stats: " + stats1_vec
# print stats1_vec[-32] + " : " # Not Used
print stats1_vec[-31] + " : XAUI Sync[3]" # XAUI SYNC
print stats1_vec[-30] + " : XAUI Sync[1]"
print stats1_vec[-29] + " : XAUI Sync[1]"
print stats1_vec[-28] + " : XAUI Sync[0]"
print stats1_vec[-27] + " : MAC Remote Fault Received" # MAC Status Vector
print stats1_vec[-26] + " : MAC Local Fault Received" # 
print str(int(stats1_vec[-25:-20],2)) + " : TX Bytes Valid on TX Clock" # MAC TX Statistics Vector (downto -1)
print stats1_vec[-20] + " : TX Previous Frame Was a VLAN Frame"
print str(int(stats1_vec[-19:-5],2)) + " : TX Previous Frame Length"
print stats1_vec[ -5] + " : TX Previous Frame Was a Control Frame"
print stats1_vec[ -4] + " : TX Previous Frame Terminated due to underrun"
print stats1_vec[ -3] + " : TX Previous Frame Was a Multicast Frame"
print stats1_vec[ -2] + " : TX Previous Frame Was a Broadcast Frame"
print stats1_vec[ -1] + " : TX Previous Frame Transmited without Error" 

reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_STATS2_FAB1B1)[4:8],16)))[2:].zfill(16)
reg_val = reg_val + bin((int(cin_functions.ReadReg( cin_register_map.REG_MAC_STATS2_FAB1B0)[4:8],16)))[2:].zfill(16)

stats2_vec = reg_val[:]

print " "
print "MAC RX Stats: " + stats2_vec
# print stats1_vec[-32] + " : " # Not Used
print stats2_vec[-30] + " : XAUI TX Ready"
print stats2_vec[-29] + " : RX Header Length/Type Does Not Match Data Length"
print stats2_vec[-28] + " : RX Frame Had Unsuported OP-Code" 
print stats2_vec[-27] + " : RX Previous Frame Was a Flow Control Frame"
print str(int(stats2_vec[-26:-22],2)) + " : RX Bytes Valid on RX Clock"
print stats2_vec[-22] + " : RX Previous Frame Was a VLAN Frame"
print stats2_vec[-21] + " : RX Previous Frame Was Out of Bounds (too long)"
print stats2_vec[-20] + " : RX Previous Frame Was a Control Frame"
print str(int(stats2_vec[-19:-5],2)) + " : RX Previous Frame Length"
print stats2_vec[ -5] + " : RX Previous Frame Was a Multicast Frame"
print stats2_vec[ -4] + " : RX Previous Frame Was a Broadcast Frame"
print stats2_vec[ -3] + " : RX Previous FCS Errors or MAC Code Errors"
print stats2_vec[ -2] + " : RX Previous Frame Received WITH Errors"
print stats2_vec[ -1] + " : RX Previous Frame Received without Errors" 

reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_XAUI_FAB1B)[4:8],16)))[2:].zfill(16)

xaui_stats_vec = reg_val[:]

print " "
print "XAUI Stats/Configuration: " + xaui_stats_vec
print xaui_stats_vec[-16] + " : XAUI RX Link Status"
print xaui_stats_vec[-15] + " : XAUI Alignment"  
print xaui_stats_vec[-14] + " : XAUI Sync[3]"
print xaui_stats_vec[-13] + " : XAUI Sync[2]"
print xaui_stats_vec[-12] + " : XAUI Sync[1]"
print xaui_stats_vec[-11] + " : XAUI Sync[0]"
print xaui_stats_vec[-10] + " : XAUI TX Local Fault"
print xaui_stats_vec[ -9] + " : XAUI RX Local Fault"

print xaui_stats_vec[ -7:-5] + " : XAUI Test Mode Testpatern: (00):High (01):Low (10):Mixed"
print xaui_stats_vec[ -5] + " : XAUI Enable Test Mode"
print xaui_stats_vec[ -4] + " : XAUI Reset RX Link Status"
print xaui_stats_vec[ -3] + " : XAUI Reset Local Fault"
print xaui_stats_vec[ -2] + " : XAUI Power Down"
print xaui_stats_vec[ -1] + " : XAUI Loopback Mode"

cin_functions.WriteReg( cin_register_map.REG_XAUI_FAB1B, "000C", 1)
cin_functions.WriteReg( cin_register_map.REG_XAUI_FAB1B, "0000", 1)

cin_functions.WriteReg( cin_register_map.REG_MAC_CONFIG_VEC_FAB1B0, "0D9B", 1)
cin_functions.WriteReg( cin_register_map.REG_MAC_CONFIG_VEC_FAB1B0, "058B", 1)

