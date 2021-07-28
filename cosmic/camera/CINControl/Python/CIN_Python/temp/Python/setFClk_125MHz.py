#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_constants
import cin_register_map
import cin_functions
import time

def writeFClkReg(addr, data):
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, addr, 0)
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, data, 0)
	cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

def FreezeDCO():
	# Freeze DCO
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F010", 0)
	cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	print "  Freeze Si570 DCO"

def UnFreezeDCO():
	# UnFreeze DCO
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F000", 0)
	cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B087", 0)
	cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F040", 0)
	cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	print "  UnFreeze Si570 DCO & Start Oscillator\n"

setFreq = str(125)
# setFreq = str(sys.argv[1])
#print setFreq
print "\n**** Set CIN FCLK to " + setFreq + "MHz"

if (setFreq == "125") : 
	FreezeDCO()
	writeFClkReg("B007", "F002")  # Bits[7:0] == HS_DIV[2:0] & N1[6:2]
	writeFClkReg("B008", "F042")  # Bits[7:0] == N1[1:0] & RFREQ[37:32]
	writeFClkReg("B009", "F0BC")  # Bits[7:0] == RFREQ[31:24]
	writeFClkReg("B00A", "F019")  # Bits[7:0] == RFREQ[23:16]
	writeFClkReg("B00B", "F06D")  # Bits[7:0] == RFREQ[15:8]
	writeFClkReg("B00C", "F08F")  # Bits[7:0] == RFREQ[7:0]
 	UnFreezeDCO()

elif (setFreq == "180") :
	FreezeDCO()
	writeFClkReg("B007", "F060")
	writeFClkReg("B008", "F0C2")
	writeFClkReg("B009", "F0C1")
	writeFClkReg("B00A", "F0B9")
	writeFClkReg("B00B", "F08A")
	writeFClkReg("B00C", "F0EF")
 	UnFreezeDCO()

elif (setFreq == "200") :
	FreezeDCO()
	writeFClkReg("B007", "F060")
	writeFClkReg("B008", "F0C3")
	writeFClkReg("B009", "F010")
	writeFClkReg("B00A", "F023")
	writeFClkReg("B00B", "F07D")
	writeFClkReg("B00C", "F0ED")
 	UnFreezeDCO()

elif (setFreq == "250") :
	FreezeDCO()
	writeFClkReg("B007", "F020")
	writeFClkReg("B008", "F0C2")
	writeFClkReg("B009", "F0BC")
	writeFClkReg("B00A", "F019")
	writeFClkReg("B00B", "F06D")
	writeFClkReg("B00C", "F08F")
 	UnFreezeDCO()

else : 
	print "Invalid FCLK Frequency\n"  
	print "Currently only 125MHz, 180MHz, 200MHz and 250MHz are Supported\n"

