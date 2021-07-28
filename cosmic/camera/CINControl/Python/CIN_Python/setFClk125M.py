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
        time.sleep(0.2)

def FreezeDCO():
        # Freeze DCO
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F010", 0)
        cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
        time.sleep(0.2)
        print "  Freeze Si570 DCO"

def UnFreezeDCO():
        # UnFreeze DCO
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F000", 0)
        cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
        time.sleep(0.2)
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_ADDRESS, "B087", 0)
        cin_functions.WriteReg(cin_register_map.REG_FCLK_I2C_DATA_WR, "F040", 0)
        cin_functions.WriteReg(cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
        time.sleep(0.2)
        print "  UnFreeze Si570 DCO & Start Oscillator\n"

print "\n**** Set CIN FCLK to 125MHz"

FreezeDCO()
writeFClkReg("B007", "F002")  # Bits[7:0] == HS_DIV[2:0] & N1[6:2]
writeFClkReg("B008", "F042")  # Bits[7:0] == N1[1:0] & RFREQ[37:32]
writeFClkReg("B009", "F0BC")  # Bits[7:0] == RFREQ[31:24]:
writeFClkReg("B00A", "F019")  # Bits[7:0] == RFREQ[23:16]
writeFClkReg("B00B", "F06D")  # Bits[7:0] == RFREQ[15:8]
writeFClkReg("B00C", "F08F")  # Bits[7:0] == RFREQ[7:0]
UnFreezeDCO()

