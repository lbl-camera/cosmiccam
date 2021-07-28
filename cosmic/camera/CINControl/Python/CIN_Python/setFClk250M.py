#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

#def setFCLK250M():
print "**** Set CIN FCLK to 250MHz"
# Freeze DCO
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F010", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
print "  Write to Reg 137 - Freeze DCO"

print "  Set Si570 Oscillator Freq to 250MHz"
# WR Reg 7
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B007", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F020", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# WR Reg 8
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B008", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F0C2", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# WR Reg 9
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B009", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F0BC", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# WR Reg 10
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B00A", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F019", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# WR Reg 11
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B00B", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F06D", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# WR Reg 12
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B00C", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F08F", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)

# UnFreeze DCO
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B089", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F000", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B087", 0)
cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_DATA_WR, "F040", 0)
cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
print "  Write to Reg 137 - UnFreeze DCO & Start Oscillator"
print "  "

# Set Clock&Bias Time Contant
cin_functions.WriteReg( cin_register_map.REG_CCDFCLKSELECT_REG, "0001", 0)


