#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

#def getFCLK():

print "  "
print "**** CIN FCLK Configuration ****\n"

regval = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_WR )
print "  FCLK OSC MUX SELECT : 0x" + regval[4:]

if   (regval[4:5] == "F") :
	# Freeze DCO
	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B189", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )
	if (reg_val[6:] != "08") : print "  Status Reg : 0x" + reg_val[6:]

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B107", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val7 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B108", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val8 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B109", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val9 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B10A", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val10 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B10B", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val11 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	cin_functions.WriteReg( cin_register_map.REG_FCLK_I2C_ADDRESS, "B10C", 0)
	cin_functions.WriteReg( cin_register_map.REG_FRM_COMMAND, cin_register_map.CMD_FCLK_COMMIT, 0)
	reg_val12 = cin_functions.ReadReg( cin_register_map.REG_FCLK_I2C_DATA_RD )

	bin_reg7 = bin(int(reg_val7[6:],16))[2:].zfill(8)
	bin_reg8 = bin(int(reg_val8[6:],16))[2:].zfill(8)

	if (bin_reg7[0:3] == "000") : print "  FCLK HS Divider = 4"
	if (bin_reg7[0:3] == "001") : print "  FCLK HS Divider = 5"
	if (bin_reg7[0:3] == "010") : print "  FCLK HS Divider = 6"

	bin_n1 = bin_reg7[3:8] + bin_reg8[0:2]
	dec_n1 = int(bin_n1,2)

	if (dec_n1%2 != 0) : dec_n1 = dec_n1 + 1
	print "  FCLK N1 Divider = " + str(dec_n1)
	print "  FCLK RFREQ = " + reg_val8[7:] + reg_val9[6:7] + "." + reg_val9[7:] + reg_val10[6:] + reg_val11[6:] + reg_val12[6:]
	if   (bin_reg7[0:3] == "000" and dec_n1 ==  8) : print "  FCLK Frequency = 156 MHz"
	elif (bin_reg7[0:3] == "000" and dec_n1 == 10) : print "  FCLK Frequency = 125 MHz"
	elif (bin_reg7[0:3] == "001" and dec_n1 ==  4) : print "  FCLK Frequency = 250 MHz"
	else : print "  FCLK Frequency UNKNOWN "

#print str(int(regval[4:5],16)-1)

elif (str(int(regval[4:5],16)-1) == "2" ): print "  FCLK Frequency = 250 MHz"
elif (str(int(regval[4:5],16)-1) == "6" ): print "  FCLK Frequency = 200 MHz"
elif (str(int(regval[4:5],16)-1) == "10"): print "  FCLK Frequency = 125 MHz"

regval = cin_functions.ReadReg( cin_register_map.REG_CCDFCLKSELECT_REG )
print "\n  CCD TIMING CONSTANT : 0x" + regval[4:]
time.sleep(0.1)

