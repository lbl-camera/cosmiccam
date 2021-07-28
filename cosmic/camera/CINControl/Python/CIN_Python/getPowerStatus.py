#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

def current_calc(reg_val, current):
	if (int(reg_val[4:8],16) >= int("8000",16)):
#	  current = 0.000000238*((int("10000",16) - int(reg_val[4:8],16)))/0.003
	  current = 0.000000476*((int("10000",16) - int(reg_val[4:8],16)))/0.003
	else:
#	  current = 0.000000238*(int(reg_val[4:8],16))/0.003
	  current = 0.000000476*(int(reg_val[4:8],16))/0.003
	return current

#def getPowerStatus():
print " "
print "****  CIN Power Monitor  ****\n"
reg_val = bin((int(cin_functions.ReadReg( cin_register_map.REG_PS_ENABLE)[4:8],16)))[2:].zfill(16)
stats_vec = reg_val[:]
if (int(stats_vec[-1]) == 1):

# ADC == LT4151
	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC1_CH1 )
	voltage = 0.025*int(reg_val[4:8],16)
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC1_CH0 )
	current = 0.00002*int(reg_val[4:8],16)/0.003
	power = voltage * current
	print "V12P_BUS Power  : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A \n"

# ADC == LT2418
	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH5 )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH5 )
	current = current_calc(reg_val,current)
	print "V3P3_MGMT Power : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH7 )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH7 )
	current = current_calc(reg_val,current)
	print "V2P5_MGMT Power : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

#	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH6 )
#	voltage = 0.00007629*(int(reg_val[4:8],16))
#	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH6 )
#	current = current_calc(reg_val,current)
#	print "  V1P8_MGMT Power : {0:.4s}".format(str(voltage)) + "V  @  {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH2 )
	voltage = 0.00007629*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH2 )
	current = current_calc(reg_val,current)
	print "V1P2_MGMT Power : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH3 )
	voltage = 0.00007629*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH3 )
	current = current_calc(reg_val,current)
	print "V1P0_ENET Power : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A\n"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH4 )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH4 )
	current = current_calc(reg_val,current)
	print "V3P3_S3E Power  : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH8 )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH8 )
	current = current_calc(reg_val,current)
	print "V3P3_GEN Power  : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CH9 )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CH9 )
	current = current_calc(reg_val,current)
	print "V2P5_GEN Power  : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A\n"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CHE )
	voltage = 0.00007629*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CHE )
	current = current_calc(reg_val,current)
	print "V0P9_V6 Power   : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CHB )
	voltage = 0.00007629*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CHB )
	current = current_calc(reg_val,current)
	print "V1P0_V6 Power   : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

#	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CHC )
#	voltage = 0.00007629*(int(reg_val[4:8],16))
#	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CHC )
#	current = current_calc(reg_val,current)
#	print "  V1P2_V6 Power   : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CHD )
	voltage = 0.00015258*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CHD )
	current = current_calc(reg_val,current)
	print "V2P5_V6 Power   : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A\n"

	reg_val = cin_functions.ReadReg( cin_register_map.REG_VMON_ADC0_CHF )
	voltage = 0.00030516*(int(reg_val[4:8],16))
	reg_val = cin_functions.ReadReg( cin_register_map.REG_IMON_ADC0_CHF )
	current = current_calc(reg_val,current)
	print "V_FP Power      : {0:.4s}".format(str(voltage)) + "V @ {0:.5s}".format(str(current)) + "A"
else:
	print "  12V Power Supply is OFF"

