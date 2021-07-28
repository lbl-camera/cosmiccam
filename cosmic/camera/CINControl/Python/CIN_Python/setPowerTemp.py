#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time


def CINPowerUp():
	print " "
	print "Powering On CIN Board ........  "
	if WriteReg( REG_PS_ENABLE, "000F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
#	time.sleep(2)
	
	if WriteReg( REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 0)
	time.sleep(4)

def CINPowerDown():
	print " "
	print "Powering Off CIN Board ........  "
#	if WriteReg( REG_PS_ENABLE, "000F", 1) != 1:
#		print 'Write register could not be verified. Aborting.'
#		sys.exit(1)
#	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
#	
#	time.sleep(2)
#	
	if WriteReg( REG_PS_ENABLE, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 0)
	time.sleep(4)

def CIN_FP_PowerUp():
	print " "
	print "Powering On Front Panel Boards ........  "
	if WriteReg( REG_PS_ENABLE, "003F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
	
	time.sleep(4)

def CIN_FP_PowerDown():
	print " "
	print "Powering Off Front Panel Boards ........  "
	if WriteReg( REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
	
	time.sleep(2)


