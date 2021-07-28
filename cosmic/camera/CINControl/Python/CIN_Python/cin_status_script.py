#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal
import time

cin_hal.getCfgFpgaStat()

frm_done = 0
frm_done = cin_hal.getFrmDone(frm_done)
if (int(frm_done) == 1):
	cin_hal.getFrmFpgaStat()
	cin_hal.getFCLK()
	time.sleep(3)

cin_hal.getPowerStatusAll()
print " "

cin_hal.getCfgEthStat()
if (int(frm_done) == 1) : cin_hal.getFrmEthStatus()
print " "
raw_input("(Press Enter/Return to Exit)")

