#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal
import time
from epics import caget, caput, cainfo

cin_hal.CINPowerDown()
cin_hal.CINPowerUp()

cin_hal.getCfgFpgaStat()

cin_hal.CINLoadFirmware("/home/user/CVSSandbox/BINARY/CIN_1kFSCCD/top_frame_fpga_r1004.bit")
#cin_hal.CINLoadFirmware("/home/user/CVSSandbox/BINARY/CIN_1kFSCCD/top_frame_fpga_flip_CCD_CTL_IN.bit")

cin_hal.getFrmFpgaStat()
#flashFpCfgLeds()
#flashFpFrmLeds()

cin_hal.getPowerStatusAll()

#cin_hal.setFCLK250M()
cin_hal.setFCLK125M()

cin_hal.getFCLK()

cin_hal.CIN_FP_PowerUp()
#CIN_FP_PowerDown()
time.sleep(2)  # Wait to allow visual check

cin_hal.getPowerStatusAll()

# import cin_sram_script

print "Powering up FCCD2 1KFSCCD Detector"
print "Powering on Acopian power supply ..."
powerStatus = caget('fccd2:WebRelay1:Y0OutB.VAL')
print "Power Supply Status (0=off, 1=on) = " + str(powerStatus)
caput ( 'fccd2:WebRelay1:Y0OutB.VAL' , 1 )
time.sleep(2.0)

print " "
raw_input("(Press Enter Key to Exit)")

