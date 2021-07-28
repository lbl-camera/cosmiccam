#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal
import time
from epics import caget, caput, cainfo

cin_hal.setCameraOff()

cin_hal.CINPowerDown()

time.sleep(2.0)

print "Powering down FCCD2 1KFSCCD Detector"
print "Powering off Acopian power supply ..."
powerStatus = caget('fccd2:WebRelay1:Y0OutB.VAL')
print "Power Supply Status (0=off, 1=on) = " + str(powerStatus)
caput ( 'fccd2:WebRelay1:Y0OutB.VAL' , 0 )

print " "
raw_input("(Press Any Key to Exit)")
