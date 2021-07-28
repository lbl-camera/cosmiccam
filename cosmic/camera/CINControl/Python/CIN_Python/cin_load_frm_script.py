#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal
import time

cin_hal.CINLoadFirmware("/root/Desktop/CINControl/top_frame_fpga.bit")
cin_hal.getFrmFpgaStat()


