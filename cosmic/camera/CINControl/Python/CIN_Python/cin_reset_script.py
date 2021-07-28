#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal
import time
cin_hal.WriteReg( cin_hal.REG_FRM_RESET, "0001", cin_hal.CIN_CONFIG_IP, cin_hal.CIN_COMMAND_PORT, 0)
cin_hal.WriteReg( cin_hal.REG_FRM_RESET, "0000", cin_hal.CIN_CONFIG_IP, cin_hal.CIN_COMMAND_PORT, 0)
cin_hal.CINSetFCLK()

