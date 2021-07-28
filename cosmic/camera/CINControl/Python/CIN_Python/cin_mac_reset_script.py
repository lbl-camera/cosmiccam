#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_hal

cin_hal.WriteFrmReg( cin_hal.REG_MAC_CONFIG_VEC_FAB1B0, "0D9B", cin_hal.CIN_CONFIG_IP, cin_hal.CIN_COMMAND_PORT, 1)
cin_hal.WriteFrmReg( cin_hal.REG_MAC_CONFIG_VEC_FAB1B0, "058B", cin_hal.CIN_CONFIG_IP, cin_hal.CIN_COMMAND_PORT, 1)