#! /usr/bin/python
# -*- coding: utf-8 -*-
import time
import socket
#import argparse 

# ============================================================================
#		Configuration Constants
# ============================================================================
CIN_FRAME_IP		= "10.0.5.207"
CIN_CONFIG_IP		= "192.168.1.207"
CIN_COMMAND_PORT 	= 49200
CIN_STREAM_IN_PORT 	= 49202
CIN_STREAM_OUT_PORT 	= 49203

# ============================================================================
#		System Constants
# ============================================================================
# Define Lakeshore Temp Controller IP & Port
LS325_TC_IP		= "192.168.1.46"
LS325_PORT		= 9001

# Define Keithley 2701 DMM IP & Port
DMM2701_IP   		= "192.168.1.47"
DMM2701_PORT 		= 1394

