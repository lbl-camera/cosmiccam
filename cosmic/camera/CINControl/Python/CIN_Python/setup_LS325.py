#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import string
import time
import socket

# Script to Setup LakeShore 325 for CCD DAQ
# User must set up the BrainBoxes ES-246 IP & COM port and then set 
# the baud rate of the LS325 to match.  We currently use 9600
# COM Port Settings: 9600, 7bits, 1 stop bit, ODD parity, Xon/Xoff
#
# SETUP ASSUMES INPUT B (RTD) and C-LOOP 1
# Since this should be a one-time script, there is no readback
# functions.  To verify setup, use the front panel keyboard
# on the Temperature Controller

# ============================================================================
#               System Constants
# ============================================================================
# Determined/set in the ES-246 Ethernet2Serial Interface
LS325_IP   = "192.168.1.46"
LS325_PORT = 9001

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
s.connect((LS325_IP,LS325_PORT))

# Clear Communications
s.sendall('*CLS\r\n')
# Wait at least 50ms before sending next command
time.sleep(0.1)
# Set Controller to Remote Mode
s.sendall('MODE 1\r\n')
# Wait at least 50ms before sending next command
time.sleep(0.1)
# First Read is a throw-away, so just readback the mode
s.sendall('MODE?\r\n')
# Wait at least 50ms before sending next command
time.sleep(0.1)
m = s.recv(16)
#print m

# Display Setup
# Display 1 = Input B, Temp C
s.sendall('DISPFLD 1,2,2\r\n')
time.sleep(0.1)
# Display 2 = Off
s.sendall('DISPFLD 2,0\r\n')
time.sleep(0.1)
# Display 3 = Set Point
s.sendall('DISPFLD 3,3\r\n')
time.sleep(0.1)
# Display 4 = Heater Output
s.sendall('DISPFLD 4,4\r\n')
time.sleep(0.1)

# Input Setup
# Set to Input B, RTD100
s.sendall('INTYPE B,2,0\r\n')
time.sleep(0.1)
# Set Curve RTD=06
s.sendall('INCRV B,6\r\n')
time.sleep(0.1)
# Set Filter
s.sendall('FILTER B,1,8,5\r\n')
time.sleep(0.1)

# Control Loop Setup
# Set to C-Loop 1, In B, C, Power Up Off, Out=Power
s.sendall('CSET 1,B,2,0,2\r\n')
time.sleep(0.1)
# Set to C-Mode  ** Not Sure Yet
#s.sendall('CMODE 1,4\r\n')
#time.sleep(0.1)
# Set Heater Resistor to 25 ohms
s.sendall('HTRRES 1,1\r\n')
time.sleep(0.1)
# Set Ramp Off
s.sendall('RAMP 1,0,0.0\r\n')
time.sleep(0.1)
# Set Temperature SetPoint
s.sendall('SETP 1,-55\r\n')
time.sleep(0.1)
# Set Heater Range to High
s.sendall('RANGE 1,2\r\n')
time.sleep(0.1)

s.close()

