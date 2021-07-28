#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import string
import time
import socket

# ============================================================================
#               System Constants
# ============================================================================
E36102A_IP   = "192.168.1.3"
E36102A_PORT = 5025

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
s.connect((E36102A_IP,E36102A_PORT))

# Clear Communications
s.sendall('*CLS\r\n')
# Wait at least 50ms before sending next command
time.sleep(0.1)

#print "FastCCD FO Power Supply Monitor"
s.sendall('MEAS:VOLT?')
v = s.recv(16)
s.sendall('MEAS:CURR?')
i = s.recv(16)
#print v
voltage=float(v)
if (voltage < 0.01) : voltage = 0.000
print str(voltage)[0:5]+" V"
#print i
current=float(i)
if (current < 0.01) : current = 0.000
print str(current)[0:5]+" A"

s.close()

