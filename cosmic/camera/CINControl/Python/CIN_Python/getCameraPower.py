#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import socket
import string
import time

# ============================================================================
#               System Constants
# ============================================================================
# Setup the Keithley 2701 DMM
DMM2701_IP   = "192.168.1.47"
DMM2701_PORT = 1394

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
s.connect((DMM2701_IP,DMM2701_PORT))

# Configure DMM
#s.sendall("RST;FORM:ELEM READ\r\n")
# Setup Channel List (@101:106) Voltage Chans, (@109:114) Current Chans
# Scan & Readout Channels one at a time

print "\nFastCCD Power Monitor"

s.sendall('MEAS:VOLT? (@101)\r\n')
v = s.recv(1024)
tv = float(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[11]+v[12]+v[13]+v[14])
s.sendall('MEAS:VOLT? (@109)\r\n')
a = s.recv(1024)
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])
#print ta
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])*50
if (tv < 0.05)  : tv = 0
if (ta < 0.001) : ta = 0
print "3.5V : "+str(tv)+"V @ "+str(ta)[:6]+"A"

s.sendall('MEAS:VOLT? (@103)\r\n')
v = s.recv(1024)
tv = float(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[11]+v[12]+v[13]+v[14])
s.sendall('MEAS:VOLT? (@111)\r\n')
a = s.recv(1024)
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])*50
if (tv < 0.05)  : tv = 0
if (ta < 0.001) : ta = 0
print "A15V : "+str(tv)+"V @ "+str(ta)[:6]+"A"

s.sendall('MEAS:VOLT? (@104)\r\n')
v = s.recv(1024)
tv = float(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[11]+v[12]+v[13]+v[14])
s.sendall('MEAS:VOLT? (@112)\r\n')
a = s.recv(1024)
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])*50
if (tv < 0.05)  : tv = 0
if (ta < 0.001) : ta = 0
print "B15V : "+str(tv)+"V @ "+str(ta)[:6]+"A"

s.sendall('MEAS:VOLT? (@105)\r\n')
v = s.recv(1024)
tv = float(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[11]+v[12]+v[13]+v[14])
s.sendall('MEAS:VOLT? (@113)\r\n')
a = s.recv(1024)
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])*50
if (tv < 0.05)  : tv = 0
if (ta < 0.001) : ta = 0
print "A30V : "+str(tv)+"V @ "+str(ta)[:6]+"A"

s.sendall('MEAS:VOLT? (@106)\r\n')
v = s.recv(1024)
tv = float(v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[11]+v[12]+v[13]+v[14])
s.sendall('MEAS:VOLT? (@114)\r\n')
a = s.recv(1024)
ta = float(a[1]+a[2]+a[3]+a[4]+a[10]+a[11]+a[12]+a[13]+a[14])*50
if (tv < 0.05)  : tv = 0
if (ta < 0.001) : ta = 0
print "B30V : "+str(tv)+"V @ "+str(ta)[:6]+"A"

print 

s.close()

