#! /usr/bin/python
# -*- coding: utf-8 -*-
import socket
import sys
import time

# Create a TCP/IP socket
try:
  remote_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  socket.setdefaulttimeout(0.1)
except socket.error, msg:
  remote_sock = None
  print 'could not open socket'
  sys.exit(1)

print 'TCP/IP socket open'

# Connect the socket to the port where the CIN is listening
#  This script uses localhost, so it must be run on the same machine as the QT main window.
#  If you want to run it on a different machine, you need the host IP where the QT is running
server_address = ('localhost', 8880)
print >>sys.stderr, 'Connecting to %s port %s' % server_address

try:
  remote_sock.connect(server_address)
except socket.error, msg:
  remote_sock.close()
  remote_sock = None
  print 'could not connect socket'
  sys.exit(1)

#  Connect the gui to the camera interface node
remote_sock.sendall('connect')
print 'Connected to CIN'

#Define a non-standard loop 
def voltage_range(start, end, step):
    while start >= end:
	yield start
	start -= step

DELTAVoltage = -5
step = 0.1

for BASEVoltage in voltage_range(-9.3, -11.3, step):
	print 'Buff_VDD = {0} and Buff_VSS = {1}'.format(BASEVoltage, BASEVoltage+DELTAVoltage-0.1)
	
	# Make sure the clocks and bias voltages are turned off before making changes
	remote_sock.sendall('biasOFF')
	remote_sock.sendall('ClockOFF')

	# Change the Buf_VDD (BASE) value
	remote_sock.sendall('changeBUFVDDGUI({0})'.format(BASEVoltage))
	time.sleep(1)
	# If everything worked, the CIN should echo back the command 
	# In order to receive data, the socket must know the maximum size to receive at once. 4096 is 
	# just a reasonable default to use for a string. 
	print remote_sock.recv(4096) + " <- this is what I read back! Check me for garbage/errors."

	# Change the DELTA value, Buff_VSS = BaseV + DeltaV -0.1V 
	remote_sock.send('changeBUFVSSGUI({0})'.format(DELTAVoltage))
	time.sleep(1)
	print remote_sock.recv(4096) + " <- this is what I read back! Check me for garbage/errors."

	# Send the new voltages to the CIN
	remote_sock.sendall('sendBiastoCIN')

	# Turn the clocks and bias voltages back on
	remote_sock.sendall('biasON')
	remote_sock.sendall('ClockON')

	# Send a software trigger 
	remote_sock.sendall('swtrig')
	
	# Capture image
	remote_sock.sendall('captureSingle')
	
	print ' Image Captured '
	# When we move VDD up, we need to increase the magnitude of delta by twice the change in VDD so that VDD and VSS move the same amount each time
	DELTAVoltage = DELTAVoltage-2*step

#Disconnect from the CIN
print 'Disconnecting from CIN'
remote_sock.sendall('disconnect')

#Close the TCP/IP port 
print >>sys.stderr, 'Closing socket'
#remote_sock.shutdown(takes an argument)
remote_sock.close()