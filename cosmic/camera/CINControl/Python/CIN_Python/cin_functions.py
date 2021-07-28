#! /usr/bin/python
# -*- coding: utf-8 -*-
import time
import socket
import cin_constants
import cin_register_map
#import argparse 

# ============================================================================
#					Socket
# ============================================================================
try:
    cin_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.setdefaulttimeout(0.1)

except socket.error, msg:
    cin_sock = None
    print 'could not open socket'
    sys.exit(1)

# ============================================================================
#					Functions
# ============================================================================
# ------------------------------------------------------< common functions >
def ByteToHex( byteStr ):
    """
    Convert a byte string to it's hex string representation e.g. for output.
    """
    
    # Uses list comprehension which is a fractionally faster implementation than
    # the alternative, more readable, implementation below
    #   
    #    hex = []
    #    for aChar in byteStr:
    #        hex.append( "%02X " % ord( aChar ) )
    #
    #    return ''.join( hex ).strip()        

    return ''.join( [ "%02X" % ord( x ) for x in byteStr ] ).strip()

def HexToByte( hexStr ):
    """
    Convert a string hex byte values into a byte string. The Hex Byte values may
    or may not be space separated.
    """
    # The list comprehension implementation is fractionally slower in this case
    #
    #    hexStr = ''.join( hexStr.split(" ") )
    #    return ''.join( ["%c" % chr( int ( hexStr[i:i+2],16 ) ) \
    #                                   for i in range(0, len( hexStr ), 2) ] )

    bytes = []

    hexStr = ''.join( hexStr.split(" ") )

    for i in range(0, len(hexStr), 2):
        bytes.append( chr( int (hexStr[i:i+2], 16 ) ) )

    return ''.join( bytes )
	
def WriteReg( regAddr, value, verify ):
	cin_sock.sendto(HexToByte(regAddr + value ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))
	time.sleep(0.001)
	return 1 # bypass for now
#	if verify == 1:
#		cin_sock.sendto(HexToByte(cin_register_map.REG_READ_ADDRESS + regAddr ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))
#		cin_sock.sendto(HexToByte(cin_register_map.REG_COMMAND + CMD_READ ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))
#		cin_sock.settimeout(1.0)
#		try:
#			data, addr = cin_sock.recvfrom(1024)
#		except cin_sock.timeout, msg:
#			return 0
#			
#		if ByteToHex(data) == value and ByteToHex(addr) == regAddr:
#			return 1
#		else:
#			return 0
#	else:
#		return 1
	
		
def ReadReg( regAddr ):
        cin_sock.sendto(HexToByte(cin_register_map.REG_READ_ADDRESS + regAddr ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))
        time.sleep(0.1)
        cin_sock.sendto(HexToByte(cin_register_map.REG_COMMAND + cin_register_map.CMD_READ_REG ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))

        #time.sleep(0.1)
        cin_sock.settimeout(1.0)
        try:
            data, addr = cin_sock.recvfrom(1024)

        except socket.timeout:
            time.sleep(0.1)
            cin_sock.sendto(HexToByte(cin_register_map.REG_COMMAND + cin_register_map.CMD_READ_REG ),(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_COMMAND_PORT))
            cin_sock.settimeout(1.0)
            data, addr = cin_sock.recvfrom(1024)

        return ByteToHex(data)

# ---------------------------------------------< Connect to QT DAQ
def connect2daq( ):
	cin_sock.sendto(( "dummy data" ),(cin_constants.CIN_FRAME_IP, cin_constants.CIN_STREAM_OUT_PORT))

# ---------------------------------------------< Configuration FPGA functions >

def getFrmDone(frm_done):
        temp = bin((int(ReadReg( cin_register_map.REG_FPGA_STATUS)[4:8],16)))[2:].zfill(16)
	frm_done = int(temp[-16])
	return frm_done

def current_calc(reg_val, current):
	if (int(reg_val[4:8],16) >= int("8000",16)):
#	  current = 0.000000238*((int("10000",16) - int(reg_val[4:8],16)))/0.003
	  current = 0.000000476*((int("10000",16) - int(reg_val[4:8],16)))/0.003
	else:
#	  current = 0.000000238*(int(reg_val[4:8],16))/0.003
	  current = 0.000000476*(int(reg_val[4:8],16))/0.003
	return current

def CINPowerUp():
	print " "
	print "Powering On CIN Board ........  "
	if WriteReg( cin_register_map.REG_PS_ENABLE, "000F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 1)
	
	if WriteReg( cin_register_map.REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 0)
	time.sleep(2)

def CINPowerDown():
	print " "
	print "Powering Off CIN Board ........  "
#	if WriteReg( REG_PS_ENABLE, "000F", 1) != 1:
#		print 'Write register could not be verified. Aborting.'
#		sys.exit(1)
#	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
#	
#	time.sleep(2)
#	
	if WriteReg( cin_register_map.REG_PS_ENABLE, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 0)
	time.sleep(1)

def CIN_FP_PowerUp():
	print " "
	print "Powering On Front Panel Boards ........  "
	if WriteReg( cin_register_map.REG_PS_ENABLE, "003F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 1)
	
	time.sleep(4)

def CIN_FP_PowerDown():
	print " "
	print "Powering Off Front Panel Boards ........  "
	if WriteReg( cin_register_map.REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PS_ENABLE, 1)
	
	time.sleep(1)

def setCameraOff():
	print " "
	print "Turning off Bias and Clocks in camera head ........  "
	if WriteReg( cin_register_map.REG_BIASCONFIGREGISTER0_REG, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)

	if WriteReg( cin_register_map.REG_CLOCKCONFIGREGISTER0_REG, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	
	time.sleep(1)
	
def clearFocusBit():
	# Get Value from Clock&Bias Control
	reg_val = ReadReg("8205")
	#print reg_val[4:]

	temp = str(hex((int(reg_val[7:],base=16)&0xD)).lstrip("0x"))
	#print temp

	str_val=reg_val[4:5]+reg_val[5:6]+reg_val[6:7]+temp
	WriteReg("8205", str_val, 1)

def setFocusBit():
        # Get Value from Clock&Bias Control
        reg_val = ReadReg("8205")
        #print reg_val[4:]
 
	#temp = str(hex((int(reg_val[7:],base=16)|0x2)).lstrip("0x"))
	#print temp
 
        str_val=reg_val[4:5]+reg_val[5:6]+reg_val[6:7]+"A"  #temp
        WriteReg("8205", str_val, 1)

def loadFrmFirmware(filename):
	print " "
	print "Loading Frame (FRM) FPGA Configuration Data ...........  "
	print "File: " + filename
	WriteReg( cin_register_map.REG_COMMAND, cin_register_map.CMD_PROGRAM_FRAME, 0)
	time.sleep(1)
	with open(filename, 'rb') as f:
		read_data = f.read(128)
		while read_data != "":
		  cin_sock.sendto(read_data,(cin_constants.CIN_CONFIG_IP, cin_constants.CIN_STREAM_IN_PORT))
		  time.sleep(0.000125)	# For UDP flow control (was 0.002)
		  read_data = f.read(128)
	f.closed
	time.sleep(1)
	WriteReg( cin_register_map.REG_FRM_RESET, "0001", 0)
	WriteReg( cin_register_map.REG_FRM_RESET, "0000", 0)
	time.sleep(1)
	# need to verify sucess!

def loadCameraConfigFile(filename):
	print " "
	print "Loading Configuration File to CCD Camera ...........  "
	print "File: " + filename
	with open(filename, 'r') as f:
		file_line = f.readline()
		while file_line != "":
			if (file_line[:1] != "#") :
				read_addr = file_line[:4]
				read_data = file_line[5:9]
				#print read_addr + read_data
				WriteReg( read_addr, read_data, 0 )
#				time.sleep(0.1)
			file_line = f.readline()
	f.closed

def flashFpCfgLeds():
# Test Front Panel LEDs
	print " "
	print "Flashing CFG FP LEDs  ............ "
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "AAAA", 1)
	time.sleep(1)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "5555", 1)
	time.sleep(1)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "FFFF", 1)
	time.sleep(1)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0001", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0002", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0004", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0008", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0010", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0020", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0040", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0080", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0100", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0200", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0400", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0800", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "1000", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "2000", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "4000", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "8000", 1)
	time.sleep(0.4)
	WriteReg( cin_register_map.REG_SANDBOX_REG00, "0000", 1)


# ---------------------------------------------< Frame FPGA functions >

def flashFpFrmLeds():
# Test Front Panel LEDs
	print " "
	print "Flashing FRM FP LEDs  ............ "
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0004", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0008", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "000C", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0010", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0020", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0030", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0040", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0080", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "00C0", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0100", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0200", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0300", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0400", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0800", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0C00", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( cin_register_map.REG_FRM_SANDBOX_REG00, "0000", 1)
	print "All OFF  ............ "
	time.sleep(0.5)
	
