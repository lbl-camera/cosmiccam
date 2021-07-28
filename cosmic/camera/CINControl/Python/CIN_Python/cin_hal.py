#! /usr/bin/python
# -*- coding: utf-8 -*-
import time
import socket
#import argparse 

# ============================================================================
#						Configuration Constants
# ============================================================================
CIN_FRAME_IP		= "10.0.5.207"
CIN_CONFIG_IP		= "192.168.1.207"
CIN_COMMAND_PORT 	= 49200
CIN_STREAM_IN_PORT 	= 49202
CIN_STREAM_OUT_PORT 	= 49203

# ============================================================================
#							CIN Registers
# ============================================================================
# ------------------------------------------------------< Common Registers: >
# Command Registers
REG_COMMAND 				= "0001"
REG_READ_ADDRESS 			= "0002"
REG_STREAM_TYPE 			= "0003"

# Ethernet Interface
REG_IF_MAC0         			= "0010"
REG_IF_MAC1         			= "0011"
REG_IF_MAC2         			= "0012"
REG_IF_IP0          			= "0013"
REG_IF_IP1	         		= "0014"
REG_IF_CMD_PORT_NUM    			= "001A"
REG_IF_STREAM_IN_PORT_NUM		= "001C"
REG_IF_STREAM_OUT_PORT_NUM		= "001D"

REG_ETH_RESET 				= "0020" # Reset Eth Hardware 1=Rx, 2=Tx, 3=Both
REG_ETH_ENABLE 				= "0021" # Enable Eth Hardware 1=Rx, 2=Tx, 3=Both
REG_PHY1_MDIO_CMD			= "0022" # Start(1), RnW(1), WDRd(1), PHY Addr(5), REG Addr(5) 
REG_PHY1_MDIO_CMD_DATA			= "0023"
REG_PHY1_MDIO_STATUS			= "0024"
REG_PHY1_MDIO_RD_ADDR			= "0025"
REG_PHY1_MDIO_RD_DATA			= "0026"
REG_MAC_CFG_VECTOR1 			= "0027" # Ethernet Hardware Conf
REG_PHY2_MDIO_CMD			= "0028"
REG_PHY2_MDIO_CMD_DATA			= "0029"
REG_PHY2_MDIO_STATUS			= "002A"
REG_PHY2_MDIO_RD_ADDR			= "002B"
REG_PHY2_MDIO_RD_DATA			= "002C"
REG_MAC_CFG_VECTOR2 			= "002D" # Ethernet Hardware Conf

# -----------------------------------------------------< Configuration FPGA Registers >
# Power Supply Control
CMD_PS_ENABLE 				= "0021" # Enable Selected Power Modules
CMD_PS_POWERDOWN 			= "0022" # Start power down sequence

REG_PS_ENABLE    			= "0030" # Power Supply Enable: 
# b(0): 12V power bus enable
# b(1): 2.5v general enable
# b(2): 3.3v general enable
# b(3): 0.9v Frame FPGA enable
# b(4): 2.5v Frame FPGA enable
# b(5): 5.0v front panel enable

REG_PS_SYNC_DIV0 			= "0031" # 2.5V Gen
REG_PS_SYNC_DIV1 			= "0032" # 3.3V Gen
REG_PS_SYNC_DIV2 			= "0033" # 2.5V Frame
REG_PS_SYNC_DIV3 			= "0034" # 0.9V Frame
REG_PS_SYNC_DIV4 			= "0035" # 5.0V FP

# Frame FPGA Control
CMD_PROGRAM_FRAME			= "0041"
REG_FRM_RESET 				= "0036"; # Frame Reset
REG_FRM_10GbE_SEL			= "0037"; # 10GbE Link Select

# Clock Enables
CMD_ENABLE_CLKS				= "0031" # Enable selected Frame FPGA clock crystals 
CMD_DISABLE_CLKS			= "0032" # Disable Frame FPGA clock crystals 
REG_CLOCK_EN_REG 			= "0038" # Clock Enable Register

# Programmable Si570 Clock Registers
REG_SI570_REG0 				= "0039"
REG_SI570_REG1 				= "003A"
REG_SI570_REG2 				= "003B"
REG_SI570_REG3 				= "003C"

# Power Monitor Registers
CMD_MON_STOP 				= "0011" # Stop voltage and current monitor
CMD_MON_START 				= "0012" # Start voltage and current monitor

REG_VMON_ADC1_CH1   			= "0040" #	V12P_BUS Voltage Monitor
REG_IMON_ADC1_CH0   			= "0041" #	V12P_BUS Current Monitor
REG_VMON_ADC0_CH5  			= "0042" #	V3P3_MGMT Voltage Monitor
REG_IMON_ADC0_CH5  			= "0043" #	V3P3_MGMT Current Monitor
REG_VMON_ADC0_CH4  			= "0044" #	V3P3_S3E Voltage Monitor
REG_IMON_ADC0_CH4  			= "0045" #	V3P3_S3E Current Monitor
REG_VMON_ADC0_CH7  			= "0046" #	V2P5_MGMT Voltage Monitor
REG_IMON_ADC0_CH7  			= "0047" #	V2P5_MGMT Current Monitor
REG_VMON_ADC0_CH6  			= "0048" #	V1P8_MGMT Voltage Monitor
REG_IMON_ADC0_CH6  			= "0049" #	V1P8_MGMT Current Monitor
REG_VMON_ADC0_CH2  			= "004A" #	V1P2_MGMT Voltage Monitor
REG_IMON_ADC0_CH2  			= "004B" #	V1P2_MGMT Current Monitor
REG_VMON_ADC0_CH3  			= "004C" #	V1P0_ENET Voltage Monitor
REG_IMON_ADC0_CH3  			= "004D" #	V1P0_ENET Current Monitor
REG_VMON_ADC0_CH8  			= "004E" #	V3P3_GEN Voltage Monitor
REG_IMON_ADC0_CH8  			= "004F" #	V3P3_GEN Current Monitor
REG_VMON_ADC0_CH9  			= "0050" #	V2P5_GEN Voltage Monitor
REG_IMON_ADC0_CH9  			= "0051" #	V2P5_GEN Current Monitor
REG_VMON_ADC0_CHE  			= "0052" #	V0P9_V6 Voltage Monitor
REG_IMON_ADC0_CHE  			= "0053" #	V0P9_V6 Current Monitor
REG_VMON_ADC0_CHD  			= "0054" #	V2P5_V6 Voltage Monitor
REG_IMON_ADC0_CHD  			= "0055" #	V2P5_V6 Current Monitor
REG_VMON_ADC0_CHB  			= "0056" #	V1P0_V6 Voltage Monitor
REG_IMON_ADC0_CHB  			= "0057" #	V1P0_V6 Current Monitor
REG_VMON_ADC0_CHC  			= "0058" #	V1P2_V6 Voltage Monitor
REG_IMON_ADC0_CHC  			= "0059" #	V1P2_V6 Current Monitor
REG_VMON_ADC0_CHF  			= "005A" #	V5P0_FP Voltage Monitor (1/2)
REG_IMON_ADC0_CHF  			= "005B" #	V5P0_FP Current Monitor (1/2)

# Status Registers
REG_DCM_STATUS				= "0080"
REG_FPGA_STATUS    			= "0081"
REG_BOARD_ID       			= "008D"
REG_HW_SERIAL_NUM  			= "008E"
REG_FPGA_VERSION   			= "008F"

# Sandbox Registers
REG_SANDBOX_REG00 			= "00F0"
REG_SANDBOX_REG01 			= "00F1"
REG_SANDBOX_REG02 			= "00F2"
REG_SANDBOX_REG03 			= "00F3"
REG_SANDBOX_REG04 			= "00F4"
REG_SANDBOX_REG05 			= "00F5"
REG_SANDBOX_REG06 			= "00F6"
REG_SANDBOX_REG07 			= "00F7"
REG_SANDBOX_REG08 			= "00F8"
REG_SANDBOX_REG09 			= "00F9"
REG_SANDBOX_REG0A 			= "00FA"
REG_SANDBOX_REG0B 			= "00FB"
REG_SANDBOX_REG0C 			= "00FC"
REG_SANDBOX_REG0D 			= "00FD"
REG_SANDBOX_REG0E 			= "00FE"
REG_SANDBOX_REG0F 			= "00FF"

# ------------------------------------------------------< Frame FPGA Registers >

# Command Registers
REG_FRM_COMMAND 			= "8001"
REG_FRM_READ_ADDRESS 			= "8002"
REG_FRM_STREAM_TYPE 			= "8003"

# Ethernet Interface
REG_IF_MAC_FAB1B0   			= "8010"
REG_IF_MAC_FAB1B1   			= "8011"
REG_IF_MAC_FAB1B2   			= "8012" 
REG_IF_IP_FAB1B0    			= "8013"
REG_IF_IP_FAB1B1    			= "8014" 
REG_IF_CMD_PORT_NUM_FAB1B   	      	= "8015" 
REG_IF_STREAM_IN_PORT_NUM_FAB1B        	= "8016" 
REG_IF_STREAM_OUT_PORT_NUM_FAB1B       	= "8017" 

REG_XAUI_FAB1B 				= "8018"

REG_MAC_CONFIG_VEC_FAB1B0 		= "8019"
REG_MAC_CONFIG_VEC_FAB1B1 		= "801A"
REG_MAC_STATS1_FAB1B0          		= "801B"
REG_MAC_STATS1_FAB1B1          		= "801C"
REG_MAC_STATS2_FAB1B0			= "801D"
REG_MAC_STATS2_FAB1B1          		= "801E"

REG_IF_MAC_FAB2B0   			= "8020"
REG_IF_MAC_FAB2B1   		    	= "8021"
REG_IF_MAC_FAB2B2   		    	= "8022" 
REG_IF_IP_FAB2B0    		    	= "8023" 
REG_IF_IP_FAB2B1    		    	= "8024" 
REG_IF_CMD_PORT_NUM_FAB2B   	    	= "8025"
REG_IF_STREAM_IN_PORT_NUM_FAB2B    	= "8026"
REG_IF_STREAM_OUT_PORT_NUM_FAB2B   	= "8027"

REG_XAUI_FAB2B 		    	    	= "8028" 

REG_MAC_CONFIG_VEC_FAB2B0 	    	= "8029" 
REG_MAC_CONFIG_VEC_FAB2B1 	    	= "802A" 
REG_MAC_STATS1_FAB2B0              	= "802B"
REG_MAC_STATS1_FAB2B1              	= "802C"
REG_MAC_STATS2_FAB2B0              	= "802D"
REG_MAC_STATS2_FAB2B1              	= "802E"

# SRAM Test Interface
REG_SRAM_COMMAND 			= "8030"
#  1 bit  [0]    >> Read NOT Write
#  2 bits [3:2] >> Modes:
#      --  Single RW "00"
#      --  Burst RW  "01"
#      --  Test/Diagnostic "10"
#      --  Sleep  "11"
#   1 bit [4]     >> start/stop
	 
REG_SRAM_START_ADDR1     		= "8031"
REG_SRAM_START_ADDR0     		= "8032"
REG_SRAM_STOP_ADDR1      		= "8033"
REG_SRAM_STOP_ADDR0      		= "8034"
REG_SRAM_FRAME_DATA_OUT1 		= "8035"
REG_SRAM_FRAME_DATA_OUT0 		= "8036"                           
REG_SRAM_FRAME_DATA_IN1			= "8037"
REG_SRAM_FRAME_DATA_IN0  		= "8038"
REG_SRAM_FRAME_DV        		= "8039"
REG_SRAM_STATUS1         		= "803A"
REG_SRAM_STATUS0         		= "803B"

# Programmable Clock                       
CMD_FCLK_COMMIT 			= "0012" # Start I2C Write/Read
REG_FCLK_I2C_ADDRESS 			= "8040" # [ Slave Address(7), RD/WRn(1), Reg Adress(8) ] Slave adddress Hx"58"  -> Hx"B" when shifted up by 1
REG_FCLK_I2C_DATA_WR 			= "8041" # [ Clock Select(2), Clock Enable (1), 0(5), Write Data (8) ]
#   Clock Select: (00): 250 MHz (01): 200 MHz (10): FPGA FCRIC Clk (11): Si570 Programmable
REG_FCLK_I2C_DATA_RD 			= "8042" # [ Read Failed (1), Write Failed(1), Toggle bit(1), 0(5), Read Data (8) ]
	
REG_TRIGGERSELECT_REG			= "8050"
REG_CCDFCLKSELECT_REG			= "8051"

# FRM Status
REG_FRM_DCM_STATUS     			= "8080"
REG_FRM_FPGA_STATUS    			= "8081"
REG_FRM_BOARD_ID       			= "808D"
REG_FRM_HW_SERIAL_NUM  			= "808E"
REG_FRM_FPGA_VERSION   			= "808F"
	
# Sandbox Registers
REG_FRM_SANDBOX_REG00  			= "80F0"
REG_FRM_SANDBOX_REG01  			= "80F1"
REG_FRM_SANDBOX_REG02  			= "80F2"
REG_FRM_SANDBOX_REG03  			= "80F3"
REG_FRM_SANDBOX_REG04  			= "80F4"
REG_FRM_SANDBOX_REG05  			= "80F5"
REG_FRM_SANDBOX_REG06  			= "80F6"
REG_FRM_SANDBOX_REG07  			= "80F7"
REG_FRM_SANDBOX_REG08  			= "80F8"
REG_FRM_SANDBOX_REG09  			= "80F9"
REG_FRM_SANDBOX_REG0A  			= "80FA"
REG_FRM_SANDBOX_REG0B  			= "80FB"
REG_FRM_SANDBOX_REG0C  			= "80FC"
REG_FRM_SANDBOX_REG0D  			= "80FD"
REG_FRM_SANDBOX_REG0E  			= "80FE"
REG_FRM_SANDBOX_REG0F  			= "80FF"
	
# Image Processing Registers
CMD_IP_SYNC_PULSE 			= "8100" # ISSUES A SYNC PULSE
CMD_IP_SYNC_PULSE 			= "8101" # COMMAND TO SYNC DETECTOR AND READOUT (SEE IMAGE PROCESSING)
CMD_IP_SYNC_PULSE 			= "8102" # RESET IMAGE PROCESSING REGISTERS
CMD_IP_SYNC_PULSE 			= "8103" # WRITE CCD BIAS REGISTERS
CMD_IP_SYNC_PULSE 			= "8104" # WRITE CCD CLOCK REGISTER
CMD_IP_SYNC_PULSE 			= "8105" # SEND CONFIG DATA TO FRIC			
CMD_IP_SYNC_PULSE 			= "8106" # RESET STATISTICS/DEBUG COUNTERS

REG_DETECTOR_REVISION_REG   		= "8100"
REG_DETECTOR_CONFIG_REG1    		= "8101"
REG_DETECTOR_CONFIG_REG2    		= "8102"
REG_DETECTOR_CONFIG_REG3    		= "8103"
REG_DETECTOR_CONFIG_REG4    		= "8104"
REG_DETECTOR_CONFIG_REG5    		= "8105"
REG_DETECTOR_CONFIG_REG6    		= "8106"
REG_DETECTOR_CONFIG_REG7		= "8107"
REG_DETECTOR_CONFIG_REG8    		= "8108"
REG_IMG_PROC_REVISION_REG   		= "8120"
REG_IMG_PROC_CONFIG_REG1    		= "8121"
REG_IMG_PROC_CONFIG_REG2    		= "8122"
REG_IMG_PROC_CONFIG_REG3    		= "8123"
REG_IMG_PROC_CONFIG_REG4    		= "8124"
REG_IMG_PROC_CONFIG_REG5    		= "8125"
REG_IMG_PROC_CONFIG_REG6    		= "8126"
REG_IMG_PROC_CONFIG_REG7    		= "8127"
REG_IMG_PROC_CONFIG_REG8    		= "8128"
	
REG_BIASANDCLOCKREGISTERADDRESS_REG	= "8200"
REG_BIASANDCLOCKREGISTERDATA_REG	= "8201"
REG_CLOCKREGISTERDATAOUT_REG		= "8202"
REG_BIASREGISTERDATAOUT_REG		= "8203"

# Bias Static Registers             
REG_BIASCONFIGREGISTER0_REG     	= "8204"

# Clock Static Registers            
REG_CLOCKCONFIGREGISTER0_REG     	= "8205"
REG_EXPOSURETIMEMSB_REG		      	= "8206"
REG_EXPOSURETIMELSB_REG		      	= "8207"
REG_TRIGGERREPETITIONTIMEMSB_REG	= "8208"
REG_TRIGGERREPETITIONTIMELSB_REG	= "8209"
REG_DELAYTOEXPOSUREMSB_REG		= "820A"
REG_DELAYTOEXPOSURELSB_REG		= "820B"
REG_NUMBEROFEXPOSURE_REG		= "820C"
REG_SHUTTERTIMEMSB_REG			= "820D"
REG_SHUTTERTIMELSB_REG			= "820E"
REG_DELAYTOSHUTTERMSB_REG		= "820F"
REG_DELAYTOSHUTTERLSB_REG		= "8210"

# Digitizer Registers         
REG_FCRIC_MASK_REG1			= "8211"
REG_FCRIC_MASK_REG2			= "8212"
REG_FCRIC_MASK_REG3			= "8213"
REG_LVDS_OVERFLOW_ERROR_REG1		= "8214"
REG_LVDS_OVERFLOW_ERROR_REG2		= "8215"
REG_LVDS_OVERFLOW_ERROR_REG3		= "8216"
REG_LVDS_PARITY_ERROR_REG1		= "8217"
REG_LVDS_PARITY_ERROR_REG2		= "8218"
REG_LVDS_PARITY_ERROR_REG3		= "8219"
REG_LVDS_STOP_BIT_ERROR_REG1		= "821A"
REG_LVDS_STOP_BIT_ERROR_REG2		= "821B"
REG_LVDS_STOP_BIT_ERROR_REG3		= "821C"
REG_FCRIC_WRITE0_REG			= "821D"
REG_FCRIC_WRITE1_REG			= "821E"
REG_FCRIC_WRITE2_REG			= "821F"
REG_FCRIC_READ0_REG			= "8220"
REG_FCRIC_READ1_REG			= "8221"
REG_FCRIC_READ2_REG			= "8222"
REG_DEBUGVIDEO0_REG			= "8223"
REG_DEBUGVIDEO1_REG			= "8224"
REG_DEBUGVIDEO2_REG			= "8225"
REG_DEBUGVIDEO3_REG			= "8226"
REG_DEBUGVIDEO4_REG			= "8227"
REG_DEBUGVIDEO5_REG			= "8228"
REG_DEBUGVIDEO6_REG			= "8229"
REG_DEBUGVIDEO7_REG			= "822A"
REG_DEBUGVIDEO8_REG			= "822B"
REG_DEBUGVIDEO9_REG			= "822C"
REG_DEBUGVIDEO10_REG			= "822D"
REG_DEBUGVIDEO11_REG			= "822E"
REG_DEBUGCOUNTER00_REG			= "822F"
REG_DEBUGCOUNTER01_REG			= "8230"
REG_DEBUGCOUNTER02_REG			= "8231"
REG_DEBUGCOUNTER03_REG			= "8232"
REG_DEBUGCOUNTER04_REG			= "8233"

# ============================================================================
#							CIN Commands
# ============================================================================
# ------------------------------------------------------< Common Commands: > 
# Common Commands
CMD_READ_REG 		= "0001" # Read Register

# ------------------------------------------------------< Frame FPGA Commands: > 
# Programmable FCLK Commands


# Image Processor Commands						


# ----------------------------------------------< Configuration FPGA Commands: > 

# ============================================================================
#								Socket
# ============================================================================
try:
    cin_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.setdefaulttimeout(0.1)

except socket.error, msg:
    cin_sock = None
    print 'could not open socket'
    sys.exit(1)

# ============================================================================
#								Functions
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
	cin_sock.sendto(HexToByte(regAddr + value ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))
	time.sleep(0.001)
	return 1 # bypass for now
#	if verify == 1:
#		cin_sock.sendto(HexToByte(REG_READ_ADDRESS + regAddr ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))
#		cin_sock.sendto(HexToByte(REG_COMMAND + CMD_READ ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))
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
        cin_sock.sendto(HexToByte(REG_READ_ADDRESS + regAddr ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))
        time.sleep(0.1)
        cin_sock.sendto(HexToByte(REG_COMMAND + CMD_READ_REG ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))

        time.sleep(0.1)
        cin_sock.settimeout(1.0)
        try:
            data, addr = cin_sock.recvfrom(1024)

        except socket.timeout:
            time.sleep(0.1)
            cin_sock.sendto(HexToByte(REG_COMMAND + CMD_READ_REG ),(CIN_CONFIG_IP, CIN_COMMAND_PORT))
            cin_sock.settimeout(1.0)
            data, addr = cin_sock.recvfrom(1024)

        return ByteToHex(data)

# ---------------------------------------------< Configuration FPGA functions >
def getBoardSerNum():
# get CIN Board Serial Number
	print " "
	reg_val = ReadReg( REG_HW_SERIAL_NUM )
	print "  HW Serial Number : " + reg_val[4:]

def getFrmDone(frm_done):
        temp = bin((int(ReadReg( REG_FPGA_STATUS)[4:8],16)))[2:].zfill(16)
	frm_done = int(temp[-16])
	return frm_done

def getCfgFpgaStat():
# get Status Registers
	print " "
	print "****  CFG FPGA Status Registers   "
	print " "
	reg_val = ReadReg( REG_HW_SERIAL_NUM )
	print "  HW Serial Number : " + reg_val[4:]
	reg_val = ReadReg( REG_FPGA_VERSION )
	print "  CFG FPGA Version : " + reg_val[4:]
	reg_val = ReadReg( REG_BOARD_ID)
	print "  CIN Board ID     : " + reg_val[4:]
	print " "
	reg_val = ReadReg( REG_FPGA_STATUS )
	print "  FPGA Status      : " + reg_val[4:]
	print " "
# FPGA Status
# 15 == FRM DONE
# 14 == NOT FRM BUSY
# 13 == NOT FRM INIT B
# 12 >> 4 == 0
# 3 >>0 == FP Config Control 3 == PS Interlock
        reg_val = bin((int(ReadReg( REG_FPGA_STATUS)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
	if (int(stats_vec[-16]) == 1):
		print "  *** Frame FPGA Configuration Done"
	else:
		print "  *** Frame FPGA NOT Configured"
	if (int(stats_vec[-4]) == 1):
		print "  *** FP Power Supply Unlocked"
	else:
		print "  *** FP Power Supply Locked Off"
	print " "
	reg_val = ReadReg( REG_DCM_STATUS )
	print "  CFG DCM Status   : " + reg_val[4:]
# DCM Status
# 15 == 0
# 14 >> 8 == CONF SW
# 7 == ATCA 48V Alarm
# 6 == tx2 src ready
# 5 == tx1 src ready
# 4 == DCM STATUS2
# 3 == DCM STATUS1
# 2 == DCM STATUS0
# 1 == DCM PSDONE
# 0 == DCM LOCKED
        reg_val = bin((int(ReadReg( REG_DCM_STATUS)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
	if (int(stats_vec[-8]) == 1):
		print "  *** ATCA 48V Alarm"
	else:
		print "  *** ATCA 48V OK"
	if (int(stats_vec[-1]) == 1):
		print "  *** CFG Clock DCM Locked"
	else:
		print "  *** CFG Clock DCM NOT Locked"
	if (int(stats_vec[-12]) == 0):
		print "  *** FP Power Supply Interlock Overide Enable"

def getCfgEthStat():
        print "**** CFG FPGA Ethernet Status Registers "
        print " "
        # Get PHY1 Status register
        WriteReg( REG_PHY1_MDIO_CMD, "C001", 1)
        time.sleep(0.1)
        WriteReg( REG_PHY1_MDIO_CMD, "0000", 1)
        reg_val = bin((int(ReadReg( REG_PHY1_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
        print "  1GbE PHY1 Status Register : "
        print stats_vec[-9] + " : Extended Status"
        print stats_vec[-7] + " : MGMT Frame Preamble Suppression"
        print stats_vec[-6] + " : Copper auto-negotiation complete"
        print stats_vec[-5] + " : Copper remote fault detect"
        print stats_vec[-4] + " : Auto-negotiation Enabled"
        print stats_vec[-3] + " : Link Up"
        print stats_vec[-2] + " : Jabber Detected"
        print stats_vec[-1] + " : Extended Capability"
        print " "

        # Get PHY1 Extended Status register
        WriteReg( REG_PHY1_MDIO_CMD, "C00F", 1)
        time.sleep(0.1)
        WriteReg( REG_PHY1_MDIO_CMD, "0000", 1)
        reg_val = bin((int(ReadReg( REG_PHY1_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
        print "  1GbE PHY1 Extended Status Register : "
        print stats_vec[-14] + " : Full Duplex 1000 Base"
        print stats_vec[-13] + " : Half Duplex 1000 Base"
        print " "
        time.sleep(2)

        # Get PHY2 Status register
        WriteReg( REG_PHY2_MDIO_CMD, "C001", 1)
        time.sleep(0.1)
        WriteReg( REG_PHY2_MDIO_CMD, "0000", 1)
        reg_val = bin((int(ReadReg( REG_PHY2_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
        print "  1GbE PHY2 Status Register : "
        print stats_vec[-9] + " : Extended Status"
        print stats_vec[-7] + " : MGMT Frame Preamble Suppression"
        print stats_vec[-6] + " : Copper auto-negotiation complete"
        print stats_vec[-5] + " : Copper remote fault detect"
        print stats_vec[-4] + " : Auto-negotiation Enabled"
        print stats_vec[-3] + " : Link Up"
        print stats_vec[-2] + " : Jabber Detected"
        print stats_vec[-1] + " : Extended Capability"
        print " "

        # Get PHY2 Extended Status register
        WriteReg( REG_PHY2_MDIO_CMD, "C00F", 1)
        time.sleep(0.1)
        WriteReg( REG_PHY2_MDIO_CMD, "0000", 1)
        reg_val = bin((int(ReadReg( REG_PHY2_MDIO_RD_DATA)[4:8],16)))[2:].zfill(16)
        stats_vec = reg_val[:]
        print "  1GbE PHY2 Extended Status Register : "
        print stats_vec[-14] + " : Full Duplex 1000 Base"
        print stats_vec[-13] + " : Half Duplex 1000 Base"
        print " "
        time.sleep(2)

#def DecodePHYStatus():
#	Main Status Bit Map
#	Bit 15 == 0
#	Bit 14 == 1 Full Duplex 100 Base
#	Bit 13 == 1 Half Duplex 100 Base
#	Bit 12 == 1 Full Duplex 10 Base
#	Bit 11 == 1 Half Duplex 10 Base
#	Bit 10 == 0
#	Bit  9 == 0
#	Bit  8 == 1 Extended Status
#	Bit  7 == 0
#	Bit  6 == 1 MGMT Frame Preamble Suppression
#	Bit  5 == 1 Copper auto-negotiation complete
#	Bit  4 == 1 Copper remote fault detect
#	Bit  3 == 1 Auto-negotiation Enabled
#	Bit  2 == 1 Link Up
#	Bit  1 == 1 Jabber Detected
#	Bit  0 == 1 Extended Capability

#	Extended Status Bit Map
#	Bit 15 == 0
#	Bit 14 == 0
#	Bit 13 == 1 Full Duplex 1000 Base
#	Bit 12 == 1 Half Duplex 1000 Base
#	Bit 11 >> 0 == 0x0

def getPowerStatusAll():
	print " "
	print "**** CIN Power Monitor "
	print " "
        reg_val = bin((int(ReadReg( REG_PS_ENABLE)[4:8],16)))[2:].zfill(16)
#	print reg_val
        stats_vec = reg_val[:]
	if (int(stats_vec[-1]) == 1):

# ADC == LT4151
		reg_val = ReadReg( REG_VMON_ADC1_CH1 )
		voltage = 0.025*int(reg_val[4:8],16)
		reg_val = ReadReg( REG_IMON_ADC1_CH0 )
		current = 0.00002*int(reg_val[4:8],16)/0.003
		power = voltage * current
		print "  V12P_BUS Power  : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
		print " "

# ADC == LT2418
		reg_val = ReadReg( REG_VMON_ADC0_CH5 )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH5 )
		current = current_calc(reg_val,current)
		print "  V3P3_MGMT Power : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CH7 )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH7 )
		current = current_calc(reg_val,current)
		print "  V2P5_MGMT Power : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CH6 )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH6 )
		current = current_calc(reg_val,current)
		print "  V1P8_MGMT Power : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CH2 )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH2 )
		current = current_calc(reg_val,current)
		print "  V1P2_MGMT Power : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CH3 )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH3 )
		current = current_calc(reg_val,current)
		print "  V1P0_ENET Power : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
		print " "

		reg_val = ReadReg( REG_VMON_ADC0_CH4 )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH4 )
		current = current_calc(reg_val,current)
		print "  V3P3_S3E Power  : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
	
		reg_val = ReadReg( REG_VMON_ADC0_CH8 )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH8 )
		current = current_calc(reg_val,current)
		print "  V3P3_GEN Power  : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
	
		reg_val = ReadReg( REG_VMON_ADC0_CH9 )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CH9 )
		current = current_calc(reg_val,current)
		print "  V2P5_GEN Power  : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
		print " "

		reg_val = ReadReg( REG_VMON_ADC0_CHE )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CHE )
		current = current_calc(reg_val,current)
		print "  V0P9_V6 Power   : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CHB )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CHB )
		current = current_calc(reg_val,current)
		print "  V1P0_V6 Power   : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CHC )
		voltage = 0.00007629*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CHC )
		current = current_calc(reg_val,current)
		print "  V1P2_V6 Power   : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"

		reg_val = ReadReg( REG_VMON_ADC0_CHD )
		voltage = 0.00015258*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CHD )
		current = current_calc(reg_val,current)
		print "  V2P5_V6 Power   : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
		print " "

		reg_val = ReadReg( REG_VMON_ADC0_CHF )
		voltage = 0.00030516*(int(reg_val[4:8],16))
		reg_val = ReadReg( REG_IMON_ADC0_CHF )
		current = current_calc(reg_val,current)
		print "  V_FP Power      : {0:.6s}".format(str(voltage)) + " V  @  {0:.6s}".format(str(current)) + " A"
		print " "

	else:
		print "******* 12V Power Supply is OFF"

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
	if WriteReg( REG_PS_ENABLE, "000F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
#	time.sleep(2)
	
	if WriteReg( REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 0)
	time.sleep(4)

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
	if WriteReg( REG_PS_ENABLE, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 0)
	time.sleep(4)

def CIN_FP_PowerUp():
	print " "
	print "Powering On Front Panel Boards ........  "
	if WriteReg( REG_PS_ENABLE, "003F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
	
	time.sleep(4)

def CIN_FP_PowerDown():
	print " "
	print "Powering Off Front Panel Boards ........  "
	if WriteReg( REG_PS_ENABLE, "001F", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	WriteReg( REG_COMMAND, CMD_PS_ENABLE, 1)
	
	time.sleep(2)

def setCameraOff():
	print " "
	print "Turning off Bias and Clocks in camera head ........  "
	if WriteReg( REG_BIASCONFIGREGISTER0_REG, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)

	if WriteReg( REG_CLOCKCONFIGREGISTER0_REG, "0000", 1) != 1:
		print 'Write register could not be verified. Aborting.'
		sys.exit(1)
	
	time.sleep(1)
	
def CINLoadFirmware(filename):
	print " "
	print "Loading Frame (FRM) FPGA Configuration ...........  "
	print "File: " + filename
	WriteReg( REG_COMMAND, CMD_PROGRAM_FRAME, 0)
	time.sleep(1)
	with open(filename, 'rb') as f:
		read_data = f.read(128)
		while read_data != "":
		  cin_sock.sendto(read_data,(CIN_CONFIG_IP, CIN_STREAM_IN_PORT))
		  time.sleep(0.0002)	# For UDP flow control (was 0.002)
		  read_data = f.read(128)
	f.closed
	time.sleep(1)
	WriteReg( REG_FRM_RESET, "0001", 0)
	WriteReg( REG_FRM_RESET, "0000", 0)
	time.sleep(1)
	# need to verify sucess!

def loadFrmFirmware(filename):
	print " "
	print "Loading Frame (FRM) FPGA Configuration Data ...........  "
	print "File: " + filename
	WriteReg( REG_COMMAND, CMD_PROGRAM_FRAME, 0)
	time.sleep(1)
	with open(filename, 'rb') as f:
		read_data = f.read(128)
		while read_data != "":
		  cin_sock.sendto(read_data,(CIN_CONFIG_IP, CIN_STREAM_IN_PORT))
		  time.sleep(0.0001)	# For UDP flow control (was 0.002)
		  read_data = f.read(128)
	f.closed
	time.sleep(1)
	WriteReg( REG_FRM_RESET, "0001", 0)
	WriteReg( REG_FRM_RESET, "0000", 0)
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
			file_line = f.readline()
	f.closed

def setCCD_BiasOn():
	WriteReg(REG_BIASCONFIGREGISTER0_REG, "0001", 0)

def setCCD_BiasOff():
	WriteReg(REG_BIASCONFIGREGISTER0_REG, "0000", 0)

def setCCD_ClocksOn():
	WriteReg(REG_CLOCKCONFIGREGISTER0_REG, "0001", 0)

def setCCD_ClocksOff():
	WriteReg(REG_CLOCKCONFIGREGISTER0_REG, "0000", 0)

#def setCCD_DelayTime():

#def setCCD_ExposureTime():
#	from sys import argv
#	exp_time = argv
#	ms_data = hex ( double (exp_time*50.0) / 65536);
#	ls_data = hex ( double (exp_time*50.0)) % 65536;
#
#	WriteReg(REG_EXPOSURETIMEMSB_REG, ms_data, 0)
#	WriteReg(REG_EXPOSURETIMELSB_REG, ls_data, 0)

#def set_FrameCountReset():

def readPHY1_Registers(cmd):
	"""
	Function to access the MDIO Registers in PHY1
	usage (cmd)
	Bit [15]   == Start(1)
	Bit [14]   == RnW(1)
	Bit [13]   == WatchDog Rd WDRd(1)
	Bit [12]   == Not Used
	Bits[11:6] == PHY Addr(5), Always 0 
	Bits[ 5:0] == MDIO REG Addr(5) 
	"""
	WriteReg( REG_PHY1_MDIO_CMD, cmd, 1) 
	WriteReg( REG_PHY1_MDIO_CMD, "0000", 1) 
	print ReadReg( REG_PHY1_MDIO_RD_ADDR)
	print ReadReg( REG_PHY1_MDIO_RD_DATA)
	print ""

def readPHY2_Registers(cmd):
	"""
	Function to access the MDIO Registers in PHY2
	usage (cmd)
	Bit [15]   == Start(1)
	Bit [14]   == RnW(1)
	Bit [13]   == WatchDog Rd WDRd(1)
	Bit [12]   == Not Used
	Bits[11:6] == PHY Addr(5), Always 0 
	Bits[ 5:0] == MDIO REG Addr(5) 
	"""
	WriteReg( REG_PHY2_MDIO_CMD, cmd, 1) 
	WriteReg(REG_PHY2_MDIO_CMD, "0000", 1) 
	print ReadReg( REG_PHY2_MDIO_RD_ADDR)
	print ReadReg( REG_PHY2_MDIO_RD_DATA)
	print ""


def flashFpCfgLeds():
# Test Front Panel LEDs
	print " "
	print "Flashing CFG FP LEDs  ............ "
	WriteReg( REG_SANDBOX_REG00, "AAAA", 1)
	time.sleep(1)
	WriteReg( REG_SANDBOX_REG00, "5555", 1)
	time.sleep(1)
	WriteReg( REG_SANDBOX_REG00, "FFFF", 1)
	time.sleep(1)
	WriteReg( REG_SANDBOX_REG00, "0001", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0002", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0004", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0008", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0010", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0020", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0040", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0080", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0100", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0200", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0400", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0800", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "1000", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "2000", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "4000", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "8000", 1)
	time.sleep(0.4)
	WriteReg( REG_SANDBOX_REG00, "0000", 1)


# ---------------------------------------------< Frame FPGA functions >
def getFrmFpgaStat():
# get Status Registers
	print " "
	print "**** Frame FPGA Status Registers  "
	print " "
	reg_val = ReadReg( REG_FRM_FPGA_VERSION )
	print "  FRM FPGA Version : " + reg_val[4:]
	reg_val = ReadReg( REG_FRM_BOARD_ID )
	print "  CIN Board ID     : " + reg_val[4:]
	print " "
	reg_val = ReadReg( REG_FRM_DCM_STATUS )
#        reg_val = bin((int(ReadReg( REG_FRM_DCM_STATUS )[4:8],16)))[2:].zfill(16)
#        stats_vec = reg_val[:]
#	if (int(stats_vec[-16]) == 1):
#		print "  *** Frame FPGA Configuration Done"
#	else:
#		print "  *** Frame FPGA NOT Configured"
# "000"                     &
# xaui_align_status_fab2    &  -- b12
# xaui_txlock_fab2          &  -- b11
# mac_tx_ll_dst_rdy_n_fab2  &  -- b10
# mac_rx_dcm_locked_fab2    &  -- b09 
# tx_mmcm_locked_fab2       &  -- b08
# "000"                     &
# xaui_align_status_fab1    &  -- b04
# xaui_txlock_fab1          &  -- b03 
# mac_tx_ll_dst_rdy_n_fab1  &  -- b02 
# mac_rx_dcm_locked_fab1    &  -- b01 
# tx_mmcm_locked_fab1;         -- b00
	print "  DCM Status       : " + reg_val[4:]
	reg_val = ReadReg( REG_FRM_FPGA_STATUS )
	print "  FPGA Status      : " + reg_val[4:]
# frm_10gbe_port_sel_i      &  -- b15
# "00000000000000"          & 
# pixel_clk_sel             &  -- b0
	print " "




def setFCLK125M():
	print "**** Set CIN FCLK to 125MHz"
	# Freeze DCO
	WriteReg( REG_FCLK_I2C_ADDRESS, "B089", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F010", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
        print "  Write to Reg 137 - Freeze DCO"

        print "  Set Si570 Oscillator Freq to 125MHz" 
	# WR Reg 7
	WriteReg( REG_FCLK_I2C_ADDRESS, "B007", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F002", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)	

	# WR Reg 8
	WriteReg( REG_FCLK_I2C_ADDRESS, "B008", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F042", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 9
	WriteReg( REG_FCLK_I2C_ADDRESS, "B009", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F0BC", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 10
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00A", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F019", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 11
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00B", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F06D", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 12
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00C", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "f08f", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# UnFreeze DCO
	WriteReg( REG_FCLK_I2C_ADDRESS, "B089", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F000", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	WriteReg( REG_FCLK_I2C_ADDRESS, "B087", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F040", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
        print "  Write to Reg 137 - UnFreeze DCO & Start Oscillator"
        print "  "

	time.sleep(1)

def setFCLK250M():
	print "**** Set CIN FCLK to 250MHz"
	# Freeze DCO
	WriteReg( REG_FCLK_I2C_ADDRESS, "B089", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F010", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
        print "  Write to Reg 137 - Freeze DCO"

        print "  Set Si570 Oscillator Freq to 250MHz" 
	# WR Reg 7
	WriteReg( REG_FCLK_I2C_ADDRESS, "B007", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F020", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)	

	# WR Reg 8
	WriteReg( REG_FCLK_I2C_ADDRESS, "B008", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F0C2", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 9
	WriteReg( REG_FCLK_I2C_ADDRESS, "B009", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F0BC", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 10
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00A", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F019", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 11
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00B", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F06D", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# WR Reg 12
	WriteReg( REG_FCLK_I2C_ADDRESS, "B00C", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "f08f", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)

	# UnFreeze DCO
	WriteReg( REG_FCLK_I2C_ADDRESS, "B089", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F000", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	WriteReg( REG_FCLK_I2C_ADDRESS, "B087", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "F040", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
        print "  Write to Reg 137 - UnFreeze DCO & Start Oscillator"
        print "  "

	time.sleep(1)

def setXFCLK250M():
	print "**** Set CIN FCLK to 250MHz - Fixed XO"
	# 
#	WriteReg( REG_FCLK_I2C_ADDRESS, "B089", 0)
	WriteReg( REG_FCLK_I2C_DATA_WR, "2000", 0)
#	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val = ReadReg( REG_FCLK_I2C_DATA_WR )
	print reg_val


def getFCLK():

	print "**** Readback CIN FCLK configuration"
	# Freeze DCO
	WriteReg( REG_FCLK_I2C_ADDRESS, "B189", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val = ReadReg( REG_FCLK_I2C_DATA_RD )
	if (reg_val[6:] != "08") : print "  Status Reg : 0x" + reg_val[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B107", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)	
	reg_val7 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg07 : " + reg_val7[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B108", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val8 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg08 : " + reg_val8[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B109", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val9 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg09 : " + reg_val9[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B10A", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val10 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg10 : " + reg_val10[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B10B", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val11 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg11 : " + reg_val11[6:]

	WriteReg( REG_FCLK_I2C_ADDRESS, "B10C", 0)
	WriteReg( REG_FRM_COMMAND, CMD_FCLK_COMMIT, 0)
	reg_val12 = ReadReg( REG_FCLK_I2C_DATA_RD )
#	print "  FCLK Reg12 : " + reg_val12[6:]

	bin_reg7 = bin(int(reg_val7[6:],16))[2:].zfill(8)
	bin_reg8 = bin(int(reg_val8[6:],16))[2:].zfill(8)

	if (bin_reg7[0:3] == "000") : print "  FCLK HS Divider = 4" 
	if (bin_reg7[0:3] == "001") : print "  FCLK HS Divider = 5" 
	if (bin_reg7[0:3] == "010") : print "  FCLK HS Divider = 6" 
	
	bin_n1 = bin_reg7[3:8] + bin_reg8[0:2]
	dec_n1 = int(bin_n1,2)
	if (dec_n1%2 != 0) : dec_n1 = dec_n1 + 1	

	print "  FCLK N1 Divider = " + str(dec_n1)
	print "  FCLK RFREQ = " + reg_val8[7:] + reg_val9[6:7] + "." + reg_val9[7:] + reg_val10[6:] + reg_val11[6:] + reg_val12[6:]
        print "  "
	
	if   (bin_reg7[0:3] == "000" and dec_n1 ==  8) : print "  FCLK Frequency = 156 MHz"
	elif (bin_reg7[0:3] == "000" and dec_n1 == 10) : print "  FCLK Frequency = 125 MHz"
	elif (bin_reg7[0:3] == "001" and dec_n1 ==  4) : print "  FCLK Frequency = 250 MHz"
	else : print "  FCLK Frequency UNKNOWN "
        print "  "
	time.sleep(0.1)


def streamReadFRM():
#
	print " "
	print "***++ Write Start Addr (Sandbox00 0x00F0) & Stream Type ++***"

	WriteReg( REG_READ_ADDRESS, REG_SANDBOX_REG00, 1 )
	WriteReg( REG_STREAM_TYPE, "0001", 1 )

	print "SANDBOX00 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX01 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX02 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX03 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX04 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX05 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX06 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX07 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX08 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX09 = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0A = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0B = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0C = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0D = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0E = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)
	print "SANDBOX0F = 0x" + ReadStream( CIN_CONFIG_IP, CIN_COMMAND_PORT)

	WriteReg( REG_STREAM_TYPE, "0000", 1 )


def flashFpFrmLeds():
# Test Front Panel LEDs
	print " "
	print "Flashing FRM FP LEDs  ............ "
	WriteReg( REG_FRM_SANDBOX_REG00, "0004", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0008", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "000C", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0010", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0020", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0030", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0040", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0080", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "00C0", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0100", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0200", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0300", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0400", 1)
	print "RED  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0800", 1)
	print "GRN  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0C00", 1)
	print "YEL  ............ "
	time.sleep(0.5)
	WriteReg( REG_FRM_SANDBOX_REG00, "0000", 1)
	print "All OFF  ............ "
	time.sleep(0.5)
	
def getFrmEthStatus():
        print " "
        print "*******************  Frame FPGA Ethernet Status  ***************** "
        print " "
        print "--- MAC Configuration: "
 
        reg_val = ReadReg( REG_IF_MAC_FAB1B2)[4:8]
        reg_val = reg_val + ReadReg( REG_IF_MAC_FAB1B1)[4:8]
        reg_val = reg_val + ReadReg( REG_IF_MAC_FAB1B0)[4:8]
 
        print "MAC Address: " + reg_val[0:2] + ":" + reg_val[2:4] + ":" + reg_val[4:6] + ":" + reg_val[6:8] + ":" + reg_val[8:10] + ":" + reg_val[10:12]
 
        reg_val = ReadReg( REG_IF_IP_FAB1B1)[4:8]
        reg_val = reg_val + ReadReg( REG_IF_IP_FAB1B0)[4:8]
 
        print "IP Address: " + str(int(reg_val[0:2],16)) + "." + str(int(reg_val[2:4],16)) + "." + str(int(reg_val[4:6],16)) + "." + str(int(reg_val[6:8],16))
 
        reg_val = bin((int(ReadReg( REG_MAC_CONFIG_VEC_FAB1B1)[4:8],16)))[2:].zfill(16)
        reg_val = reg_val + bin((int(ReadReg( REG_MAC_CONFIG_VEC_FAB1B0)[4:8],16)))[2:].zfill(16)
 
        mac_config_vec = reg_val[:]
 
        print "MAC Configuration: " + mac_config_vec
        print mac_config_vec[-21] + " : Control Frame Length Check Disable"
        print mac_config_vec[-20] + " : Receiver Length/Type Error Disable"
        print mac_config_vec[-19] + " : Receiver Preserve Preamble Enable"
        print mac_config_vec[-18] + " : Transmitter Preserve Preamble Enable"
        print mac_config_vec[-17] + " : Reconciliator Sublayer Fault Inhibit"
        print mac_config_vec[-16] + " : Reserved"
        print mac_config_vec[-15] + " : Deficite Idle Cont Enable"
        print mac_config_vec[-14] + " : TX Flow Control"
        print mac_config_vec[-13] + " : RX Flow Control"
        print mac_config_vec[-12] + " : TX Reset"
        print mac_config_vec[-11] + " : TX Jumbo Enable"
        print mac_config_vec[-10] + " : TX FCS Enable"
        print mac_config_vec[ -9] + " : TX Enable"
        print mac_config_vec[ -8] + " : TX VLAN Enable"
        print mac_config_vec[ -7] + " : Adjustable Frame Gaps"
        print mac_config_vec[ -6] + " : Large Frame Gaps"
        print mac_config_vec[ -5] + " : RX Reset"
        print mac_config_vec[ -4] + " : RX Jumbo Enable"
        print mac_config_vec[ -3] + " : Pass FCS Enable"
        print mac_config_vec[ -2] + " : RX Enable"
        print mac_config_vec[ -1] + " : RX VLAN Enable"
 
        time.sleep(1)

        reg_val = bin((int(ReadReg( REG_MAC_STATS1_FAB1B1)[4:8],16)))[2:].zfill(16)
        reg_val = reg_val + bin((int(ReadReg( REG_MAC_STATS1_FAB1B0)[4:8],16)))[2:].zfill(16)
 
        stats1_vec = reg_val[:]
 
        print " "
        print "MAC TX Stats: " + stats1_vec
        # print stats1_vec[-32] + " : " # Not Used
        print stats1_vec[-31] + " : XAUI Sync[3]" # XAUI SYNC
        print stats1_vec[-30] + " : XAUI Sync[1]"
        print stats1_vec[-29] + " : XAUI Sync[1]"
        print stats1_vec[-28] + " : XAUI Sync[0]"
        print stats1_vec[-27] + " : MAC Remote Fault Received" # MAC Status Vector
        print stats1_vec[-26] + " : MAC Local Fault Received" # 
        print str(int(stats1_vec[-25:-20],2)) + " : TX Bytes Valid on TX Clock" # MAC TX Statistics Vector (downto -1)
        print stats1_vec[-20] + " : TX Previous Frame Was a VLAN Frame"
        print str(int(stats1_vec[-19:-5],2)) + " : TX Previous Frame Length"
        print stats1_vec[ -5] + " : TX Previous Frame Was a Control Frame"
        print stats1_vec[ -4] + " : TX Previous Frame Terminated due to underrun"
        print stats1_vec[ -3] + " : TX Previous Frame Was a Multicast Frame"
        print stats1_vec[ -2] + " : TX Previous Frame Was a Broadcast Frame"
        print stats1_vec[ -1] + " : TX Previous Frame Transmited without Error" 
 
        time.sleep(1)

        reg_val = bin((int(ReadReg( REG_MAC_STATS2_FAB1B1)[4:8],16)))[2:].zfill(16)
        reg_val = reg_val + bin((int(ReadReg( REG_MAC_STATS2_FAB1B0)[4:8],16)))[2:].zfill(16)
 
        stats2_vec = reg_val[:]
 
        print " "
        print "MAC RX Stats: " + stats2_vec
        # print stats1_vec[-32] + " : " # Not Used
        print stats2_vec[-30] + " : XAUI TX Ready"
        print stats2_vec[-29] + " : RX Header Length/Type Does Not Match Data Length"
        print stats2_vec[-28] + " : RX Frame Had Unsuported OP-Code" 
        print stats2_vec[-27] + " : RX Previous Frame Was a Flow Control Frame"
        print str(int(stats2_vec[-26:-22],2)) + " : RX Bytes Valid on RX Clock"
        print stats2_vec[-22] + " : RX Previous Frame Was a VLAN Frame"
        print stats2_vec[-21] + " : RX Previous Frame Was Out of Bounds (too long)"
        print stats2_vec[-20] + " : RX Previous Frame Was a Control Frame"
        print str(int(stats2_vec[-19:-5],2)) + " : RX Previous Frame Length"
        print stats2_vec[ -5] + " : RX Previous Frame Was a Multicast Frame"
        print stats2_vec[ -4] + " : RX Previous Frame Was a Broadcast Frame"
        print stats2_vec[ -3] + " : RX Previous FCS Errors or MAC Code Errors"
        print stats2_vec[ -2] + " : RX Previous Frame Received WITH Errors"
        print stats2_vec[ -1] + " : RX Previous Frame Received without Errors" 
 
        time.sleep(1)

        reg_val = bin((int(ReadReg( REG_XAUI_FAB1B)[4:8],16)))[2:].zfill(16)
 
        xaui_stats_vec = reg_val[:]
 
        print " "
        print "XAUI Stats/Configuration: " + xaui_stats_vec
        print xaui_stats_vec[-16] + " : XAUI RX Link Status"
        print xaui_stats_vec[-15] + " : XAUI Alignment"  
        print xaui_stats_vec[-14] + " : XAUI Sync[3]"
        print xaui_stats_vec[-13] + " : XAUI Sync[2]"
        print xaui_stats_vec[-12] + " : XAUI Sync[1]"
        print xaui_stats_vec[-11] + " : XAUI Sync[0]"
        print xaui_stats_vec[-10] + " : XAUI TX Local Fault"
        print xaui_stats_vec[ -9] + " : XAUI RX Local Fault"
 
        print xaui_stats_vec[ -7:-5] + " : XAUI Test Mode Testpatern: (00):High (01):Low (10):Mixed"
        print xaui_stats_vec[ -5] + " : XAUI Enable Test Mode"
        print xaui_stats_vec[ -4] + " : XAUI Reset RX Link Status"
        print xaui_stats_vec[ -3] + " : XAUI Reset Local Fault"
        print xaui_stats_vec[ -2] + " : XAUI Power Down"
        print xaui_stats_vec[ -1] + " : XAUI Loopback Mode"
 
        WriteReg( REG_XAUI_FAB1B, "000C", 1)
        WriteReg( REG_XAUI_FAB1B, "0000", 1)
 
        WriteReg( REG_MAC_CONFIG_VEC_FAB1B0, "0D9B", 1)
        WriteReg( REG_MAC_CONFIG_VEC_FAB1B0, "058B", 1)

def setSingleSRAM( Addr, Data ):
        WriteReg( REG_SRAM_START_ADDR1, Addr[0:4], 0 )
        time.sleep(0.01)
        WriteReg( REG_SRAM_START_ADDR0, Addr[4:8], 0 )
        time.sleep(0.01)
        WriteReg( REG_SRAM_FRAME_DATA_OUT1, Data[0:4], 0 )
        time.sleep(0.01)
        WriteReg( REG_SRAM_FRAME_DATA_OUT0, Data[4:8], 0 )
        time.sleep(0.01)

        WriteReg( REG_SRAM_COMMAND, "0000" , 0 )
        time.sleep(0.01)
        WriteReg( REG_SRAM_COMMAND, "0010" , 0 )
        time.sleep(0.01)

def getSingleSRAM( Addr ):
        WriteReg( REG_SRAM_START_ADDR1, Addr[0:4], 0 )
        time.sleep(0.1)
        WriteReg( REG_SRAM_START_ADDR0, Addr[4:8], 0 )
        time.sleep(0.1)
        WriteReg( REG_SRAM_COMMAND, "0001", 0 )
        time.sleep(0.1)
        WriteReg( REG_SRAM_COMMAND, "0011", 0 )
        time.sleep(0.1)
        Data1 = ReadReg( REG_SRAM_FRAME_DATA_IN1 )
        time.sleep(0.1)
        Data0 = ReadReg( REG_SRAM_FRAME_DATA_IN0 )
        time.sleep(0.1)
        return Data1[4:8] + Data0[4:8]

