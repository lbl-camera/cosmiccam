#! /usr/bin/python
# -*- coding: utf-8 -*-
import time
import socket
#import argparse 

# ============================================================================
#		CIN Registers
# ============================================================================
# -----------------------------------< Configuration FPGA Registers >
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

# ---------------------------------< Frame FPGA Registers >

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
	
REG_TRIGGERSELECT_REG		      	= "8050"
REG_TRIGGERMASK_REG		      	= "8051"  # [00]==SW Trigger, [01]==FP TrigIn2, [10]==FP TrigIn1, [11]==FP TrigIn1OR2  
REG_CCDFCLKSELECT_REG		      	= "8052"
REG_CDICLKDISABLE_REG		      	= "8053"

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
REG_ALTEXPOSURETIMEMSB_REG	      	= "8306"
REG_ALTEXPOSURETIMELSB_REG	      	= "8307"

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
#                                                       CIN Commands
# ============================================================================
# ------------------------------------------------------< Common Commands: >
# Common Commands
CMD_READ_REG            = "0001" # Read Register

# ------------------------------------------------------< Frame FPGA Commands: >
# Programmable FCLK Commands


# Image Processor Commands

