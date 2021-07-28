#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import time
import cin_constants
import cin_register_map
import cin_functions

ClampVoltage = str(sys.argv[1])
#print ClampVoltage

# Device Locator Word
cin_functions.WriteReg ("821D", "9E00", 0 )
# Register Address
cin_functions.WriteReg ("821E", "0001", 0 )

# Register Data
if   (ClampVoltage == "1.60") : cin_functions.WriteReg ("821F", "8055", 0 )
elif (ClampVoltage == "1.65") : cin_functions.WriteReg ("821F", "8054", 0 )
elif (ClampVoltage == "1.70") : cin_functions.WriteReg ("821F", "8051", 0 )
elif (ClampVoltage == "1.75") : cin_functions.WriteReg ("821F", "8050", 0 )
elif (ClampVoltage == "1.80") : cin_functions.WriteReg ("821F", "8045", 0 )
elif (ClampVoltage == "1.85") : cin_functions.WriteReg ("821F", "8044", 0 )
elif (ClampVoltage == "1.90") : cin_functions.WriteReg ("821F", "8041", 0 )
elif (ClampVoltage == "1.95") : cin_functions.WriteReg ("821F", "8040", 0 )
elif (ClampVoltage == "2.00") : cin_functions.WriteReg ("821F", "8015", 0 )
elif (ClampVoltage == "2.05") : cin_functions.WriteReg ("821F", "8014", 0 )
elif (ClampVoltage == "2.10") : cin_functions.WriteReg ("821F", "8011", 0 )
elif (ClampVoltage == "2.15") : cin_functions.WriteReg ("821F", "8010", 0 )
elif (ClampVoltage == "2.20") : cin_functions.WriteReg ("821F", "8005", 0 )
elif (ClampVoltage == "2.25") : cin_functions.WriteReg ("821F", "8004", 0 )
elif (ClampVoltage == "2.30") : cin_functions.WriteReg ("821F", "8001", 0 )
elif (ClampVoltage == "2.35") : cin_functions.WriteReg ("821F", "8000", 0 )
else : 	
	print "Invalid Voltage (1.60 to 2.35 in 0.05 steps only)"
	exit

# Send Data Command
cin_functions.WriteReg ("8001", "0105", 0 )


