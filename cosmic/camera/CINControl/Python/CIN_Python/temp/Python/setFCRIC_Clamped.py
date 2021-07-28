#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions

# Mask Triggers & turn off Bias
#import setTriggerSW
#cin_functions.setCameraOff()

# Clamp Mode registers
# Write clampr
cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0048", 0 )
cin_functions.WriteReg ("821F", "0001", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0049", 0 )
cin_functions.WriteReg ("821F", "00FF", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

# Write clamp
cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0050", 0 )
cin_functions.WriteReg ("821F", "0001", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0051", 0 )
cin_functions.WriteReg ("821F", "00FF", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

# Write ac on
cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0058", 0 )
cin_functions.WriteReg ("821F", "00FF", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0059", 0 )
cin_functions.WriteReg ("821F", "0001", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "005A", 0 )
cin_functions.WriteReg ("821F", "00FF", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "005B", 0 )
cin_functions.WriteReg ("821F", "0001", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

# Bias On & allow Ext Triggers
#cin_functions.setCameraOn()
#import setTrigger0

