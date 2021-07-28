#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions

# Mask Triggers & turn off Bias
#import setTriggerSW
#cin_functions.setCameraOff()

# Write CompV
cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0020", 0 )
cin_functions.WriteReg ("821F", "00A3", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("821D", "A000", 0 )
cin_functions.WriteReg ("821E", "0021", 0 )
cin_functions.WriteReg ("821F", "00C7", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )


