#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions

# Write to FO Module Register to send Test Pattern
cin_functions.WriteReg ("821D", "9E00", 0 )
cin_functions.WriteReg ("821E", "0000", 0 )
cin_functions.WriteReg ("821F", "0001", 0 )
cin_functions.WriteReg ("8001", "0105", 0 )

cin_functions.WriteReg ("8211", "0000", 0 )
cin_functions.WriteReg ("8212", "0000", 0 )
cin_functions.WriteReg ("8213", "0000", 0 )
