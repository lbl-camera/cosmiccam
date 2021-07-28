#! /usr/bin/python
# -*- coding: utf-8 -*-
import cin_constants
import cin_register_map
import cin_functions
import time

regval = cin_functions.ReadReg( cin_register_map.REG_TRIGGERSELECT_REG )
print "  TRIGGER SELECT REG VALUE: 0x" + regval[4:]
print "  "
time.sleep(0.1)



