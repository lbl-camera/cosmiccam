#! /usr/bin/python
# -*- coding: utf-8 -*-
#import sys
#import cin_functions

#print "  *********************  "
#print "SCRIPT NOT YET FUNCTIONAL!!!!"
#print "  *********************  "

#addr = sys.argv[1]
#data = sys.argv[2]
#mask = sys.argv[3]

#print data
#print mask

#regval = cin_functions.ReadReg( addr )[4:]
#print regval

#print "  *********************  "
#tdata = regval & mask
#print tdata
# wdata == tdata | data

#wdata = str( 1111 )

#cin_functions.WriteReg( addr, wdata, 0 )
#regval = cin_functions.ReadReg( addr ) 
#print regval[4:]

#reg_val = bin((int(cin_functions.ReadReg( addr )[4:8],16)))[2:].zfill(16)

#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import cin_functions
 
addr   = str(sys.argv[1])
data   = int(sys.argv[2],16)
bitpos = int(sys.argv[3])
width  = int(sys.argv[4])
 
# Select Bit Mask
count = 0
maskbit = 0x0000
while (count < width):
        maskbit = maskbit << 1
        maskbit = maskbit | 0x0001
        count = count + 1
 
# Create bit clearing mask
temp = maskbit << int(bitpos)
clrval = ~temp & 0xFFFF
# Create bitwise insert value
insval = (maskbit & data) << int(bitpos)
 
# Read Selected Register value
regval = cin_functions.ReadReg( addr )[4:]
# Clear write location
twdata = int(regval,16) & clrval
# Input new data bits
wdata = twdata | insval
# Format Data word for WriteReg Function
data = "{0:0>4}".format(str(hex(wdata)).lstrip("0x"))
# Write new data word
cin_functions.WriteReg( addr, data, 1 )
 
#regval = cin_functions.ReadReg( addr )[4:]
#print regval
