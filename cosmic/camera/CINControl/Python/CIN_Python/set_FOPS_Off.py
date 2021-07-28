#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import string
import time
import socket

# ============================================================================
#               System Constants
# ============================================================================
E36102A_IP   = "192.168.1.3"
E36102A_PORT = 5025

# Create a socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
s.connect((E36102A_IP,E36102A_PORT))

# Clear Communications
s.sendall('*CLS\r\n')
# Wait at least 50ms before sending next command
time.sleep(0.1)

# Turn off FO Power Supply Output
s.sendall('OUTP 0 \r\n')

time.sleep(0.2)
import get_FOPS_Status

s.close()

