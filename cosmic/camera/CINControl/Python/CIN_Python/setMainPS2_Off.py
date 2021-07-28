#! /usr/bin/python
# -*- coding: utf-8 -*-
import urllib2
 
# ============================================================================
#               System Constants
# ============================================================================

url = 'http://192.168.1.2/state.xml'
out2_off = 'relay2State=0'

urllib2.urlopen(url + '?' + out2_off,timeout=0.02)
