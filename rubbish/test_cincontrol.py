
from cosmic.camera import cincontrol
import time
from PyQt4 import QtCore,QtNetwork

import logging
logging.getLogger().setLevel(20)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

p = {
'cam_port': 8880,       #irrelevant 
'cam_ip': 'localhost',  #irrelevant 
'cin_ip': '127.0.0.1',
'stxm_port': 8880
}

QA = QtCore.QCoreApplication([])

CC = cincontrol.CINController(**p)
#CC.createSendCommandsSocket()
CC.createReceiveCommandsSocket_qtcpserver()

QA.exec_()
