import logging
import socket
import time
import zmq
from zmq import ssh
from PyQt4 import QtCore

from ..comm.zmqsocket import ZmqSocket
import nanosurveyor.camera.udpframereader as udpr

class Framegrabber(QtCore.QThread):
    """Grab frames from the camera and send assembled frames to backend."""

    statusMessage = QtCore.pyqtSignal(str)
    sizeUploaded  = QtCore.pyqtSignal(int)

    def __init__(self, readport, sendport,cin_ip="10.0.5.207",cin_port=49203,fsize= 2*1152*1940):
        QtCore.QThread.__init__(self)

        # Configuration
        self.readport = readport
        self.sendport = sendport
        self.cin_address = (cin_ip, cin_port)
        # Size of frame (unit16)
        self.fsize = fsize
        
        # Buffers and counters
        self.fbuffer  = None
        self.fnumber = None
        self.fnumber0 = 0
        self.nreceive = 0
        self.nsend = 0
        self.updaterate = 100
        self.t0 = time.time()

    def createReadFrameSocket(self):
        # A socket for reading from camera
        self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.camera_socket.setblocking(1)
        self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        print "binding to readport", self.readport
        self.camera_socket.bind(('', self.readport))
        self.camera_socket.sendto("dummy_data", self.cin_address)
        self.status = "<font color=\"orange\"> Framegrabber is listening to data from the camera on port %d" %(self.readport) + "</font>"
        self.statusMessage.emit(self.status)
            
    def createSendFrameSocket(self):
        # A socket for sending data (including meta data) to the backend
        self.backend_socket = ZmqSocket(zmq.PUB)
        self.backend_socket.bind('tcp://*:%d' %(self.sendport))
        self.status = "<font color=\"orange\"> Framegrabber is sending data to backend on %d" %(self.sendport) + "</font>"
        self.statusMessage.emit(self.status)

    def _recvframe(self):
        """receive frames from the FCCD. Return as soon as a new frame is ready, otherwise it is blocking."""
        print "RECV: ", self.camera_socket.fileno(), self.fsize
         
        #data = self.camera_socket.recv(self.fsize)
        #print len(data)
        self.fbuffer, self.fnumber = udpr.read_frame(self.camera_socket.fileno(), self.fsize)
        if self.fbuffer is not None:
          print "READ: ", len(self.fbuffer), self.fnumber
        self.nreceive += 1
        if (self.nreceive == self.updaterate):
            t1 = time.time()
            self.status = "<font color=\"blue\"> Reading at %.2fHz" %(self.nreceive/(t1-self.t0)) + "</font>"
            self.statusMessage.emit(self.status)
            logging.debug("dropped %d frames", (self.fnumber + 1 - self.fnumber0) - self.updaterate)
            self.fnumber0 = self.fnumber + 1
            self.t0 = t1
            self.nreceive = 0

    def _sendframe(self):
        """Sending frames to the processing backend."""
        if self.fbuffer is not None:
          logging.debug("sending out a frame, id = %d", self.fnumber)
          self.backend_socket.send("rawframe", zmq.SNDMORE)
          self.backend_socket.send_multipart([str(self.fnumber), self.fbuffer])
          self.sizeUploaded.emit(self.fsize)
            
    def run(self):
        """This triggers the event loop."""
        self.status = "<font color=\"blue\"> Framegrabber is starting the event loop </font>"
        self.statusMessage.emit(self.status)
        self.isRunning = True
        self.hasFinished = False
        print "running "
        while self.isRunning:
            #try:
              self._recvframe()
              self._sendframe()
            #except:
            #  print "exception occurred..."
            #  pass
        print "has finished"
        self.hasFinished = True
        
    def stop(self):
        """Stop the event loop and close sockets."""
        self.status = "<font color=\"red\"> Framegrabber is stopping the event loop </font>"
        self.statusMessage.emit(self.status)
        self.isRunning = False
        #while not self.hasFinished:
        #    #print "Waiting for framegrabber to be finished ..."
        #    time.sleep(0.1)
        self.backend_socket.close()
        self.camera_socket.close()
        print "Framegrabber has finished"
    
        
