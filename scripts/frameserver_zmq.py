"""
Experimental framgrabber that should work without pyQT4 and zmq

Can actualluy be deleted once the framegrabbing is stable
"""


import logging
import socket
import time
import numpy as np
from collections import deque
import urllib.request, urllib.error, urllib.parse
import zmq
import tifffile
from cosmic.camera.fccd import FCCD

logger = logging.getLogger('Frameserver')

def splitaddr(addr):
    host, port = urllib.parse.splitport(addr)
    if port is None:
        return host, port
    else:
        return host,int(port)

class DummySignal(object):

    def __init__(self,tpe=str):
        
        self.tpe=tpe

    def emit(self,msg):
        logger.info(self.tpe(msg))
    
try:
    from PyQt4 import QtCore
    __has_qt4 = True
    Thread = QtCore.QThread
    Signal = QtCore.pyqtSignal
except:
    __has_qt4 = False
    import threading
    Thread = threading.Thread
    Signal = DummySignal
    
import threading
Thread = threading.Thread
Signal = DummySignal

def to_image(name, b):

    shape = (1040,1152)
    num_rows = shape[0] // 2
    num_adcs = shape[1] // 6
    CCD = FCCD(nrows=num_rows)
	
    npbuf = np.frombuffer(b, '<u2')
    npbuf = npbuf.reshape((12 * num_rows, num_adcs))
    image = CCD.assemble2(npbuf.astype(np.uint16))
    tifffile.imsave(name, image)

#class Framegrabber(QtCore.QThread):
class Framegrabber(Thread):
    """Grab frames from the camera and send assembled frames to backend."""

    newFrame= Signal(int)
    statusMessage = Signal(str)

    def __init__(self, fsize, 
                       read_addr= "localhost:49205", 
                       send_addr= "127.0.0.1:50000", 
                       udp_addr ="10.0.5.207:49209"):
        
        Thread.__init__(self)
        
        # Configuration
        self.read_addr = splitaddr(read_addr)
        self.send_addr = splitaddr(send_addr) if send_addr is not None else None
        self.udp_addr = splitaddr(udp_addr) 
        
        # Size of frame (unit16)
        self.fsize = fsize
        self.fbytes = None
        self._status = ''
        
        # Buffers and counters
        self.fbuffer  = None
        self.lastbuffer = None
        self.fnumber = -1
        self.fnumber0 = 0
        self.fnumber1 = 0
        self.nreceive = 0
        self.nrecord = 0
        self.nsend = 0
        self.updaterate = 100
        self.t0 = time.time()
        self.t1 = time.time()
        
        self.circbuffer = deque(maxlen = 200)
        self.extbuffer = None
        
        # udpreader
        import cosmic.camera.udpframereader as udpr
        self.udpr = udpr
        self.status = "-- Starting ZMQ Frame publisher --"

    
    @property
    def status(self):
        return self._status
            
    @status.setter
    def status(self, stat):
        self._status = stat
        self.statusMessage.emit(stat)
        
    def createReadFrameSocket(self):
        # A socket for reading from camera
        self.camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.camera_socket.setblocking(1)
        self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #self.camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        self.camera_socket.bind(self.read_addr)
        self.camera_socket.sendto(b"dummy_data", self.udp_addr)
        self.status = "Framegrabber is listening to data from the camera on ip %s port %d" % self.read_addr
        
    def udpdump_to_file(self,filename='/tmp/udpdump.hex',npackets = 2000, packet_size=4104):
        
        f = open(filename,'w')
        for i in range(npackets):
            f.write(self.camera_socket.recv(packet_size))
        f.close()
            
        

    def createSendFrameSocket(self):
        # A socket for sending data (including meta data) to the backend
        context = zmq.Context()
        self.backend_socket = context.socket(zmq.PUB)
        self.backend_socket.bind('tcp://%s:%d' % self.send_addr)
        self.backend_socket.set_hwm(10000)
        self.status = "Framegrabber is sending data to backend on ip %s port %d" % self.send_addr
    
    def _recvframe(self, buffer_size = 2*1152*1940):
        """receive frames from the FCCD. Return as soon as a new frame is ready, otherwise it is blocking."""
        #print "RECV: ", self.camera_socket.fileno(), self.fsize
        
        pkg = self.udpr.read_frame(self.camera_socket.fileno(), buffer_size)
        self.fbuffer, fnumber, self.fbytes = pkg
        self.nreceive += 1

        #to_image("img" + str(fnumber) + ".tif", self.fbuffer)

        if fnumber > self.fnumber1:
            logger.debug("Dropped Frame(s) range %d - %d" % (self.fnumber1, fnumber))
        self.fnumber1 = fnumber + 1
        
        if (self.nreceive == self.updaterate):
            t1 = time.time()
            self.status = "Reading at %.2fHz" %(self.nreceive/(t1-self.t0))
            #logger.debug("Dropped %d frames", (fnumber + 1 - self.fnumber0) - self.updaterate)
            self.fnumber0 = fnumber + 1
            self.t0 = t1
            self.nreceive = 0
        
        self.fnumber = fnumber
        
    def _sendframe(self):
        """Sending frames to the processing backend."""
        if self.fbuffer is not None:

            #logger.debug("sending out a frame, id = %d", self.fnumber)
            #self.backend_socket.send("rawframe", zmq.SNDMORE)
            #print(type(self.fbuffer))
            #self.backend_socket.send_multipart([str(self.fnumber), self.fbuffer])
            self.backend_socket.send_multipart([b'%d' % self.fnumber, self.fbuffer])
            #self.backend_socket.send(str(self.fnumber), zmq.SNDMORE)
            #self.backend_socket.send(msg)
            #logging.debug("storing frame, id = %d", self.fnumber)
           
            #print self.fnumber,self.fbuffer[:8]
       
            #self.circbuffer.append((self.fnumber,mframe))
            if self.extbuffer is not None:
                mframe = bytearray(len(self.fbuffer))
                mframe[:] = self.fbuffer
                self.extbuffer.append((self.fnumber,mframe))
                self.nrecord += 1
                if (self.nrecord == self.updaterate):
                    t1 = time.time()
                    self.status = "Recording at %.2fHz" %(self.nrecord/(t1-self.t1))
                    self.t1 = t1
                    self.nrecord = 0
    
    @property
    def lastframe(self):
        num,frame = self.circbuffer[-1]
        return num, np.frombuffer(frame,'<u2')
                
    def run(self):
        """This triggers the event loop."""
        self.status = "Framegrabber is starting the event loop"
        self.isRunning = True
        self.hasFinished = False
        self.isRecording = False
        print("running ")
        #self._recvframe()
        """
        for i in range(3):
            self._recvframe()
        """
        tsend = 0
        ii = 0
        while self.isRunning:
            ii +=1
            self._recvframe(self.fsize)
            t = time.time()
            self._sendframe()
            ts = time.time()-t
            tsend = 0.05*ts + 0.95*tsend
            if ii % 40 == 0: 
                self.status = "UDP load is %.2f ms" % (tsend * 1000) 
        print("has finished")
        
        self.hasFinished = True
        
    def stop(self):
        """Stop the event loop and close sockets."""
        self.status = "Framegrabber is stopping the event loop"
        self.isRunning = False
        #while not self.hasFinished:
        #    #print "Waiting for framegrabber to be finished ..."
        #    time.sleep(0.1)
        
        self.backend_socket.close()
        self.camera_socket.close()
        print("Framegrabber has finished")

        
if __name__=='__main__':
    logger.setLevel(logging.DEBUG)
    
    #FG=Framegrabber(2*1152*2000,read_addr= "127.0.0.1:49205",send_addr ="10.0.0.16:49206",udp_addr ="127.0.0.1:49203")
    #-----
    ## use this With CIN
    #FG=Framegrabber(2*1152*1040,read_addr= "10.0.5.55:49205",send_addr ="127.0.0.1:49206",udp_addr ="10.0.5.207:49203")
    #-----
    #FG=Framegrabber(2*1152*1040,read_addr= "10.0.5.55:49205",send_addr ="127.0.0.1:49206",udp_addr ="localhost:49203")
    #FG=Framegrabber(2*1152*1040,read_addr="127.0.0.1:49205",send_addr="127.0.0.1:49206",udp_addr ="127.0.0.1:49203")
    ## Simulation with test_stxmcontrol:
    FG=Framegrabber(2*1152*1040,read_addr="127.0.0.1:49205",send_addr="127.0.0.1:49207",udp_addr ="127.0.0.1:49209" )
    FG.createReadFrameSocket()
    FG.createSendFrameSocket()
    FG.daemon =True
    FG.start()
    while True:
        time.sleep(0.1)
