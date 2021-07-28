from cosmic.ext.ptypy.utils.validator import ArgParseParameter
from cosmic.camera import cincontrol
import time
from PyQt4 import QtCore,QtNetwork
from cosmic.camera.framegrabber_nozmq import Framegrabber
from cosmic.camera.fccd import FCCD
from cosmic.ext.ptypy.io import interaction
import numpy as np
import curses
import os
from threading import Thread
from urllib2 import splitport
from collections import deque, OrderedDict
from cosmic.ext import tiffio
from cosmic.camera.fccd import FCCD
from cosmic.utils import ScanInfo

import gc

term = {
'default' : lambda x : x, 
'red' : lambda x : '\x1b[0;31;40m %s \x1b[0m' % str(x),
'green' : lambda x : '\x1b[0;32;40m %s \x1b[0m' % str(x),
'yellow' : lambda x : '\x1b[0;33;40m %s \x1b[0m' % str(x),
'blue' : lambda x : '\x1b[0;34;40m %s \x1b[0m' % str(x),
}

scaninfo_translation = {
'num_pixels_x' : 'exp_num_x',
'num_pixels_y' : 'exp_num_y',
'step_size_x' : 'exp_step_x',
'step_size_y' : 'exp_step_y',
'background_pixels_x' : 'dark_num_x',
'background_pixels_y' : 'dark_num_y',
}

import logging, warnings
logging.getLogger().setLevel(20)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

global lastbuf

class Scan(object):
    """
    This class saves all information available at the beamline.
    
    It has its own methods to
    save frames and info to disk or potentially transmit its content to
    another socket (NOT IMPLEMENTED).
    
    Frames are stored as stored as Python buffers in the attributes 
    ``.dark_frames`` and ``.exp_frames`` which are lists for thread
    safety reasons.
    """
    def __init__(self, shape=(1940,1152),max_dark=400,max_exp=30000):
        """
        """
        self.name = ''
        self.dark_dir = ''
        self.exp_dir = ''
        self.json_dir = ''
        self.info = None
        self.info_raw = ''
        self.dark_frames = deque(maxlen=max_dark)
        self.exp_frames = deque(maxlen=max_exp)
        self.dark_full = False
        self.dark_processed = 0
        self.dark_max = max_dark
        self.exp_full = False
        self.exp_processed = 0
        self.exp_max = max_exp
        self.num_rows = shape[0] /2 
        self.num_adcs = shape[1] /6
        self.shape = shape #(2 * num_rows, num_adcs * 6)
        
        self._set_scheme('image%06d.tif')
        self.CCD = FCCD(nrows=self.num_rows)
        #privat
        self._num_offset = 0
        
    def save_json(self,json_dir=None):
        jdir = json_dir if json_dir is not None else self.json_dir
        jdir = jdir if jdir else os.path.split(self.exp_dir)[0]
        if jdir:
            import json
            self.info['dark_dir'] = self.dark_dir
            self.info['exp_dir'] = self.exp_dir
            filename = jdir + '/%s_info.json' % self.name
            f = open(filename,'w')
            json.dump(self.info,f)
            f.close()
            return filename        

    def _set_scheme(self, scheme):
    
        self.scheme = scheme if scheme.startswith(os.path.sep) else os.path.sep + scheme
    
    def _set_dirs(self,dark_dir=None,
                       exp_dir=None,
                       create_dirs=False):
        
        # sanity check
        _c = lambda x: x is not None and str(x)!='' and str(x)==x 
            
        if _c(dark_dir): self.dark_dir = dark_dir
        if _c(exp_dir): self.exp_dir = exp_dir
        
        if create_dirs:
            if self.dark_dir: os.makedirs(self.dark_dir)
            if self.exp_dir: os.makedirs(self.exp_dir)
    
    def to_tif(self, frame, lastnum, scheme):
            
        nr = self.num_rows
        num, buf = frame
        
        # wrap around
        if num < lastnum:
            num = lastnum+1
        
        path = scheme % num
        """
        with open(path+name,'w') as f:
            f.write(buf)
            f.close()
        """
        npbuf = np.frombuffer(buf,'<u2')
        npbuf = npbuf.reshape((12*nr,self.num_adcs)).astype(np.uint16)
        im = self.CCD.assemble2(npbuf)
        
        tiffio.imsave(path,im)
        
        return num
                
    
    def save_tifs(self):
        """
        Save frames as 16 bit TIFFS. 
        """
        path = self.dark_dir + self.scheme
        
        nummer =0 
        for i in range(self.dark_max):
            while True:
                try:
                    d = self.dark_frames.popleft()
                    break
                except IndexError:
                    d = None
                    if self.dark_full:
                        break
                    else:
                        time.sleep(0.1)
                    
            if self.dark_full and d is None:
                break
            else:
                nummer = self.to_tif(d, 0, path)
                self.dark_processed += 1
                #print self.dark_processed
                
            """
            while True:
                has_frames = len(darks) > i
                if has_frames or self.dark_full:
                    break
                else:
                    time.sleep(0.1)
            
            if self.dark_full and not has_frames:
                break
            else:
                nummer = self.to_tif(darks[i], 0, path)
            """
        path = self.exp_dir  + self.scheme

        for i in range(self.exp_max):
            while True:
                try:
                    d = self.exp_frames.popleft()
                    break
                except IndexError:
                    d = None
                    if self.exp_full:
                        break
                    else:
                        time.sleep(0.1)
                    
            if self.exp_full and d is None:
                break
            else:
                nummer = self.to_tif(d, nummer, path)
                self.exp_processed += 1
            """
                has_frames = len(exps) > i
                if has_frames or self.exp_full:
                    break
                else:
                    time.sleep(0.1)
            
            if self.exp_full and not has_frames:
                break
            else:
                nummer = _to_tif(exps[i], nummer)
            """
            
    def send(self, ip = 'localhost', port = 5555, protocol='zmq_pub'):
        """
        Send entire scan to address `ip:port` using protocol `protocol`
        """
        return NotImplementedError
            
    def stream_send(self, ip = 'localhost', port = 5555, protocol='zmq_pub'):
        """
        Stream scan on a per-frame basis to address `ip:port` using protocol `protocol`.
        """
        return NotImplementedError
    
    def receive(self, ip = 'localhost', port = 5555, protocol='zmq_sub'):
        """
        Send entire scan to address `ip:port` using protocol `protocol`
        """
        return NotImplementedError
            
    def stream_recv(self, ip = 'localhost', port = 5555, protocol='zmq_sub'):
        """
        Send entire scan to address `ip:port` using protocol `protocol`.
        """
        return NotImplementedError
        

from StringIO import StringIO
DEFAULT = ArgParseParameter(name='Grabber')
DEFAULT.load_conf_parser(StringIO(
"""[shape]
default = (2000,1152)
help = (H,W) of the detector in pixel including overscan pixels

[tcp_addr]
default = "10.0.0.16:8880"
help = IP:Port address of command tcp server.

[statusterm]
default = "/dev/pts/2"
help = Second terminal pointer to display status updates.

[iaserver]
default = None
help = Interaction server parameters

[framegrabber]
default = None
help = Framegrabber init parameters
"""))

#DEFAULT_framegrabber = ArgParseParameter(name='framegrabber')
DEFAULT.children['framegrabber'].load_conf_parser(StringIO(
"""[udp_addr]
default = "10.0.5.207:49203"
help = ip:port type address of the udp frame server

[send_addr]
default = None
help = ip:port the framegrabber publishe frames. [NOT IMPLEMENTED]

[read_addr]
default = "10.0.5.55:49205"
help = ip:port type address the grabber contacts the udp server from.
"""))

DEFAULT.children['iaserver'].load_conf_parser(StringIO(
"""[ia_addr] 
default = "tcp://10.0.2.56:5560"
help = Default address for primary connection
"""
))
    

class Grabber(object):
    
    ## Parameters

    
    def __init__(self, pars = None):
        
        default = DEFAULT.make_default(depth =10)
        pars = default if pars is None else pars
        self.pars = pars
        #self.scans = deque(maxlen=2)
        self.shape = pars['shape']
        self.scan = Scan(self.shape)
        self.save_dir = ''
        self.dnum_max = None
        self.enum_max = None
        self.dnum = 0
        self.enum = 0
        self._status = ''
        self.scan_is_dark = False
        self.scan_stopped = True
        self.scan_info = ''
        self.scan_path = ''
        self.QA = QtCore.QCoreApplication([])
        self.SI = ScanInfo
        
        self.statusterm = open(pars['statusterm'],'a')
        
        p = pars['framegrabber']
        fsize = self.shape[0] * self.shape[1] * 2
        self.FG = Framegrabber(fsize,**p)
        self.FG.statusMessage.connect(self.print_fg_status)
        
        p = pars['iaserver']
        ip,port = splitport(p['ia_addr'])
        #self.IA = interaction.Server(address = ip, port =int(port))
        #self.IA.register(self.scans,'scans')
        #self.IA.register(self.FG,'grabber')
        
    def prepare(self):
        ip, port = splitport(self.pars['tcp_addr'])
        self.createReceiveCommandsSocket_qtcpserver(ip=ip,port=int(port))
        self.FG.createReadFrameSocket()

        
    def createReceiveCommandsSocket_qtcpserver(self, ip="127.0.0.1",port = 8880):
        # A socket for receiving commands from the frontend using QTCP server
        self.qtcpserver = QtNetwork.QTcpServer()
        self.qtcpserver.newConnection.connect(self.acceptRemoteConnection)
        hostAddress = QtNetwork.QHostAddress(ip)
        self.qtcpserver.listen(hostAddress, port)
        self.status = "CINController is listening to commands on %d, using QTCPServer" % port

    def acceptRemoteConnection(self):
        self.qtcpclient = self.qtcpserver.nextPendingConnection()
        self.qtcpclient.readyRead.connect(self.answer_command_qtcp)
        self.status = "Accepted connection : %s:%d" %(self.qtcpclient.peerAddress().toString(), self.qtcpclient.peerPort())

    def answer_command_qtcp(self):
        newline = "\n\r"
        while self.qtcpclient.canReadLine():
            cmd = self.qtcpclient.readLine()
            cmd = str(QtCore.QString(cmd))
            if len(cmd) == 0:
                return self.qtcpclient.send(self.response(""))

            #sub command
            cmd = str(QtCore.QString(cmd))
            resp_command = cmd

            if "stopCapture" in resp_command:
                
                self.scan_stopped = True
                self.on_finished_scan()
                
            elif "startCapture" in resp_command:
                
                if self.scan_stopped:
                    self.scan_stopped = False
                    self.scan_is_dark = True
                else:
                    self.scan_is_dark = False
                    
                msg = 'dark' if self.scan_is_dark else 'exp'
                self.on_start_capture(msg)
                self.status = "Capturing '%s'." % msg
            
            elif "setCapturePath" in resp_command:
                
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                newpath = cmd[index+1:].strip()
                if len(newpath) > 0 and newpath != self.scan_path:
                    self.status = "Save output to: " + newpath
                    self.scan_path = newpath
                    #self.newParam_str.emit('cxiwrite', 'saveDir', values.strip())
                    #self.startNewRun.emit()
                    self.on_new_path(newpath)
            
            elif "sendScanInfo" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                values = cmd[index+1:].strip()

                self.scan_info = values.strip()
                self.status = "New scan :\n %s" % '\n'.join(self.scan_info.split(","))
                self.on_new_info(self.scan_info)

          
            result = self.response(resp_command)
            #print "Responding with: ", result
            self.qtcpclient.write(result)
            self.qtcpclient.flush()
            #print "Here with: ", result
            
    def response(self, cmd):
        return(b'Command: ' + cmd + b'\n\r')
        
        
    def print_cin_status(self,status):
        self.statusterm.write(term['red'](status)+'\n')
        self.statusterm.flush()
        
    def print_fg_status(self,status):
        self.statusterm.write(term['green'](status)+'\n')
        self.statusterm.flush()
            
            
    def on_finished_scan(self):
        self.FG.extbuffer = None
        self.scan.exp_full = True
        self.print_cin_status("Scan completed.")
        #self.scan.save_tifs()
        try:
            self.statusterm.write(term['blue']("%d frames in buffer" % len(self.FG.circbuffer)) +'\n')
        except:
            pass
        
        

        
    def on_new_info(self, info):
       
        self.status = "New info %s" %info
        self.scan = Scan(shape=self.shape)
        gc.collect()
        
        self.scan.info_raw = info
        d = self.SI.read_raw(info)
        self.scan.info = d
        
    def on_new_path(self,nnpath):
        npath = str(nnpath)
        scan = self.scan
        if str(scan.name)=='':
            scan.name = os.path.split(os.path.split(npath)[0])[1]
            self.statusterm.write(term['blue']('Name: %s\n' % scan.name))
        self.save_dir = npath
        
    def on_start_capture(self, msg):
        scan = self.scan
        if self.scan_is_dark:
            self.FG.extbuffer = scan.dark_frames
            scan.dark_dir =  self.save_dir
        else:
            scan.dark_full = True
            scan.exp_dir =  self.save_dir
            scan.save_json()
            self.FG.extbuffer = scan.exp_frames
        
        self.scan._thread = Thread(target = self.scan.save_tifs)
        self.scan._thread.daemon = True
        self.scan._thread.start()
    
    
    def get_scan_status(self):
        scan = self.scan
        dct = dict([(k,v) for k,v in self.scan.__dict__.items() if type(v) is str])
        #self.dnum = len(scan.dark_frames)
        #self.enum = scan.exp_processed
        dmax = str(scan.info.get('dark_num_total',0)) if scan.info else 'unknown'
        emax = str(scan.info.get('exp_num_total',0)) if scan.info else 'unknown'
        dct['dark_num'] = str(scan.dark_processed)+'/'+ dmax
        dct['exp_num'] = str(scan.exp_processed)+'/'+ emax
        infostring = ["%20s : %s\n" % (k,dct[k]) for k in sorted(dct.keys())]
        return infostring
        
    @property
    def status(self):
        return self._status
            
    @status.setter
    def status(self, stat):
        self._status = stat
        self.print_cin_status(stat)
        
#if __name__ == '__main__':
    

def main(stdscr):
    win = stdscr
    G = Grabber()
    G.prepare()
    #G.IA.activate()
    G.FG.start()
    while True:
        # process pending events
        G.QA.processEvents()
        time.sleep(0.1)
        if win is not None:
            win.clear()
            for line in G.get_scan_status():
                win.addstr(line)
            win.refresh()
        # handle client requests
        #G.IA.process_requests()

#parser = DEFAULT.add2argparser()
DEFAULT.parse_args()

curses.wrapper(main)
#main(None)
