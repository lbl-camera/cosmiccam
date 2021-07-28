from cosmic.ext.ptypy.utils.validator import ArgParseParameter
import time
from PyQt5 import QtCore,QtNetwork
#from cosmic.camera.framegrabber_nozmq import Framegrabber
#from cosmic.camera.fccd import FCCD
from cosmic.camera.cin import CIN
from cosmic.ext.ptypy.io import interaction
import numpy as np
import curses
import os
from threading import Thread
from urllib.parse import splitport
from collections import deque, OrderedDict
import tifffile as tiffio
from cosmic.camera.fccd import FCCD
from cosmic.utils import ScanInfo
import json
import gc
import zmq

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
    
    def __init__(self, shape=(1940,1152),num_dark=400,num_exp=60000, max_frames=30000):
        """
        """
        self.name = ''
        self.stack_idx= ''
        self.dark_dir = ''
        self.exp_dir = ''
        self.json_dir = ''
        self.info = None
        self.info_raw = ''
        self.max_frames_memory = max_frames
        self.dark_frames = deque(maxlen=1000)
        self.exp_frames = deque(maxlen=max_frames)
        self.dark_full = False
        self.dark_processed = 0
        self.dark_num = num_dark
        self.exp_full = False
        self.exp_processed = 0
        self.exp_num = num_exp
        self.num_rows = shape[0] /2 
        self.num_adcs = shape[1] /6
        self.shape = shape #(2 * num_rows, num_adcs * 6)
        self.fbuffer = None
        self._set_scheme('image%06d.tif')
        self.CCD = FCCD(nrows=self.num_rows) 
        #privat
        self._num_offset = 0
        self._thread = None
            
    def save_json(self,json_dir=None):
        jdir = json_dir if json_dir is not None else self.json_dir
        jdir = jdir if jdir else os.path.split(self.exp_dir)[0]
        if jdir:
            self.info['dark_dir'] = self.dark_dir
            self.info['exp_dir'] = self.exp_dir
            filename = jdir + '/%s_%s_info.json' % (self.name, self.stack_idx)
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
    
    def to_tif(self, frame, scheme, procnum=0):
            
        num, buf = frame
       
        #path = scheme % (procnum, num)
        path = scheme % procnum
        
        npbuf = np.frombuffer(buf,'<u2')
        npbuf = npbuf.reshape((12*self.num_rows,self.num_adcs))
        im = self.CCD.assemble2(npbuf.astype(np.uint16))
        
        tiffio.imsave(path,im)
        #tiffio.imsave(fastpath,im)
        
        return procnum, num
                
    
    def save_tifs(self):
        """
        Save frames as 16 bit TIFFS. 
        """
        path = self.dark_dir + self.scheme
       
        
        nummer =0 
        processed = []
        tries = 0
        for i in range(self.dark_num):
            while True:
                try:
                    d = self.dark_frames.popleft()
                    tries = 0
                    break
                except IndexError:
                    d = None
                    tries +=1
                    time.sleep(0.1)
                    if self.dark_full and tries > 3:
                        break
                    
            if self.dark_full and d is None:
                break
            else:
                nummer, ci = self.to_tif(d, path, i)
                self.dark_processed += 1
                processed.append('%06d %05d\n' % (nummer, ci))
                
                #print self.dark_processed
                
        ftext = open(os.path.split(path)[0] + '/allframes.txt','w')
        ftext.writelines(processed)
        ftext.close()
        
        path = self.exp_dir  + self.scheme
        processed = []
        tries = 0
        for i in range(self.exp_num):
            while True:
                try:
                    d = self.exp_frames.popleft()
                    tries = 0
                    break
                except IndexError:
                    d = None
                    tries +=1
                    time.sleep(0.1)
                    if self.exp_full and tries > 5:
                        break

                    
            if self.exp_full and d is None:
                break
            else:
                nummer, ci = self.to_tif(d, path, i)
                self.exp_processed += 1
                processed.append('%06d %05d\n' % (nummer, ci))
                
        ftext = open(os.path.split(path)[0] + '/allframes.txt', 'w')
        ftext.writelines(processed)
        ftext.close()
    
    def auto_save_tifs(self):
        if not self._thread:
            self._thread = Thread(target = self.save_tifs)
            self._thread.daemon = True
            self._thread.start()
        
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
        

from io import StringIO
DEFAULT = ArgParseParameter(name='Grabber')
DEFAULT.load_conf_parser(StringIO(
"""[shape]
default = (1040,1152)
help = (H,W) of the detector in pixel including overscan pixels

[tcp_addr]
default = "131.243.73.45:8880"
help = IP:Port address of command tcp server.

[statusterm]
default = "/dev/pts/1"
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

[pub_addr]
default = "127.0.0.1:49206"
help = ip:port the framegrabber publishe frames. [NOT IMPLEMENTED]

[read_addr]
default = "10.0.0.16:49207"
help = ip:port type address the grabber contacts the udp server from.

[timeout]
default = 50
help = Blocking timeout on receive in ms. If set to zero, the framegrabber
  will receive and discard an additional frame. Otherwise chose a timeout
  that is smaller than the waiting time in between scans to avoid overlap.
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
        self._thread = None
        self.scan_is_dark = True
        self.scan_stopped = True
        self.scan_info = ''
        self.scan_path = ''
        self.QA = QtCore.QCoreApplication([])
        self.SI = ScanInfo()
        self.cin = CIN()
        
        try:
            self.statusterm = open(pars['statusterm'],'a')
        except IOError as e:
            print(e)
            self.statusterm = None
        
        fsize = self.shape[0] * self.shape[1] * 2
        
        p = pars['iaserver']
        ip,port = splitport(p['ia_addr'])
        #self.IA = interaction.Server(address = ip, port =int(port))
        #self.IA.register(self.scans,'scans')
        #self.IA.register(self.FG,'grabber')
        
    def prepare(self):
        ip, port = splitport(self.pars['tcp_addr'])
        self.createReceiveCommandsSocket_qtcpserver(ip=ip,port=int(port))
        
        #context = zmq.Context()
        p = self.pars['framegrabber']
        #self.frame_socket = context.socket(zmq.SUB)
        #self.frame_socket.setsockopt(zmq.SUBSCRIBE, '')
        #self.frame_socket.set_hwm(2000)
        #self.frame_socket.bind('tcp://%s' % p['read_addr'])
        
    def zmq_receive(self,pub_address=None):
        addr = 'tcp://%s' % self.pars['framegrabber']['pub_addr']
        timeout = int(self.pars['framegrabber']['timeout']) 
        context = zmq.Context()
        frame_socket = context.socket(zmq.SUB)
        frame_socket.setsockopt(zmq.SUBSCRIBE, '')
        frame_socket.set_hwm(2000)
        frame_socket.connect(addr)
        self.print_status("Starting zmq reading Thread",'blue')
        if timeout == 0:
            while not self.scan_stopped:
                number, frame = frame_socket.recv_multipart() # blocking
                # Could have been stopped in the meantime, so we need to
                # discard this one
                if not self.scan_stopped:
                    self.scan.fbuffer.append((int(number), frame))
        else:
            # This could probablu have been done cleaner with a Poller
            # But I rather have the GIL released with sleep instead
            # of polling all the time.
            slp = timeout / 1000.
            while not self.scan_stopped:
                try:
                    number, frame = frame_socket.recv_multipart(flags=zmq.NOBLOCK) # blocking
                except zmq.ZMQError:
                    time.sleep(slp)
                    continue
                self.scan.fbuffer.append((int(number), frame))
        self.print_status("Stopped reading frames",'blue')
        frame_socket.disconnect(addr)
        
        #self.frame_socket.close()
        
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
            cmd = str(cmd) #str(QtCore.QString(cmd))
            self.print_status('Getting: ' + cmd[:-1],'green')
            if len(cmd) == 0:
                return self.qtcpclient.send(self.response(""))

            #sub command
            #cmd = str(QtCore.QString(cmd))
            resp_command = cmd
            if "stopCapture" in resp_command:
                
                self.on_finished_scan()
                                
            elif "startCapture" in resp_command:
                """
                if self.scan_stopped:
                    self.scan_stopped = False
                    self.scan_is_dark = True
                else:
                    self.scan_is_dark = False
                """
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
                self.print_status("New scan :\n %s" % '\n'.join(self.scan_info.split(",")))
                self.on_new_info(self.scan_info)
                
            elif "sendScanJSON" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                values = cmd[index+1:].strip()

                self.scan_json = values.strip()
                self.print_status("New scan :\n %s\n" % self.scan_json )
                self.on_new_info(self.scan_json , is_json=True)
                
            elif "setExp2" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = float(cmd[index+1:].strip()) + 3
                self.print_status("Setting alternate exposure to %.2f.\n This often corrsponds to a shutter time of %.2f\n" % (exp,exp-10.))
                # add artifical delay
                #exp = min(exp,10)
                self.cin.setAltExpTime(exp)
                
            elif "setExp" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = float(cmd[index+1:].strip()) + 3
                self.print_status("Setting exposure to %.2f.\n" % exp)
                #exp += min(exp,10)
                self.cin.setExpTime(exp) 

            elif "setDoubleExpCount" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = int(cmd[index+1:].strip())
                if exp==0:
                    self.print_status("Setting to single exposure mode.")
                    self.cin.set_register("8050","0000",1)
                elif exp in range(1,8):
                    self.print_status("Setting to double exposure mode (type %d)." % exp)
                    self.cin.set_register("8050","8%d00" % (exp-1),1)
                else:
                    self.print_status("Double exposure mode %d ignored." % exp)

            elif "resetCounter" in resp_command:
                self.cin.set_register("8001", "0106", 0)
                #time.sleep(0.002)
                #0x8001, 0x0106
                
            result = self.response(resp_command)
            #print "Responding with: ", result
            self.qtcpclient.write(result)
            self.qtcpclient.flush()
            #print "Here with: ", result

    
    def on_new_info(self, info_raw, is_json=False):
        # This is needed for batched scans
        self.scan.exp_full = True
        
        if is_json:
            info = json.loads(info_raw)
            self.status = "New json" #% info
        else:
            self.status = "New info %s" % info_raw
            info = self.SI.read_tcp(info_raw)
        
        # this could maybe be part of read_tcp
        fac = info.get('repetition',1) 
        fac *= info.get('isDoubleExp',0) + 1 
        num_dark = info.get('dark_num_total',0) * fac 
        num_exp = info.get('exp_num_total',0) * fac 
        
        scan = Scan(self.shape, num_dark, num_exp)
        #scan.info_raw = info_raw  # uncomment to avoid display
        scan.info = info
        self.scan = scan
        
        gc.collect()
        
        
        if self.scan_stopped:
            self.scan_stopped = False
            self._thread = Thread(target = self.zmq_receive)
            self._thread.daemon = True
            self._thread.start()
            
    def on_new_path(self,nnpath):
           
        npath = str(nnpath)
        scan = self.scan
        trunk, scan.stack_idx = os.path.split(npath) 
        if str(scan.name)=='':
            scan.name = os.path.split(trunk)[1]
            self.print_status('Name: %s' % scan.name)
        self.save_dir = npath
        
    def on_start_capture(self, msg):
        scan = self.scan
        if self.scan_is_dark:
            self.print_status("Starting dark capture")
            scan.fbuffer = scan.dark_frames
            scan.dark_dir =  self.save_dir
            self.scan_is_dark= False
        else:
            self.print_status("Starting exp capture")
            scan.dark_full = True
            scan.exp_dir =  self.save_dir
            scan.save_json()
            scan.fbuffer = scan.exp_frames
            self.scan_is_dark= True
            self.scan.auto_save_tifs() # moved up

    def response(self, cmd):
        return(b'Command: ' + cmd + b'\n\r')
        
    def print_status(self,status,color='yellow'):
        msg = term[color](status)+'\n'
        if self.statusterm is not None:
            self.statusterm.write(msg)
            self.statusterm.flush()
        else:
            print(msg)
            
    def print_cin_status(self,status):
        self.print_status(status,'red')
        #self.statusterm.write(term['red'](status)+'\n')
        #self.statusterm.flush()
        
    def print_fg_status(self,status):
        self.print_status(status,'green')
        #self.statusterm.write(term['green'](status)+'\n')
        #self.statusterm.flush()
            
    def on_finished_scan(self):
        self.scan_stopped = True
        self.scan.exp_full = True
        self.print_cin_status("Scan completed.")
        #self.scan.save_tifs()
        try:
            self.print_status("%d frames in buffer" % len(self.scan.fbuffer), 'blue')
        except:
            pass
    
    def get_scan_status(self):
        scan = self.scan
        dct = dict([(k,v) for k,v in self.scan.__dict__.items() if type(v) is str])
        #self.dnum = len(scan.dark_frames)
        #self.enum = scan.exp_processed
        #dmax = str(scan.info.get('dark_num_total',0)) if scan.info else 'unknown'
        #emax = str(scan.info.get('exp_num_total',0)) if scan.info else 'unknown'
        emax = str(scan.exp_num)
        dmax = str(scan.dark_num)
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
    #G.FG.start()
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

#curses.wrapper(main)
main(None)

"""
            elif "resetCounter" in resp_command:
                self.cin.set_register("8001", "0106", 0)
                time.sleep(0.002)
                #0x8001, 0x0106
"""
