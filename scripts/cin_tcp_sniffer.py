from cosmic.ext.ptypy.utils.validator import ArgParseParameter
import time
from PyQt4 import QtCore,QtNetwork
import curses
import os
from threading import Thread
from urllib.parse import splitport
from collections import deque, OrderedDict
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


from io import StringIO
DEFAULT = ArgParseParameter(name='Grabber')
DEFAULT.load_conf_parser(StringIO(
"""[tcp_addr]
default = "10.0.0.16:8890"
help = IP:Port address of command tcp server.

[statusterm]
default = "/dev/pts/2"
help = Second terminal pointer to display status updates.

[cin_gui_addr]
default = "127.0.0.1:8880"
help = IP:Port address of command tcp server.

[cin_recv_addr]
default = "127.0.0.1:8700"
help = IP:Port address of command tcp server.

"""))


class Grabber(object):
    
    ## Parameters

    
    def __init__(self, pars = None):
        
        default = DEFAULT.make_default(depth =10)
        pars = default if pars is None else pars
        self.pars = pars
        #self.scans = deque(maxlen=2)
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
        
        self.statusterm = open(pars['statusterm'],'a')
    
    def prepare(self):
        ip, port = splitport(self.pars['tcp_addr'])
        self.createReceiveCommandsSocket_qtcpserver(ip=ip,port=int(port))
        import socket
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        ip, port = splitport(self.pars['cin_recv_addr'])
        self.sock.bind((ip,int(port)))
        ip, port = splitport(self.pars['cin_gui_addr'])
        self.sock.connect((ip,int(port)))
        self.sock.settimeout(0.1)
        
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
            print(cmd)
            self.sock.settimeout(2)
            self.sock.sendall(cmd)
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

            result = self.sock.recv(2048)
            #result = self.response(resp_command)
            self._status = result
            #print "Responding with: ", result
            self.qtcpclient.write(result)
            self.qtcpclient.flush()
            #print "Here with: ", result
            self.sock.settimeout(0.1)
            
    def response(self, cmd):
        return(b'Command: ' + cmd + b'\n\r')
        
        
    def print_cin_status(self,status):
        self.statusterm.write(term['red'](status)+'\n')
        self.statusterm.flush()
        
    def print_fg_status(self,status):
        self.statusterm.write(term['green'](status)+'\n')
        self.statusterm.flush()
            
            
    def on_finished_scan(self):
        self.print_cin_status("Scan completed.")
        """
        #self.scan.save_tifs()
        try:
            self.statusterm.write(term['blue']("%d frames in buffer" % len(self.FG.circbuffer)) +'\n')
        except:
            pass
        """
        

        
    def on_new_info(self, info):
        # interpret info
        import ast
        
        self.status = "New info %s" %info
                
        din=dict([kv.strip().split() for kv in info.split(',')])
        dt = {}
        for k,v in din.items():
            try:
                vi = ast.literal_eval(v)
            except:
                vi = v
            
            dt[scaninfo_translation.get(k,k)] = vi
            
        d = OrderedDict([(k,dt[k]) for k in sorted(dt.keys())])
        try:
            Ny = d['exp_num_y']
            Nx = d['exp_num_x']
            d['dark_num_total'] = d['dark_num_x']*d['dark_num_y']
            d['exp_num_total'] = Nx * Ny
            dx, dy = d['exp_step_x'], d['exp_step_y']
            d['translations']= [(x*dx,y*dy) for x in range(Nx) for y in range(Ny)]
            
        except KeyError as e:
            warnings.warn('Translation extraction failed: (%s)' % e.message) # create postiion
            d['dark_num_total'] = None
            d['exp_num_total'] = None
            d['translations'] = None
            
        self.scan_info = d
        
    def on_new_path(self,nnpath):
        pass
        """
        npath = str(nnpath)
        scan = self.scan
        if str(scan.name)=='':
            scan.name = os.path.split(os.path.split(npath)[0])[1]
            self.statusterm.write(term['blue']('Name: %s\n' % scan.name))
        self.save_dir = npath
        """
        
    def on_start_capture(self, msg):
        pass
        """
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
        """
    
    def get_scan_status(self):
        pass
        """
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
        """
        
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
    while True:
        # process pending events
        G.QA.processEvents()
        try:
            print(G.sock.recv(1024))
        except:
            pass
        #print 'yeah'
        #time.sleep(0.05)
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
