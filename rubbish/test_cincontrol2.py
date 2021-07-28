
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

term = {
'default' : lambda x : x, 
'red' : lambda x : '\x1b[0;31;40m %s \x1b[0m' % str(x),
'green' : lambda x : '\x1b[0;32;40m %s \x1b[0m' % str(x),
'yellow' : lambda x : '\x1b[0;33;40m %s \x1b[0m' % str(x),
'blue' : lambda x : '\x1b[0;34;40m %s \x1b[0m' % str(x),
}

import logging
logging.getLogger().setLevel(20)

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


class Scan:
    
    def __init__(self, shape = (1940,1152)):
        
        self.name = ''
        self.dark_dir = ''
        self.exp_dir = ''
        self.info = None
        self.dark_frames = []
        self.exp_frames = []

    def save_tifs(self):
        from cosmic.ext import tiffio
        from cosmic.camera.fccd import FCCD
        
        CCD = FCCD(nrows=970)
        def _to_tif(frame, path):
            num, buf = frame
            npbuf = np.frombuffer(buf,'<u2')
            im = CCD.assemble2(npbuf.reshape((12*970,192))).astype(np.uint16)
            name = '/img%06d.tif' % num
            tiffio.imsave(path+name,im)
            
        for frame in self.dark_frames:
            _to_tif(frame, self.dark_dir)
        
        for frame in self.exp_frames:
            _to_tif(frame, self.exp_dir)
            
class Viewer:
    
    from matplotlib import pyplot as plt
    
    def __init__(self, shape=(1940,1152)):
        #super(FccdSimulator,self).__init__()
        self.dark = np.zeros(shape).flatten()
        self.exp = np.ones(shape).flatten()
        
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(6,12))
        self.ax= self.fig.add_subplot(111)
        self.fccd = FCCD(nrows=970)
        self.art = self.ax.imshow(np.ones((1940,1152)))#self.assemble(self.exp-self.dark))

        self.plt.draw()
    
    def assemble(self,flatframe):
        return self.fccd.assemble2(flatframe.reshape((12*970,192)))
        
    def update(self):
        self.ax.images[0].set_data(self.assemble(self.exp-self.dark))
        self.plt.draw()
    
class Grabber:
    
    
    def __init__(self, shape = (1940,1152)):
        
        self.scans = []
        self.shape = shape
        self.scan = Scan()
        self.save_dir = ''
        self.dnum_max = None
        self.enum_max = None
        self.dnum = 0
        self.enum = 0
        
        self.QA = QtCore.QCoreApplication([])
        
        self.statusterm = open('/dev/pts/22','a')
        self.CC = cincontrol.CINController(cin_ip="127.0.0.1",stxm_port = 8880)
        self.FG = Framegrabber(49207, None, cin_ip ="127.0.0.1", cin_port = 49205, fsize = 2*1152*1940)
        self.IA = interaction.Server()
        #self.V = Viewer()
        #CC.createSendCommandsSocket()
        
        self.IA.register(self.scans,'scans')
        self.CC.statusMessage.connect(self.print_cin_status)
        self.FG.statusMessage.connect(self.print_fg_status)
        self.CC.finishedScan.connect(self.on_finished_scan)
        self.CC.newSavePath.connect(self.on_new_path)
        self.CC.gotNewInfo.connect(self.on_new_info)
        self.CC.startAcq.connect(self.on_start_capture)
        
    def prepare(self):
        self.CC.createReceiveCommandsSocket_qtcpserver()
        self.FG.createReadFrameSocket()

        
    def print_cin_status(self,status):
        self.statusterm.write(term['red'](status)+'\n')
        self.statusterm.flush()
        
    def print_fg_status(self,status):
        self.statusterm.write(term['green'](status)+'\n')
        self.statusterm.flush()
            
            
    def on_finished_scan(self):
        self.scans.append(self.scan)
        self.scan.completed = True
        #self.scan.save_tifs()
        self.scan._thread = Thread(target = self.scan.save_tifs)
        self.scan._thread.daemon = True
        self.scan._thread.start()
        self.statusterm.write(term['blue']("%d frames in buffer" % len(self.FG.allbuffer)) +'\n')
        self.scan = Scan(self.shape)
        self.scan.num_int = len(self.scans)
        

        
    def on_new_info(self, info):
        self.scan.info = str(info)
        
    def on_new_path(self,nnpath):
        npath = str(nnpath)
        scan = self.scan
        if str(scan.name)=='':
            scan.name = os.path.split(os.path.split(npath)[0])[1]
            self.statusterm.write(term['blue']('Name: %s\n' % scan.name))
        self.save_dir = npath
        
    def on_start_capture(self):
        scan = self.scan
        if self.CC.scan_is_dark:
            self.FG.curbuffer = scan.dark_frames
            scan.dark_dir =  self.save_dir
        else:
            scan.exp_dir =  self.save_dir
            self.FG.curbuffer = self.scan.exp_frames
            #self.dark = np.mean([np.frombuffer(frame,'<u2').astype(np.int) for num,frame in self.scan.dark_frames])
            
    def analyse_frame_count(self):
        self.dnum_max = None
        self.enum_max = None
        if self.scan.info is not None:
            try:
                d=dict([kv.strip().split() for kv in self.scan.info.split(',')])
                self.dnum_max = int(d['background_pixels_x'])*int(d['background_pixels_x'])
                self.enum_max = int(d['num_pixels_x'])*int(d['num_pixels_x'])
            except:
                pass
        return self.dnum_max, self.enum_max
              

    def get_status(self):
        dct = dict([(k,v) for k,v in self.scan.__dict__.items() if type(v) is str])
        self.dnum = len(self.scan.dark_frames)
        self.enum = len(self.scan.exp_frames)
        self.analyse_frame_count()
        dct['dark_num'] = str(self.dnum)+'/'+str(self.dnum_max)
        dct['exp_num'] = str(self.enum)+'/'+str(self.enum_max)
        infostring = ["%20s : %s\n" % (k,v) for k,v in dct.items()]
        return infostring
        
#if __name__ == '__main__':
    
def main(stdscr=None):
    win = stdscr
    G = Grabber()
    G.prepare()
    G.IA.activate()
    G.FG.start()
    G.CC.start()
    while True:
        # process pending events
        G.QA.processEvents()
        
        if win is not None:
            win.clear()
            for line in G.get_status():
                win.addstr(line)
            win.refresh()
        # handle client requests
        G.IA.process_requests()

#curses.wrapper(main)
main(None)
