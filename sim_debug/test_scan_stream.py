import numpy as np
import os
from threading import Thread
from urllib2 import splitport
from collections import deque, OrderedDict
import zmq

class Frames(object):
    
    def __init__(self,mx=400):
        
        self.full = False
        self.processed = 0
        self.max = mx
        self.dir = ''
        self.frames = deque(maxlen=mx)
    
    def iterpop(self):
        nummer =0
        for i in range(self.max):
            while True:
                try:
                    d = self.frames.popleft()
                    break
                except IndexError:
                    d = None
                    if self.full:
                        break
                    else:
                        time.sleep(0.1)
                    
            if self.full and d is None:
                break
            else:
                self.processed += 1
                yield nummer, d

    def fill(self, bts = 100000):
        
        for num in range(self.mx):
            self.frames.append((num % (2**16), bytearray(bts)))
        
    def set_dir(self, dir=None, mkdir=False):
    
        # sanity check
        _c = lambda x: x is not None and str(x)!='' and str(x)==x 
            
        if _c(dir): self.dir = dir
        
        if mkdir:
            if self.dir: os.makedirs(self.dir)
        
class RawScan(object):
    """
    This class saves all information available at the beamline.
    
    It has its own methods to
    save frames and info to disk or potentially transmit its content to
    another socket (NOT IMPLEMENTED).
    
    Frames are stored as stored as Python buffers in the attributes 
    ``.dark_frames`` and ``.exp_frames`` which are lists for thread
    safety reasons.
    """
    def __init__(self,max_dark=400,max_exp=30000):
        self.name = ''
        self.stack_idx= ''
        self.info = None
        self.info_raw = ''
        self.dark = Frames(max_dark)
        self.exp = Frames(max_exp)
        self.scheme = 'image%06d.raw'
        self.offset = 0
        
    def save_info(self,json_dir=None):
        jdir = json_dir if json_dir is not None else self.json_dir
        jdir = jdir if jdir else os.path.split(self.exp_dir)[0]
        if jdir:
            import json
            self.info['dark_dir'] = self.dark_dir
            self.info['exp_dir'] = self.exp_dir
            filename = jdir + '/%s_%s_info.json' % (self.name, self.stack_idx)
            f = open(filename,'w')
            json.dump(self.info,f)
            f.close()
            return filename   
            
    def save(self):
        """
        Save frames as 16 bit TIFFS. 
        """
        path = self.dark.dir + self.scheme
        for num, d in self.dark.iterpop():
            nummer = self._to_disk(d, 0, path)
        
        path = self.exp.dir + self.scheme
        for num, d in self.exp.iterpop():
            nummer = self._to_disk(d, nummer, path)
    
    @publicmethod
    def to_disk(frame, lastnum, scheme):
        
        num, buf = frame        
        # wrap around
        if num < lastnum:
            if num+self.offset < lastnum
                self.offset += 2**16
            num += self.offset
        
        path = scheme % num
        
        with open(path,'w') as f:
            f.write(buf)
            f.close()
        
        return num
        
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
        
class Scan(RawScan):
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
        self.stack_idx= ''
        self.dark_dir = ''
        self.dark_dir_ram = '/dev/shm/dark'
        self.exp_dir = ''
        self.exp_dir_ram = '/dev/shm/exp'
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
        self.fbuffer = None
        self._set_scheme('image%06d.tif')
        self.CCD = FCCD(nrows=self.num_rows) 
        #privat
        self._num_offset = 0
        for dr in [self.dark_dir_ram,self.exp_dir_ram]:
            if not os.path.exists(dr):
                os.makedirs(dr)
            else:
                for f in os.listdir(dr):
                    os.remove(dr + os.path.sep + f)
            
    def save_json(self,json_dir=None):
        jdir = json_dir if json_dir is not None else self.json_dir
        jdir = jdir if jdir else os.path.split(self.exp_dir)[0]
        if jdir:
            import json
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
    
    def to_tif(self, frame, lastnum, scheme, fastscheme=None):
            
        nr = self.num_rows
        num, buf = frame
        
        # wrap around
        if num < lastnum:
            num = lastnum+1
        
        path = scheme % num
        fastpath = fastscheme % num if fastscheme is not None else None
        """
        with open(path+name,'w') as f:
            f.write(buf)
            f.close()
        """
        npbuf = np.frombuffer(buf,'<u2')
        npbuf = npbuf.reshape((12*nr,self.num_adcs)).astype(np.uint16)
        im = self.CCD.assemble2(npbuf)
        
        tiffio.imsave(path,im)
        #tiffio.imsave(fastpath,im)
        
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

            

        
