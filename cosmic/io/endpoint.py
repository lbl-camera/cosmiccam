import numpy as np

has_comm = True
  
"""
This file describes the interface to a "sharp streaming socket" 
for ptychographic reconstructions.

v 0.0.0 - initial draft


CONVENTIONS

When updating dicts it is considered equivalent whether the key to an 
entry is None  or if the key does not exist in the dictionary, 
although the latter case should be avoided.
"""

# Setup defaults
DEFAULT_setup = {
    
    ## General information the reflects the computing architecture
    ## All information to activate sharp on a remote server should be
    ## be put here. 
    
    ### STRINGS ###
    'address' : 'n0001:5010',
    # The sharp endpoint socket address
    
    'save_dir' : '/global/groups/cosmic/Data/streamed/', # OPTIONAL
    # Save directory
    
    ## SCALARS
    'num_proc_ids' : 1,
    # Number of workers that will push data to client.
}

DEFAULT_prepare = {
    
    ## A priori information available before the scan even started.
    ### STRINGS ###
    'run_file' : 'tmp.cxi', # OPTIONAL
    # File name to save the reconstruction into
           
    'energy' : 800 * 1.602e-19, # MANDATORY
    # Energy (in Joule) of the diffraction experiment.
    
    'distance' : 0.083, # MANDATORY
    # object - detector distance in meters

    ### TUPLES ###
    # These may also be scalar, if square diffraction array and pixels 
    # are the standard in sharp.
    
    'pixelsize': (90e-6,90e-6), # MANDATORY
    # Dimension of the detector pixel (dx,dy)
     
    'shape' : (256,256), # MANDATORY
    # Shape, diffraction frame array shape (Nx,Ny) and probe array
    
    'field' : (1000,1000), # MANDATORY
    # Shape of the object array size/shape 
    
    ### SCALARS ###
    'num_frames' : 800, # MANDATORY

    ### OTHER ###
    'dhalf' : 1, # MANDATORY
    # Total number of frames considered in the recon. If None, this value may be 
    # inferred from the length of `positions` is provided
}

DEFAULT_update = {

    ## A priori information available at runtime, shortly before 
    ## the first dataset moves in. Reacting to this data may be time-critical.
    
    ### Arrays ###
    
    'positions' : None,  # RECOMMENDED
    # (N,2)-array (float) of theoretical positions / translations (in meter)
    #
    # These may be used to reshape memory in the sharp hoster
    # and as a backup in case the meta data of the frame does not
    # contain shift positions. 
        
    'probe_init' : None, # RECOMMENDED (shape,)-array (complex)
    # initial probe guess, should be in the shape of `shape` and complex
        
    'probe_fmask' : None, # RECOMMENDED (shape,)-array (bool or uint8)
    # binary mask of valid probe pixels in frequency space

    'probe_mask' : None, # OPTIONAL (shape,)-array (bool or uint8)
    # binary mask of valid probe pixels in real space
    
    'detector_mask' : None, # OPTIONAL
    # binaray mask of valid pixels in the detector
    
}

# Setup defaults
PAYLOADS = {
    
    'clean' : {
        
        ## Clean data payload.
    
        'num' : 0, # scalar (int), MANDATORY
        # Number of the frame in the scan, will be used to determine the
        # shift from the setup in case the `position` meta info is not set.
        
        'data' : None, # (shape,)-array (float or int), MANDATORY
        # Actual diffraction data. The numeric type is not yet set.
        # I think it should be float32.
        
        'position' : None, #  2-tuple (float), RECOMMENDED
        # Translational shift of the ptycho scan point.
        
        'mask' : None, # (shape,)-array (bool), OPTIONAL
        # A mask of valid pixels of for the data array. I don't know if that
        # is actually need or if we just encode negative values in the data
        # array as "bad-pixel".
    
        'process_id' : None, # scalar (int), OPTIONAL
        # The id of the preprocessor, may just be the rank of an mpi call
        # Must be smaller than `num_proc_ids`.
        
        }, 
        
}

import zmq

class MyComm:
    def __init__(self, address):
        ctx = zmq.Context()
        self.zmq_sock = ctx.socket(zmq.PUSH)
        self.zmq_sock.connect("tcp://" + str(address))
        self.zmq_sock.set_hwm(2000)
        self.zmq_sock.setsockopt(zmq.SNDTIMEO,500)
        self.active = True
        
    def send(self, obj):
        if self.active:
            try:
                #print "ZMQ PUSH to endpoint."
                self.zmq_sock.send_pyobj(obj)
            except zmq.ZMQError as e:
                print("ZMQ PUSH endpoint not responsive. Deactivating...")
                self.active = False
        else:
            pass
            #print "PUSH endpoint inactive."
                
class SharpClient(object):
    
    # This is mainly for documentation
    SETUP = DEFAULT_setup
    PREPARE = DEFAULT_prepare
    UPDATE = DEFAULT_update
    PAYLOADS = PAYLOADS
    
    def __init__(self):
        self._setup = DEFAULT_setup.copy() 
        self._prepare = DEFAULT_prepare.copy() 
        #if has_comm:
        #    comm_interface.global_comm_task.setup("cosmic", wait=False)
        #    self.comm = comm_interface.global_comm_task
        self.comm = MyComm(self._setup['address'])
        
    def _load_config(self, config_file):
        """
        Loads config file and updates self.config
        """
        return {}
    
    @staticmethod
    def _load_cfg(dct,cfile,kw):
        
        if cfile is not None:
            dct.update(self._load_config(cfile))
        
        for k, v in kw.items():

            if k in dct.keys():
                dct[k] = v
            else:
                print("Parameter %s not understood. Discarded" % k)
        
    def setup(self, config_file = None, **kwargs):
        """
        Attempts to configure from config file if not None.
        """
        self._load_cfg(self._setup, config_file, kwargs)
                    
        print("Will setup SHARP streaming solver with the following parameters:\n")
        for key, val in self._setup.items():
            print("%20s : %s" % (key,str(val)))
        

        """
        COMMUNICATION MAGIC
        """
        """
        ERROR MESSAGE MAYBE
        """
        if has_comm:
            command = dict()
            command["command"] = "setup"
            command["data"] = self._setup 
            self.comm.send(command)

    def prepare(self, config_file = None, **kwargs):
        """
        Attempts to configure from config file if not None.
        Signals the beginning of a scan.
        """
        self._load_cfg(self._prepare, config_file, kwargs)

        print("Will configure SHARP geometry with the following parameters:\n")
        for key, val in self._prepare.items():
            print("%20s : %s" % (key,str(val)))
        

        """
        COMMUNICATION MAGIC
        """
        """
        ERROR MESSAGE MAYBE
        """
        if has_comm:
            command = dict()
            command["command"] = "prepare"
            command["data"] = self._prepare
            self.comm.send(command)
        
    def update(self, **kwargs):
        
        upd = DEFAULT_update.copy()
        
        self._load_cfg(upd, None, kwargs)
        
        """
        COMMUNICATION MAGIC
        """
        """
        ERROR MESSAGE MAYBE
        """
        if has_comm:
            command = dict()
            command["command"] = "update"
            command["data"] = upd
            self.comm.send(command)
        
    def push(self, what, **kwargs):
        
        """
        Push a payload to Sharp Remote Socket.
        This call may happen in parallel on many processes
        and repeated.
        """
        
        frame = PAYLOADS[what].copy()
        
        self._load_cfg(frame, None, kwargs)
    
        """
        COMMUNICATION MAGIC
        """
        """
        ERROR MESSAGE MAYBE
        """
        if has_comm:
            command = dict()
            command["command"] = "frame"
            command["data"] = frame
            self.comm.send(command)
        
    def completed(self):
        
        """
        Signals the completion of a scan 
        """
        
        print("Transfer to Sharp Socket completed")
        if has_comm:
            command = dict()
            command["command"] = "completed"
            command["data"] = "finished"
            self.comm.send(command)
            
    def file_ready(self, cxifile):
        
        """
        Signals the availabilty of cxi file for storaf
        """
        
        print("File ready")
        if has_comm:
            command = dict()
            command["command"] = "file"
            command["data"] = cxifile
            self.comm.send(command)
