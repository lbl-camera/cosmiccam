import numpy as np
import logging

def eV2m(eV):
    """\
    Convert photon energy in eV to wavelength (in vacuum) in micrometers.
    """
    wl = 1./eV * 4.1356e-7 * 2.9998   
    
    return wl

def abs2(X):
    return np.abs(X)**2

class NanoBallSim(object):
    """Simulate streaming from beamline scripts."""

    def __init__(self, **kwargs):

        # defaults
        self.seed = 1983
        
        # geometry
        self.energy = 800  # eV
        
        # detector
        self.N = 960
        #self.psize = 30
        self.offset = 0 #10000
        self.full_well = 100000 #63000
        self.io_noise = 5
        self.photons_per_sec = 2e7 
        self.resolution = 5e-9
        self.dist = 8e-2
        self.adu_per_photon = 4 #34
        self.dwell_times = (1000,)
        
        # probe
        self.zp_dia_outer = 15    # pixel  on screen
        self.zp_dia_inner = 6    # pixel on screen        
        
        # object
        self.nanoball_rad = 5     # nanoball radius
        self.nanoball_num = 400
              
        # scan
        self.num_data_frames = 400
        self.num_dark_frames = 25
        self.step = 80e-9
    
        self.__dict__.update(kwargs)

    @property
    def psize(self):
        return self.dist * eV2m(self.energy) / (self.resolution * self.N)
        
    @property
    def shape(self):
        return (self.N, self.N)
        
    def make_ptycho_data(self):
        self.status = "Preparing ptycho data .."
        self.create_darks()
        self.status = "Prepared dark data .."
        self.create_data()
        self.status = "Prepared ptycho data .."
        
    def create_darks(self):
        N = self.num_dark_frames
        self.darkstacks = [self._draw(np.zeros((N,)+self.shape).astype(int)) for d in self.dwell_times]
        
    def create_data(self):
        """ makes a raster ptycho scan """
        # seed the random generatot to fixed value
        np.random.seed(self.seed)
        
        self.photons = [self.photons_per_sec * d /1000 for d in self.dwell_times] 

        sh = (self.N, self.N)
        
        # positions
        num = np.int(np.sqrt(self.num_data_frames))
        R = np.arange(num)
        pos = np.array([(self.step*j,self.step*k) for j in R for k in R[::-1]])
        pixelpos = np.round(pos / self.resolution).astype(int)
        pixelpos -= pixelpos.min()
        pixelpos +=5
               
        # object
        osh = pixelpos.max(0)+np.array(sh)+10
        
        self.positions = pos
        
        self.status = "Preparing object %s .." % str(osh)
        
        nb = self.nanoball_object(osh,rad=self.nanoball_rad, num=self.nanoball_num)
        nb /= nb.max()
        self.ob = np.exp(1j*nb-nb/2.)
        
        self.status = "Preparing probe .."
        # probe
        pr, pr_fmask = self.stxm_probe(sh, outer = self.zp_dia_outer, inner = self.zp_dia_inner)
        pr /= np.sqrt(abs2(pr).sum())
        self.pr = pr
        self.pr_fmask = np.abs(pr_fmask).astype(bool)
        
        self.status = "Preparing diffraction data .."
        # exit waves
        a,b = sh
        exits = np.array([self.pr * self.ob[pr:pr+a,pc:pc+b] for (pr,pc) in pixelpos])

        fs = lambda e : np.fft.fftshift(np.fft.fft2(np.fft.fftshift(e))) / np.sqrt(sh[0] * sh[1])
        
        stack = np.array([abs2(fs(e)) for e in exits])
        
        self.diffstacks = [np.random.poisson(stack * ph) * self.adu_per_photon for ph in self.photons]
        
    def _draw(self,frames):
        return frames + np.random.normal(loc=self.offset,scale=self.io_noise,size=frames.shape).astype(int)
        
    def _convert(self,frame):
        frame[frame>self.full_well]=self.full_well
        return frame.astype(np.uint16).byteswap().tostring()
        
    @staticmethod  
    def stxm_probe(shape, inner = 8, outer = 25):
        
        #d = np.float(np.min(shape))
        X,Y = np.indices(shape).astype(float)
        X-=X.mean()
        Y-=Y.mean()
        R = (np.sqrt(X**2+Y**2) <  outer).astype(complex)
        r = (np.sqrt(X**2+Y**2) >  inner).astype(complex)
        
        probe_fmask = np.fft.fftshift(R*r)
        probe_init = np.fft.fftshift(np.fft.ifft2(probe_fmask))
        
        return probe_init, probe_fmask
    
    @staticmethod
    def nanoball_object(shape,rad=5, num=40):
        """ creates nanoballs as transmission """
        
        def cluster_coords(shape,rad = 5, num=40):
            sh= shape
            def pick():
                return np.array([np.random.uniform(0,sh[0]-1),np.random.uniform(0,sh[1]-1)])
            coords = [ np.array([np.random.randint(sh[0]/3,2*sh[0]/3-1),np.random.randint(sh[0]/3,2*sh[0]/3-1)])] 
            #np.rand.uniform(0,1.,tuple(sh)):
            for ii in range(num-1):
                noresult = True
                for k in range(10000):
                    c = pick()
                    dist = np.sqrt(np.sum(abs2(np.array(coords) - c),axis=1))
                    if (dist<2*rad).any(): 
                        continue
                    elif (dist>=2*rad).any() and (dist<=3*rad).any():
                        break
                    elif (0.001 + np.sum(8/(dist**2)) > np.random.uniform(0,1.)):
                        break
        
                coords.append(c)
            return np.array(coords)
            
        sh=shape
        out = np.zeros(sh)
        xx,yy = np.indices(sh)
        coords = cluster_coords(sh,rad,num)
        for c in coords:
            h = rad**2 - (xx-c[0])**2 - (yy-c[1])**2
            h[h<0]=0.
            out += np.sqrt(h)
            
        return out 


    @property
    def status(self):
        return self._status
        
    @status.setter
    def status(self,status):
        self.status_emit(status)

    def status_emit(self,status,color=None):
        """Change and emit status signal `status. If `color` is None use default"""
        self._status = status
        logging.info(status)
        print(status)

if __name__=='__main__':
    NBS = NanoBallSim(N=256)
    NBS.make_ptycho_data()
    
    out_dir =  "/tmp/" # "/global/groups/cosmic/code/test_data_sets/"
    out_file = "sim.cxi"
    out_path = out_dir + out_file
    
    data = NBS.diffstacks[0] - NBS.darkstacks[0].mean()
    data[data <0] = 0.
       
    num_frames = len(data)
    
    # translation
    trans = np.zeros((num_frames,3))
    trans[:,:2] = NBS.positions[:,::-1]
    
    # detector corner
    Dhalf = NBS.N * NBS.psize / 2.
    
    from cosmic.utils import circle

    from cosmic.io.endpoint import SharpClient

    pr_rmask = circle(NBS.N, NBS.N / 4, NBS.N / 2, NBS.N / 2)

    """
    from cosmic.io.readCXI import cxi
    from cosmic.io.cxiwriter import writeCXI
    # probe from data
    cxi_obj = cxi()
    cxi_obj.process = {'a':'b'}
    cxi_obj.probe = NBS.pr
    cxi_obj.beamline = 'sim'
    cxi_obj.energy = NBS.energy * 1.602e-19
    cxi_obj.ccddata = data
    cxi_obj.probemask = NBS.pr_fmask
    cxi_obj.probeRmask = pr_rmask
    cxi_obj.datamean = data.mean(0)
    cxi_obj.illuminationIntensities =  data.mean(0)
    cxi_obj.stxmInterp = 'None'
    cxi_obj.stxm = 'None'
    cxi_obj.xpixelsize = NBS.psize 
    cxi_obj.ypixelsize = NBS.psize
    cxi_obj.corner_x = Dhalf
    cxi_obj.corner_y = Dhalf
    cxi_obj.corner_z = NBS.dist
    cxi_obj.translation = trans
    cxi_obj.indices = np.arange(len(data))
    
    writeCXI(cxi_obj, fileName = out_path)
    """

    SC = SharpClient()
    
    SC.setup(
        num_proc_ids =1,
        save_dir = out_dir,
    )
    
    SC.prepare(
        energy = NBS.energy * 1.602e-19,
        distance = NBS.dist,
        num_frames = num_frames,
        shape = NBS.shape,
        pixelsize = (NBS.psize,NBS.psize),
        run_file = 'sim_streamed.cxi',
        dhalf = Dhalf
    )
    
    SC.update(
        probe_init = NBS.pr, 
        probe_fmask = NBS.pr_fmask, 
        probe_mask = pr_rmask,
        positions = trans[:,:2],
        translations = trans    
    )
    
    for ind, d in enumerate(data):
        out_dict = dict(
            num = ind,
            process_id = 0,
            position = trans[ind][:2],
            data = d,
            mask = None,
        )
        SC.push('clean' , **out_dict)
        print("Pushing frame %d" % ind)

    SC.completed()
    # call sharp
    #import os
    #cmd = "salloc --nodes=2 --tasks-per-node=2 -p batch mpirun sharp %s -t -o 10 -i 400 -b 0.6 -r 2 -M -T 2 -B" % out_path
    #os.system(cmd)
    
    
    
    
    
    
