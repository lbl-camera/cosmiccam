import numpy as np
from cosmic.preprocess import processFCCD
from cosmic.preprocess import process_5321 as pro
import time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
rows = 480
aalrows = 2*rows ##here the vertical size is equal to the final size because overscan modulates horizontal
ndd = 10 ##number of data columns for super column
nos = 2 ##number of overscan columns per super column
ns = 192 #
sh_overscan = (rows,ns*nos)
sh_frame = (rows,ns*(nos+ndd))
sh_data = (rows,ns*ndd)

osmask = np.zeros(sh_frame)
osmask[:,0::12] = 1
osmask[:,11::12] = 1
dmask = 1 - osmask
indexd = np.where(dmask)
indexo = np.where(osmask)

base = '/global/groups/cosmic/code/test_data_sets/161102035/'
bkg_dir = base + '002/'
bshort, blong, frames = pro.aveDirFrames(bkg_dir, multi = True)

short_data_file = base + '002/image7401.tif'

dshort = pro.load_and_crop([short_data_file],rows=960,gap=68,dtype= np.float)[0]

flip = lambda x: np.rot90(x,2)
get_dcols = lambda x: np.reshape(x[indexd],(rows,ndd*ns))
get_ocols = lambda x: np.reshape(x[indexo],(rows,nos*ns))

def _stretch(A):
    ny = A.shape[0]
    return np.hstack((A[ny / 2:ny,:], flip(A[0:ny / 2,:])))

bkg = _stretch(bshort)
data = _stretch(dshort)

pre_overscan = get_dcols(data-bkg)
os = get_ocols(data-bkg)

Pro = processFCCD.ProcessFCCDraw()

out = Pro.process_stack([dshort-bshort])
out *= (out > 0)

out2 = processFCCD.processfCCDFrame_old(dshort-bshort, None , indexd, indexo, 20, \
                        40000, verbose = True, fullProcess = True)

print np.allclose(out,out2)
"""
plt.figure()
plt.imshow(pre_overscan)

obkg, matrix = fccd.fitOverscan(os)
print obkg.shape
plt.figure()
plt.imshow(obkg)
plt.figure()
plt.imshow(pre_overscan-obkg)

A = pre_overscan-obkg
jk = np.reshape(A, (rows * ndd, ns))
plt.figure()
b= jk*(np.abs(jk)<4.0)
fb = np.fft.fft(b,axis=0)
plt.imshow(np.abs(fb)>400)

plt.show()



"""
class FitOverscan(object):
    
    def __init__(self,A,flat_pixels = 100):
        
        self.flat_pixels = flat_pixels
        self.sh = A.shape
        
    def setup_numpy(self):
        ll = self.sh[0]
        xxb = np.matrix(np.linspace(0,1,ll)).T
        w = np.ones((ll))
        w[0:self.flat_pixels] = 0
        w = np.matrix(np.diag(w))
        Q = np.matrix(np.hstack((np.power(xxb,8),np.power(xxb,6),np.power(xxb,4),np.power(xxb,2),(xxb * 0. + 1.))))
        self.fitmatrix = np.array(Q * (Q.T * w * Q).I * (w * Q).T)
        
        self.jkjk1 = np.zeros((2*ll,192))
        self.tt = np.reshape(np.linspace(1,12*ll,12*ll),(ll,12))
        
    def compute_numpy(self, a):
        ll = self.sh[0]
        jkjk1 = self.jkjk1 
        bnfw = np.array(np.rot90(np.dot(self.fitmatrix , a),2))
        jkjk1[::2,0:192] = bnfw[:,::2]
        jkjk1[1::2,0:192] = bnfw[:,1::2]
        tt= self.tt
        ttd = tt[:,9::-1]
        tts = tt[:,12:9:-1]
    
        f = interp1d(tts.flatten(), jkjk1, 'linear',axis = 0,bounds_error = False,fill_value = jkjk1.mean())
        bkg = np.rot90(np.reshape(f(ttd),(ll,192 * 10)),2)
        bkg = f(ttd)
        bkg1 = np.zeros((ll,1920))
    
        for i in xrange(10):
            bkg1[:,i::10] = bkg[:,i,:]

        return np.rot90(bkg1,2)   
        
    def setup_afire(self,ctx=None):
        import arrayfire as af
        self.af = af
        self.setup_numpy()
        a = self.fitmatrix
        tt = self.tt
        d = np.zeros(12*480,192)
        self.setup_numpy()
        self.fitmatrix_gpu = af.Array(a.ctypes.data, a.shape[::-1], a.dtype.char)
        self.tt_gpu = af.Array(tt.ctypes.data, tt.shape[::-1], tt.dtype.char)
        self.jkjk1_gpu = af.Array(d.ctypes.data, d.shape[::-1], d.dtype.char)
        
    def compute_afire(self, a):
        ll = self.sh[0]
        af = self.af

        if type(a) is not af.array.Array:
            a = af.Array(a.ctypes.data, a.shape[::-1], a.dtype.char)

        jkjk1 = self.jkjk1_gpu
        bnfw = af.flip(af.flip(af.matmul(a,self.fitmatrix_gpu),0),1)
        jkjk1[0:192,::12] = bnfw[::2,:]
        jkjk1[0:192,11::12] = bnfw[1::2,:]

        jkjk1 = np.transpose(np.array(jkjk1))
        tt= self.tt
        ttd = tt[:,9::-1]
        tts = tt[:,12:9:-1]
        arrayfire.signal.approx1(signal, pos0)
        f = interp1d(tts.flatten(), jkjk1, 'linear',axis = 0,bounds_error = False,fill_value = jkjk1.mean())
        bkg = np.rot90(np.reshape(f(ttd),(ll,192 * 10)),2)
        bkg = f(ttd)
        bkg1 = np.zeros((ll,1920))
    
        for i in xrange(10):
            bkg1[:,i::10] = bkg[:,i,:]

        return np.rot90(bkg1,2) 
        
"""
FO = FitOverscan(os)
FO.setup_afire()
N=10
t = time.time()
for it in range(N):
    out_npy = FO.compute_numpy(os)
dt_npy = (time.time()-t) / N

t = time.time()
for it in range(N):
    out_afy = FO.compute_afire(os)
dt_afy = (time.time()-t) / N

print np.allclose(out_npy,out_afy)

print dt_npy,dt_afy
"""


