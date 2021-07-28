import numpy as np
from util import shift
from scipy.interpolate import interp1d
import sys
"""
This should be part of a class
"""

def fitOverscan(a, flat_pixels = 100):

    ll = a.shape[0]
    xxb = np.matrix(np.linspace(0,1,ll)).T
    w = np.ones((ll))
    w[0:flat_pixels] = 0
    w = np.matrix(np.diag(w))
    Q = np.matrix(np.hstack((np.power(xxb,8),np.power(xxb,6),np.power(xxb,4),np.power(xxb,2),(xxb * 0. + 1.))))
    bnfw = Q * (Q.T * w * Q).I * (w * Q).T * np.matrix(a)
    bnfw1 = np.rot90(bnfw,2)

    jkjk = np.array(bnfw1) # / np.rot90(mask,2)
    jkjk1 = np.zeros((2*ll,192))
    jkjk1[::2,0:192] = jkjk[:,::2]
    jkjk1[1::2,0:192] = jkjk[:,1::2]

    tt = np.reshape(np.linspace(1,12*ll,12*ll),(ll,12))
    ttd = tt[:,9::-1]
    tts = tt[:,12:9:-1]

    f = interp1d(tts.flatten(), jkjk1, 'linear',axis = 0,bounds_error = False,fill_value = jkjk1.mean())
    bkg = np.rot90(np.reshape(f(ttd),(ll,192 * 10)),2)
    bkg = f(ttd)
    bkg1 = np.zeros((ll,1920))

    for i in xrange(10):
        bkg1[:,i::10] = bkg[:,i,:]

    return np.rot90(bkg1,2)

def processFCCDstack(data, data_short, bkg, bkg_short,t_exp, \
            saturation_threshold, verbose = True, fullProcess = True):
    
    flip = lambda x: np.rot90(x,2)
        
    def _stretch(A):
        ny = A.shape[0]
        return np.hstack((A[ny / 2:ny,:], flip(A[0:ny / 2,:])))
        
    def _stack(A):
        n = A.shape[1] / 2 
        return np.vstack((flip(A[:,n:]),A[:,0:n]))
    
    osmask = _stretch(np.zeros(bkg.shape))
    osmask[:,0::12] = 1
    osmask[:,11::12] = 1
    dmask = 1 - osmask
    indexd = np.where(dmask)
    indexo = np.where(osmask)
    
    fac=1.0
    
    res = []
    for ii, d in enumerate(data):
        
        _data = _stretch(d-bkg)
        if data_short is None:
            _data_short = None
        else:
            _data_short = _stretch(data_short[ii]- fac * bkg_short[ii])
        
        dd =  processfCCDFrame(_data, _data_short, indexd, \
             indexo, t_exp, saturation_threshold,verbose, fullProcess)
        
        res.append(_stack(dd))
        if verbose:
            sys.stdout.write('Processed %d percent \r' % (100*ii/len(data)))
            sys.stdout.flush()
    
    return np.asarray(res)
             

class FitOverscan(object):
    
    def __init__(self,A,flat_pixels = 100):
        
        self.flat_pixels = flat_pixels
        self.sh = A.shape

        
    def setup_numpy(self):
        ll = self.sh[0]
        xxb = np.matrix(np.linspace(0,1,ll)).T
        w = np.ones((ll))
        w[0:flat_pixels] = 0
        w = np.matrix(np.diag(w))
        Q = np.matrix(np.hstack((np.power(xxb,8),np.power(xxb,6),np.power(xxb,4),np.power(xxb,2),(xxb * 0. + 1.))))
        self.fitmatrix = Q * (Q.T * w * Q).I * (w * Q).T
        
        self.jkjk1 = np.zeros((2*ll,192))
        self.tt = np.reshape(np.linspace(1,12*ll,12*ll),(ll,12))
        
    def compute_A_numpy1(self, a):
        jkjk1 = self.jkjk1 
        bnfw = np.array(np.rot90(self.fitmatrix * np.matrix(a),2))
        jkjk1[::2,0:192] = bnfw[:,::2]
        jkjk1[1::2,0:192] = bnfw[:,1::2]   
        
    def setup_arrayfire(self,ctx=None):
        import arrayfire as af
        self.af = af
        a = self.fitmatrix
        tt = self.tt
        d = self.jkjk1
        self.setup_numpy()
        self.fitmatrix_gpu = af.Array(a.ctypes.data, a.shape, a.dtype.char)
        self.tt_gpu = af.Array(tt.ctypes.data, tt.shape, tt.dtype.char)
        self.jkjk1_gpu = af.Array(d.ctypes.data, d.shape, d.dtype.char)
        
    def compute_A_arrayfire1(self, a):
        af = self.af

        if type(a) is not af.array.Array:
            a = af.Array(a.ctypes.data, a.shape, a.dtype.char)

        jkjk1 = self.jkjk1_gpu
        bnfw = af.flip(af.flip(af.matmul(self.fitmatrix_gpu,a),0),1)
        jkjk1[::2,0:192] = bnfw[:,::2]
        jkjk1[1::2,0:192] = bnfw[:,1::2] 
        
        
        """
        zmin = en * 4. / 1000.
        dd = (dd / zmin).astype('int')

        smask = dd < saturation_threshold
        """
     
    
def processfCCDFrame(data, data_short, indexd, indexo, t_exp, 
                        saturation_threshold, 
                        verbose = True, 
                        fit_overscan = True,
                        rm_spikes = True):

    ny,nx = data.shape ## data already flipped here
    n = 2*ny ##here the vertical size is equal to the final size because overscan modulates horizontal
    ndd = 10 ##number of data columns for super column
    nos = 2 ##number of overscan columns per super column
    ns = 192 ##number of super columns
    fThreshold = 400.  ##intensity threshold for the FFT filter
    fRange = 500 ##range of low frequency pixels to ignore
    bThreshold = 4.0

    data = [data, data_short]
    
    out = []
    
    for ind, d in enumerate(data):
        
        if d is None:
            # return processed frame
            return out[0]
            break
        
        dd = np.reshape(d[indexd],(ny , ndd * ns))
        
        if fit_overscan:
        
            os = np.reshape(d[indexo],(ny , nos * ns))
            ##in this context, OSMASK masks pixels in overscan which are 0, from dropped packets
            osmask = os > 0. #ignores the dropped packets, I hope
            os = os * osmask
            
            bkg_os = fitOverscan(os)
            
            dd -= bkg_os
        
        if rm_spikes:
            
            bkg_ripple = spike_detection(dd, thr = fThreshold, 
                                        thr_frequency = fRange, 
                                        thr_background = bThreshold, 
                                        original=True)
            
            dd -= bkg_ripple
        
        out.append(dd)
        
    # assemble hdr frame
    dlong,dshort = out
    smask = dlong < saturation_threshold
    return dlong * smask + (1 - smask) * dshort * t_exp[0] / t_exp[1]     
    
                                
                        
def processfCCDFrame_old(data, data_short, indexd, indexo, t_exp, \
                        saturation_threshold, verbose = True, fullProcess = True):

    n,nx = data.shape ## data already flipped here
    ny = n/2 ##here the vertical size is equal to the final size because overscan modulates horizontal
    ndd = 10 ##number of data columns for super column
    nos = 2 ##number of overscan columns per super column
    ns = 192 ##number of super columns
    fThreshold = 400.  ##intensity threshold for the FFT filter
    fRange = 500 ##range of low frequency pixels to ignore
    bThreshold = 4.0
    
    if fullProcess:
        data = np.hstack((data[n / 2:n,:], np.fliplr(np.flipud(data[0:n / 2,:]))))
        ll = data.shape[0]
        #data = data - bkg
        
        dd = np.reshape(data[indexd],(ny , ndd * ns))
        os = np.reshape(data[indexo],(ny , nos * ns))
        ##in this context, OSMASK masks pixels in overscan which are 0, from dropped packets
        osmask = os > 0. #ignores the dropped packets, I hope
        os = os * osmask
        smask = dd < saturation_threshold
        bkg1, foo = fitOverscan(os)
        dd -= bkg1
     
        ll = dd.shape[0]
        lw = dd.shape[1]

        jk = np.reshape(np.transpose(np.reshape(dd,(ll,ndd,ns)),(0, 1, 2)), (ll * ndd, ns))
        jkf=(np.fft.fftn((jk*(np.abs(jk)<4.0))))#; % only when it is background...
        msk = abs(jkf) > fThreshold; 
        msk[0:fRange,:] = 0 #; % Fourier mask, keep low frequencies.
        jkif = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk),(ndd,ll,ns)),(0, 1, 2)),(ll,ndd * ns))
        dd -= jkif.real

        if data_short is not None:
            data_short = np.hstack((data_short[n / 2:n,:], np.fliplr(np.flipud(data_short[0:n / 2,:]))))
            #data_short -= 1.0 * bkg_short
            dd_short = np.reshape(data_short[indexd],(ny , ndd * ns))
            osshort = np.reshape(data_short[indexo],(ny , nos * ns))
            osmask = osshort > 0.
            osshort = osshort * osmask
            bkg1_short, foo  = fitOverscan(osshort)
            dd_short -= 1.0 * bkg1_short
            jk = np.reshape(np.transpose(np.reshape(dd_short,(ll,ndd,ns)),(0, 1, 2)), (ll * ndd, ns))
            jkf = (np.fft.fftn((jk * (abs(jk) < bThreshold))))#; % only when it is background...
            msk = abs(jkf) > fThreshold; 
            msk[0:fRange,:] = 0 #; % Fourier mask, keep low frequencies.
            jkif = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk),(ndd,ll,ns)),(0, 1, 2)),(ll,ndd * ns))
            dd_short -= jkif.real
            dd = dd * smask + (1 - smask) * dd_short * t_exp[0] / t_exp[1]
		    
        dd =  np.vstack((np.flipud(np.fliplr(dd[:,n:(ndd * ns)])),dd[:,0:n]))
        
    ##I don't understand why this is needed.  
    ##After background subtraction there are two bright rows with an average of ~1 photon
    ##just subtract the mean of those rows.
    ## BE: I guess that is because the loading is slightly different
    
    #dd[479,:] -= dd[479,-20:-1].mean()
    #dd[480,:] -= dd[480,-20:-1].mean()
    
    dd *= (dd > 0)
    if np.isnan(dd.min()): 
        print "Nan encountered!"
    return dd
