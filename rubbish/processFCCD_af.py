import numpy as np
from util import shift
from scipy.interpolate import interp1d
import sys
import arrayfire
"""
This should be part of a class
"""


class ProcessFCCD(object):
    # All paramaters at class level. 
    # There will be only one instance so that is ok.
    
    # Parameters
    rows = 480
    n = 2*rows ##here the vertical size is equal to the final size because overscan modulates horizontal
    ndd = 10 ##number of data columns for super column
    nos = 2 ##number of overscan columns per super column
    ns = 192 ##number of super columns
    
    kwargs_spike_detection = dict(
        fThreshold = 400.,         ##intensity threshold for the FFT filter
        fRange = 500,             ##range of low frequency pixels to ignore
        bThreshold = 4.0,
        original = True,
    )

    kwargs_overscan = dict(
        flat_pixels = 100,      ## ignored pixel
    )
    
    def __init__(self,verbose = True, 
                      fullProcess = True,
                      bkg = None, 
                      bkg_short = None):
                      
        osmask = _stretch(np.zeros(bkg.shape))
        osmask[:,0::12] = 1
        osmask[:,11::12] = 1
        dmask = 1 - osmask
        self.indexd = np.where(dmask)
        self.indexo = np.where(osmask)
        self.bkg
        self.bkg_short
        self.fullProcess = fullProcess
        self.fitmatrix   # for overscan
        
    def processStack(self, data, data_short, fac = 1.0):
               
        bkg =  self.bkg
        bkg_short = self.bkg_short       
               
        res = []
        for ii, d in enumerate(data):
            _data = d - bkg if bkg is not None else d
            _data = self.stretch(_data)
            
            if data_short is None:
                _data_short = None
            else:
                if bkg_short is None:
                    _data_short = data_short[ii]
                else:
                    _data_short = data_short[ii]- fac * bkg_short[ii] 
                _data_short = self.stretch(_data_short)
            
            dd =  process_one_frame(_data,_data_short)
            
            res.append(_stack(dd))
            if verbose:
                sys.stdout.write('Processed %d percent \r' % (100*ii/len(data)))
                sys.stdout.flush()
    
        return np.asarray(res)

    def process_one_frame(self,data, data_short, 
                        do_dual_exposure = self.fullProcess,
                        kwargs_dual_exposure = None, 
                        do_overscan = self.fullProcess,
                        kwargs_overscan = None,
                        do_spike_removal = self.fullProcess,
                        kwargs_spike_removal = None):
   
        data = [data, data_short]
        
        out = []
        
        for ind, d in emumerate(data):
            
            if d is None or not do_dual_exposure:
                # return processed frame
                return out[0]
                break
            
            dd = np.reshape(d[indexd],(ny , ndd * ns))
            
            if do_overscan:
            
                os = np.reshape(d[indexo],(ny , nos * ns))
                ##in this context, OSMASK masks pixels in overscan which are 0, from dropped packets
                osmask = os > 0. #ignores the dropped packets, I hope
                os = os * osmask
                
                kw = kwargs_overscan
                kwargs = self.kwargs_overscan.copy() if kw is None else kw
                
                bkg_os = fitOverscan(os, fitmatrix= self.fitmatrix, **kwargs)
                
                dd -= bkg_os
            
            if do_spike_removal:
                
                kw = kwargs_spike_removal
                kwargs = self.kwargs_spike_removal.copy() if kw is None else kw
                
                bkg_ripple = spike_detection(dd, **kwargs)
                
                dd -= bkg_ripple
            
            out.append(dd)
            
        # assemble hdr frame
        dlong,dhort = out
        kw = kwargs_dual_exposure.copy()
        smask = dlong < kw['saturation_threshold']
        t0,t1 = kw['t_exp']
        return dlong * smask + (1 - smask) * dshort * t0 / t1 
    
    
    @publicmethod
    def flip(A):
        return np.rot90(x,2)
        
    @publicmethod
    def stretch(A):
        ny = A.shape[0]
        return np.hstack((A[ny / 2:ny,:], flip(A[0:ny / 2,:])))
    
    @publicmethod   
    def stack(A):
        n = A.shape[1] / 2 
        return np.vstack((flip(A[:,n:]),A[:,0:n]))



def fitOverscan(a, flat_pixels = 100, fitmatrix=None):
    
    # avoids double computation
    if fitmatrix is not None:
        ll = a.shape[0]
        xxb = np.matrix(np.linspace(0,1,ll)).T
        w = np.ones((ll))
        w[0:flat_pixels] = 0
        w = np.matrix(np.diag(w))
        Q = np.matrix(np.hstack((np.power(xxb,8),np.power(xxb,6),np.power(xxb,4),np.power(xxb,2),(xxb * 0. + 1.))))
        fitmatrix = Q * (Q.T * w * Q).I * (w * Q).T 
     
    bnfw = np.rot90(fitmatrix * np.matrix(a),2)

    jkjk = np.array(bnfw) # / np.rot90(mask,2)
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
    
def spike_detection(A, thr = 400, thr_frequency = 500, thr_background = 4.0, original=True):
    """
    Suppresses high-frequency ripple noise across adc readout blocks (supecolumns). 
    """
    bthr = thr_background
    rows = A.shape[0]
    cols = A.shape[1]
    ndd  = 10
    ns   = cols / ndd
    
    if original:
        # original
        jk = np.reshape(np.transpose(np.reshape(A,(rows,ndd,ns)),(0, 1, 2)), (rows * ndd, ns))
        jkf=(np.fft.fftn((jk*(np.abs(jk)<bthr))))#; % only when it is background...
        msk = abs(jkf) > thr; 
        msk[0:thr_frequency,:] = 0 #; % Fourier mask, keep low frequencies.
        bkg = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk),(ndd,rows,ns)),(0, 1, 2)),(rows,ndd * ns))
    
    else:    
        # simplified
        jk = np.reshape(A, (rows * ndd, ns))
        bkg = jk*(np.abs(jk)<bthr)
        
        # readouts independent (transform only along supercolumn)
        bkg_fourier = np.fft.fft2(bkg,axis=0)
        
        # threshold dependent on colum width
        msk = abs(jkf) > thr; 
        
        # keep low frequencies (why?)
        msk[0:thr_frequency,:] = 0
        
        bkg = np.reshape(np.fft.ifft2(bkg_fourier * mask,axis=0),(rows,ndd * ns))
        
    return bkg.real
    
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
    
    for ind, d in emumerate(data):
        
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
    dlong,dhort = out
    smask = dlong < saturation_threshold
    return dlong * smask + (1 - smask) * dshort * t_exp[0] / t_exp[1]     


    
                                
                            
def processfCCDFrame_old(data, data_short, indexd, indexo, t_exp, \
                        saturation_threshold, verbose = True, fullProcess = True):

    ny,nx = data.shape ## data already flipped here
    n = 2*ny ##here the vertical size is equal to the final size because overscan modulates horizontal
    ndd = 10 ##number of data columns for super column
    nos = 2 ##number of overscan columns per super column
    ns = 192 ##number of super columns
    fThreshold = 400.  ##intensity threshold for the FFT filter
    fRange = 500 ##range of low frequency pixels to ignore
    bThreshold = 4.0
    
    if fullProcess:
        ##data = np.hstack((data[ny / 2:ny,:], np.fliplr(np.flipud(data[0:ny / 2,:]))))
        ll = data.shape[0]
        #data = data - bkg
        
        dd = np.reshape(data[indexd],(ny , ndd * ns))
        os = np.reshape(data[indexo],(ny , nos * ns))
        ##in this context, OSMASK masks pixels in overscan which are 0, from dropped packets
        osmask = os > 0. #ignores the dropped packets, I hope
        os = os * osmask
        smask = dd < saturation_threshold
        bkg1 = fitOverscan(os)
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
            ##data_short = np.hstack((data_short[ny / 2:ny,:], np.fliplr(np.flipud(data_short[0:ny / 2,:]))))
            #data_short -= 1.0 * bkg_short
            dd_short = np.reshape(data_short[indexd],(ny , ndd * ns))
            osshort = np.reshape(data_short[indexo],(ny , nos * ns))
            osmask = osshort > 0.
            osshort = osshort * osmask
            bkg1_short = fitOverscan(osshort)
            dd_short -= 1.0 * bkg1_short
            jk = np.reshape(np.transpose(np.reshape(dd_short,(ll,ndd,ns)),(0, 1, 2)), (ll * ndd, ns))
            jkf = (np.fft.fftn((jk * (abs(jk) < bThreshold))))#; % only when it is background...
            msk = abs(jkf) > fThreshold; 
            msk[0:fRange,:] = 0 #; % Fourier mask, keep low frequencies.
            jkif = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk),(ndd,ll,ns)),(0, 1, 2)),(ll,ndd * ns))
            dd_short -= jkif.real
            dd = dd * smask + (1 - smask) * dd_short * t_exp[0] / t_exp[1]
		    
        ##dd =  vstack((flipud(fliplr(dd[:,n:(ndd * ns)])),dd[:,0:n]))
        
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

"""
def processfCCDFrame(data, data_short, bkg, bkg_short, os_bkg, indexd, indexo, t_exp, saturation_threshold,noise_threshold, xc,yc,fNorm,framenum, dSize, version = 1, verbose = True, fullProcess = True):
    
    if yc > 970:
        xc, yc = xc - 84, yc - 572
    else:
        xc, yc = xc - 84, yc - 500

    if verbose:
        import sys
        sys.stdout.write("Extracting fCCD data frame %i \r" %framenum)
        sys.stdout.flush()

    #fullProcess = False

    if fullProcess:
        #if version == 2: data = np.hstack((data[534:1000,:], np.fliplr(np.flipud(shift(data[1:467,:],0,2)))))
        #elif version == 1: data = np.hstack((data[534:1000,:], np.fliplr(np.flipud(shift(data[1:467,:],-1,2)))))
        data = np.hstack((data[534:1000,:], np.fliplr(np.flipud(shift(data[1:467,:],0,2)))))
        bkg = np.hstack((bkg[534:1000,:], np.fliplr(np.flipud(shift(bkg[1:467,:],0,2)))))
        ll = data.shape[0]
        data = data - 0.999 * bkg
        osmask = np.zeros(bkg.shape)
        osmask[:,0::12] = 1
        osmask[:,11::12] = 1
        dmask = 1 - osmask
        indexd = np.where(dmask)

        indexo = np.where(osmask)
        dd = np.reshape(data[indexd],(466,10*192))
        os = np.reshape(data[indexo],(466,2*192))
        osmask = os > 0. #ignores the dropped packets, I hope

        if data_short != None:
            data_short = np.hstack((data_short[534:1000,:], np.fliplr(np.flipud(shift(data_short[0:466,:],-1,0)))))
            data_short -= bkg_short
            dd_short = np.reshape(data_short[indexd],(466,10*192))
            osshort = np.reshape(data_short[indexo],(466,2*192))
            os = os * osmask + (1 - osmask) * osshort
        else: os = os * osmask

        smask = dd < saturation_threshold


        bkg1 = fitOverscan(os, osmask)
        

        dd -= bkg1
        dd = dd * (dd > 0)

        ll = dd.shape[0]
        lw = dd.shape[1]

        jk = np.reshape(np.transpose(np.reshape(dd,(ll,10,192)),(0, 1, 2)), (ll * 10, 192))
        #jkf=(np.fft.fftn((jk*(np.abs(jk)<4.0))))#; % only when it is background...
        #msk = np.abs(jkf) > 400.; msk[0:500,:] = 0 #; % Fourier mask, keep low frequencies.
        #jkif = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk),(10,ll,192)),(0, 1, 2)),(ll,1920))
        #dd -= jkif.real

        dd[0,9::10] = 0
        
        if data_short != None: dd = dd * smask + (1 - smask) * dd_short * t_exp[1] / t_exp[0]

        dd =  np.vstack((np.flipud(np.fliplr(dd[:,960:1920])),dd[:,0:960]))

    else:
        dd = np.vstack((data[0:466,:], shift(data[534:1000,:],0,0)))
        dd = np.reshape(dd[indexd],(932,960))
        dd -= np.reshape(np.vstack((bkg[0:466,:], shift(bkg[534:1000,:],0,0)))[indexd],(932,960))

    if version == 1:
        ##do some shifting because the CCD de-scrambling is incorrect
        dd[466:932,9::10] = shift(dd[466:932,9::10],0,1)
        dd[0:466,0::10] = shift(dd[0:466,0::10],0,-1)
        dd[0:467,:] = shift(dd[0:467,:],0,-4)
        dd = dd[yc - dSize / 2:yc + dSize / 2, xc - dSize / 2: xc + dSize / 2]

    elif version == 2:
        dd[0:470,:] = shift(dd[0:470,:],0,-8)
        dd[468,8::10] = dd[468,0::10]
        dd[468,9::10] = dd[468,3::10]
        dd[470,0::10] = dd[471,0::10]
        dd[470,1::10] = dd[470,2::10]
        dd[469,:] = (dd[468,:] + dd[470,:]) / 2.
        # temp = dd[:,0:50].mean(axis = 1)
        # dd = dd - np.reshape(temp,(len(temp),1))
        ##crop
        dd = dd[yc - dSize / 2:yc + dSize / 2, xc - dSize / 2: xc + dSize / 2]

    ##remove negatives and multiply by the beamstop scale factor
    ##comment the next two lines if there is no beamstop
    dd -= dd[0,:].mean() / 2.
    dd -= noise_threshold
    dd *= dd > 0
    #dd *= fNorm ##this is the beamstop normalization

    if np.isnan(dd.min()): 
        print "Nan encountered!"
    return dd
"""
