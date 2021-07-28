import time, os, sys

from cosmic.utils import dist, shift, circle, e2l, eval_hdr

from scipy import random, ndimage, interpolate

from scipy import polyfit, misc, ndimage, mgrid, rand

from processFCCD import ProcessFCCDraw # fitOverscan, processfCCDFrame

import numpy as np

import cosmic.ext as ce

from cosmic.io import cxi,writeCXI, SharpClient

import warnings

from glob import glob
    
parallel = ce.utils.parallel

#imread = ce.tifffile.imread

class AttrDict(object):
    """
    Wrapper around dictionary for convenient attribute access
    """
    def __init__(self,dct=None):
        
        self.set_dict(dct)
    
    def get_dict(self):
        return self.__dict__
        
    def set_dict(self,dct = None):
        
        if dct is not None:
            self.set_dict(dct)
        
class Point(object):

    def __init__(self,x=0,y=0, dtype=np.float):
        self.arr = np.array([y,x],dtype=dtype)
    
    @property 
    def x(self):
        return self.arr[1]
    
    @x.setter
    def x(self,val):
        self.arr[1] = val
        
    @property 
    def y(self):
        return self.arr[0]
    
    @y.setter
    def y(self,val):
        self.arr[0] = val
        

class Log(object):

    def __init__(self, verbose=True):
        self.verbose = verbose and parallel.master
    
    @property    
    def emit(self):
        # non mpi
        print self.msg
    
    @emit.setter
    def emit(self,msg):
        self.msg = msg
        if self.verbose:
            print msg

from cosmic.ext.ptypy.utils.validator import ArgParseParameter
from StringIO import StringIO

DEFAULT = ArgParseParameter(name='Preprocessor')

DEFAULT.load_conf_parser(StringIO(
"""
[scan]
default = None
help = Scan specific parameters. These are mostly set by STXM control 
       and will be sideloaded if possible from a json file.
       
[scan.dark_dir]
default = None
help = Directory with diffraction patterns obtained at closed shutter.
           
[scan.exp_dir]
default = None
help = Directory with exposed diffraction patterns
                        
[scan.number]
default = None
help = Number of scan
                        
[scan.date]
default = None
help = Date of scan
                        
[scan.id]
default = None
help = Identifying string for scan
                        
[scan.dwell] 
default = None
help = Dwell times 

[scan.repetitions]
default = 1
help = Exposure repetition per scan point

[scan.double_exposure]
default = False
help = Inidicates double exposure mode

[process]
default = None
help = Processing paramater for diffraction images

[process.resolution]
default = 5e-9
help = Target resolution value in nm

[process.shape]
default = 256
help = Target diffraction pattern shape after cropping / binning

[process.gap]
default = 70
help = Vertical gap in the diffraction images. 
            
[process.threshold] 
default = 50000
help = Fast CCD per pixel saturation threshold
alias = s_threshold

[process.crop]
default = 960
help = Raw data will be cropped to a pixel areas of this size
alias = cropSize

[process.scols] 
default = 192
help = Total number of supercolumns
                        
[process.do_overscan]
default = True
help = Use overscan areas additional background substraction.
                        
[process.do_spike_removal]
default = True
help = Remove spikes from cross channel crosstalk.

[geometry]
default = None
help = Information about the imaging geometry. Maybe filled

[geometry.zp_diameter]
default = 240.0
help = Diameter of the zone plate in micrometer
alias = zd

[geometry.zp_outer_width]
default = 0.1
help = Zone plate outer zone width in micrometer
alias = dr

[geometry.energy]
default = 800
help = Photon energy in eV
alias = e

[geometry.psize]
default = 30 
help = Physical CCD pixel size in um, usually 30 microns
alias = ccdp

[geometry.distance]
default = 79.55
help = Distance from sample to CCD in mm
alias = ccdz

[post]
default = None
help = Parameters for secondary processing of diffraction data

[post.truncate]
default = False
help = Suppress high spatial frequncies with a Gaussian
alias = low_pass 

[post.low_pass]
default = 0.
help = RMS of low pass applied filter to diffraction data if >0
alias = filter_width 

[post.noise_level]
default = 0.5
help = Baseline noise level to subtract
alias = nl

[post.probe_threshold]
default = 0.1
help = Threshold for annulus to estimate probe from diffraction patterns.
alias = pthreshold

[post.defocus]
default = 0.0
help = Amount to defocus the probe by after the initial estimate if >0
"""))

def process(param=None, scanfile=None,verbose = True, **kwargs):

    log = Log(verbose)
    sharpy = SharpClient()
    
    param = DEFAULT.make_default(depth=10) if param is None else param
    import json
            
    if scanfile is not None:
        
        try:
            if scanfile.endswith('json'):
                with open(scanfile,'r') as f:
                    scan = json.load(f)
                    scan['dwell'] = (scan['dwell1'],scan['dwell2'])
                    f.close()
                # sideload scan specific stuff
                for k in param['scan'].keys():
                    if k in scan:
                        param['scan'][k] = scan[k]
                        
                for k in param['geometry'].keys():
                    if k in scan:
                        param['geometry'][k] = scan[k]
            
            if scanfile.endswith('.hdr'):
                d = eval_hdr(scanfile)
                param['geometry']['energy'] = 700
                param['geometry']['energy'] = 700
                param['geometry']['energy'] = 700
                param['geometry']['energy'] = 700
                param['geometry']['energy'] = 700
                param['scan']['dark_dir'] = 700
                param['scan']['dark_dir'] = 700
                param['scan']['dark_dir'] = 700
                param['scan']['dark_dir'] = 700
                param['scan']['dark_dir'] = 700
                param['scan']['dark_dir'] = 700
            else:
                log.emit = 'Scanfile %s is not understood' % scanfile
                
                    
        except:
            log.emit = 'Could not load parameter file %s' % scanfile
            return
        
    ### IO #####
    
    #if parallel.master:
    #    print json.dumps(param,indent=2)
    #    print json.dumps(scan,indent=2)
    
    """
    ### parameters needed
    dataPath = param['dataPath']
    scanDate = param['scanDate']
    scanNumber = param['scanNumber']
    scanID = param['scanID']
    bgScanDate = param['bgScanDate']
    bgScanNumber = param['bgScanNumber']
    bgScanID = param['bgScanID']
    
    ######################################################################################################
    ###setup some of the file name prefixes
    scanDir = param["scanDir"]
    
    if bgScanDate == '0': bgScanDate = scanDate
    bg_subtract = True
    bgScanDir = param["bgScanDir"]
    bg_prefix = 'NS_' + bgScanDate + bgScanNumber + '_'
    """
    scan_dir = param['scan']['exp_dir']
    if not param['scan']['id']:
        identifier = 'NS_'+os.path.split(os.path.split(scan_dir)[0])[1]

    param['scan']['id'] = identifier
    
    outputPath = param['scan']['exp_dir'] + os.path.sep
    cxifile = identifier +'.cxi'
    
    ## SETUP sharp client
    if parallel.master:
        sharpy.setup( num_proc_ids = parallel.size )
        
    #sh_out = Point(int(param['sh_sample_x']),int(param['sh_sample_y']),dtype=np.int) 
    # size to which the raw data is resampled, usually 256x256
    N = param['process']['shape']
    sh_out = Point(N,N,dtype=np.int)
    
    ss = Point(scan['exp_step_y']*1000,scan['exp_step_x']*1000)
    # scan step size in X/Y
    resol = param['process']['resolution']
    pixnm = Point(resol,resol, dtype=np.single)
    # requested reconstructed pixel size, usually ~5 nm
    # actually only pixnm.x is used
    
    pts = Point(scan['exp_num_x'],scan['exp_num_y'], dtype=np.int)
    # number of scan points in X and Y for raster scans
    geo = param['geometry']
    zd = geo['zp_diameter'] ## zone plate diameter
    dr = geo['zp_outer_width'] ## zone plate outer zone width
    e = geo['energy'] #photon energy
    ccdp = geo['psize'] ##physical CCD pixel size, usually 30 microns
    ccdz = geo['distance'] * 1000. ## sample to CCD distance, converted to microns
    
    #useBeamstop = int(param['useBeamstop']) ##flag to apply beamstop correction
    """
    fw = float(param['filter_width']) ##width of the low pass filter (smoothing) of the data
    df = int(param['lowPassFilter']) ##flag to smooth the data prior to downsampling, filter with a gaussian

    # beamstop calculation
    bsNorm = 1. / float(param['beamStopTransmission']) ##normalization factor for the beamstop correction
    beamstopThickness = int(param['beamstop'])
    bsXshift = int(param['beamstopXshift'])
    bsYshift = int(param['beamstopYshift'])
    """

    cropSize = param['process']['crop']
    multi = param['scan']['double_exposure']
    dwell = (1.0,scan['dwell1']/scan['dwell2']) if multi else [1.0] # may be switched
    ######################################################################################################
    ###calculate the number of pixels to use for raw data (the "crop" size)
    #sh_out.y, sh_out.x = sh_out  ##final size of the data array, usually 256x256
    
    # wavelength in nm
    l = e2l(e) 
    
    #zp focal length microns
    f = zd * dr / l 
    
    # zp numerical aperture
    na = zd / f / 2. 
    
    # scattering angle
    theta = l / 2 / pixnm.arr
    
    # number of pixels (at physical CCD pixel size) before downsampling 
    # required for the requested pixel size
    sh_pad = Point()
    sh_pad.arr = 2 * np.round(ccdz * theta / ccdp)
    
    
    log.emit = "Scan Number: %s, Scan ID: %s" %(str(param['scan']['number']), identifier)
    log.emit = "Output file: %s" % cxifile
    log.emit = "Photon energy: %.2f eV" %e
    log.emit = "Wavelength: %.2f nm" %l
    log.emit = "Zone plate focal length: %.2f mm" %f
    log.emit = "sample-CCD distance: %.2f mm" %(ccdz / 1000.)
    log.emit = "Requested pixel size (nm): %.2f" %(pixnm.x)
    log.emit = "Will pad the data to: %i x %i (HxV)" %(sh_pad.x,sh_pad.y)
    log.emit = "Will downsample data to: %i x %i (HxV)" %(sh_out.x,sh_out.y)
    log.emit = "Probe step in pixels (x,y): %.2f,%.2f" %(ss.x / pixnm.x, ss.y / pixnm.y)
    log.emit = "Intensity threshold percent for probe calculation: %i" %(100. * param['post']['probe_threshold'])

    expectedFrames = pts.x * pts.y
    
    log.emit = "Expecting %i frames." %(expectedFrames)    
    
    t0=time.time()
    
    ## PREPARE sharp client
    detpx = float(sh_pad.x) / float(sh_out.x) * float(ccdp) / 1e6
    if parallel.master:
        sharpy.prepare(
            energy = e * 1.602e-19,
            distance = ccdz * 1e-6,
            num_frames = expectedFrames,
            shape = tuple(sh_out.arr),
            pixelsize = (detpx,detpx),
            run_file = param.get('outputFilename'),
            )
    ######################################################################################################
    ###load the list of tiff file names
    proc = param['process']
    Dealer = TiffDealer(scan['exp_dir'],
                        scan['dark_dir'],
                        exp_total=scan['exp_num_total'],
                        dark_total=scan['dark_num_total'],
                        gap=proc['gap'], 
                        dtype= np.float,
                        dwell = dwell, 
                        repetitions = param['scan']['repetitions'], ## to be replaced by scan['repetitions'] once stxm control share the info
                        threshold = proc['threshold'], 
                        rows = proc['crop'] / 2,
                        scols = proc['scols'],
                        do_overscan = proc['do_overscan'],
                        do_spike_removal = proc['do_spike_removal'])
                       
    while Dealer.frames_available() <=0:
        log.emit = "Waiting for frames"
        time.sleep(0.1)
      
    log.emit = "Starting frame is: %i" %(Dealer.frame_nums[0])

    ######################################################################################################
    ###load and average the background frames
    ###this alternates long and short frames regardless of "multi", should be fixed
    log.emit = "Averaging background frames..."
        
    ndarks = Dealer.set_dark()
    
    log.emit = "Done. Averaged %d frames." % ndarks

    ######################################################################################################
    ## Translation & data preparation
    pixIndex = [(i / pts.x, i % pts.x) for i in range(expectedFrames)]
    shiftData = [(-(float(pixIndex[i][0]) - float(pts.y - 1) / 2) * ss.y, (float(pixIndex[i][1]) - float(pts.x - 1) / 2) * ss.x) for i in range(expectedFrames)]
    x = np.array(shiftData)[:,1]
    y = np.array(shiftData)[:,0]
    translationX, translationY = (x - x.min()) * 1e-9, (y - y.min()) * 1e-9
    
    #(x,y,z) in meters
    translation = np.column_stack((translationX, translationY, np.zeros(translationY.size))) 
    
    #(x,y,z) in meters
    corner_position = [float(sh_pad.x) * ccdp / 2. / 1e6, float(sh_pad.y) * ccdp / 2. / 1e6, ccdz / 1e6] 

    nPoints = pts.x * pts.y
    indices = np.arange(nPoints)
    
    # prepare the final output
    dataStack = np.zeros((nPoints, sh_out.y, sh_out.x))
    
    ######################################################################################################
    ###set up the low pass filter and beamstop normalization
    ##beamstop
    """
    if beamstopThickness == 20:
        bs_r = 31.  ##beamstop radius in pixels
    else: 
        bs_r = 37 * pixnm.x / 15.25 * e / 800.

    bs = 1. + ((bsNorm - 1) * circle(sh_out.x, bs_r, sh_out.x / 2, sh_out.x / 2))
    bs = np.roll(np.roll(bs, bsXshift, axis = 1), bsYshift, axis = 0)
    fNorm = ndimage.filters.gaussian_filter(bs, sigma = 0.5) #we multiply the data by this to normalize
    """



    #### MPI DISTRIBUTION #################
    
    #indices = range(len(tiffs))
    chunk_size = 4 * parallel.size


    tPrep = time.time() - t0
    t0 = time.time()
    
    tRead = 0
    tProcess = 0

    start = 0
    
    end_of_scan = False
    
    for i in range(1000): # maximum wait time is 1000sec
        
        if end_of_scan: break
        
        parallel.barrier()
        

        # indices to distribute
        frames_available  = Dealer.frames_available(start)
        
        log.emit = "%d new frames." %  frames_available
        
        stop = start+chunk_size
        
        if frames_available  < chunk_size:
            # too little frames means either end-of scan or pause
            if frames_available >= 0 and (stop >= expectedFrames):
                # end of scan
                stop = expectedFrames
                end_of_scan = True
            else:
                #pause
                time.sleep(1.0)
                continue

        chunk_indices = range(start,stop)
            
        log.emit = "Processing frames %d through %d." %(start,stop) 
        
        #chunk_indices = indices[start:stop]
        
        # MPI distribution keys
        parallel.loadmanager.reset()
        distributed = parallel.loadmanager.assign(chunk_indices)
         
            
        # node specific
        node_indices = [chunk_indices[j] for j in distributed[parallel.rank]]
        
        if not node_indices and end_of_scan:
            # You may leave now
            break
        
        # load and crop from indices
        data = Dealer.get_clean_data(node_indices)

        tcRead = time.time()-t0
        t0 = time.time()
        log.emit = "Read %d frames on %d processes in %.2f seconds" % (len(chunk_indices),parallel.size,tcRead)
        
        ## Cropping / Rebinning ##
        
        if data.shape[-2] != cropSize or data.shape[-2] != cropSize:
            self.emit = "Raw data shape is %d x %d" % data.shape[-2:] + " and should be %d x %d."% (cropSize,cropSize)
            self.emit = "Non-rectangular diffraction frames may lead to unexpected results in the interpolation"
           
        post = param['post']
        
        """
        from matplotlib import pyplot as plt
        print data[-1].min()
        plt.imshow(data[-1]-data[-1].min())
        plt.show()
        """
        if start == 0:
            ## center detection
            dataAve = data.sum(axis = 0) 
            
            """
            from matplotlib import pyplot as plt
            plt.imshow(np.log10(dataAve+1000))
            plt.show()
            """
            # average dataset across all nodes
            parallel.allreduce(dataAve)
            
            # locate the center of mass of the average and shift to corner
            dataAve = dataAve * (dataAve > 10.) #threshold, 10 is about 1 photon at 750 eV
            
            cen = ce.utils.mass_center(dataAve)
            log.emit = "Found center at %s " % str(cen)
            
            # filter for smoothing the diffraction data
            fw = post['low_pass']
            if fw > 0.:
                log.emit = "Applying filter on data transform"
                data = ce.utils.gf_2d(data, fw)
            
            
            sh = data.shape[-2:]
            row = np.linspace(0, sh[0]-1, sh[0]) - cen[0]
            col = np.linspace(0, sh[1]-1, sh[1]) - cen[1]
            symspace = np.linspace(0,sh_pad.x-1,sh_out.x) - sh_pad.x /2
            evcol, evrow = np.meshgrid(symspace,symspace)
    
            # same for the average
            spline = interpolate.RectBivariateSpline(row, col, dataAve)
            dataAve = spline.ev(evrow.flatten(), evcol.flatten()).reshape(sh_out.x,sh_out.x)
            
            dataAve = ce.utils.switch_orientation(dataAve,(True,False,True))
            
            # calculate probe from dataAve
            log.emit = "Calculating probe and mask"
            
            # Calculate the probe from the average diffraction pattern
            dataAve[dataAve < 0.] = 0.
            fave = np.fft.fftshift(dataAve)
            probe_Fmask = (fave > post['probe_threshold'] * fave.max())
            probe = np.fft.fftshift(np.fft.ifftn(np.sqrt(fave) * probe_Fmask)) 
            N = probe_Fmask.shape[0]
            probe_Rmask = circle(N, N / 4, N / 2, N / 2)
            
            # We can propagate this to some defocus distance if needed.
            """
            if post['defocus'] != 0:
                from propagate import propnf
                probe = propnf(probe, defocus * 1000. / pixnm.x, l / pixnm.x)
            """
            
            ## SHARP ENDPOINT access ###
            if parallel.master:
                sharpy.update(
                    probe_init = probe, 
                    probe_fmask = probe_Fmask, 
                    probe_mask = probe_Rmask,
                    positions = translation[:,:2]
                )
            
        interpolated = []
        
        for d in data:
            spline = interpolate.RectBivariateSpline(row, col, d)
            interpolated.append(spline.ev(evrow.flatten(), evcol.flatten()).reshape(sh_out.x,sh_out.x))
            
        data = np.asarray(interpolated)
        

    
        # truncate with a gaussian if requested
        if post['truncate']:
            data *= np.exp(-dist(sh_out.x)**2 / 2. / (sh_out.x / 4)**2)
    
        # subtract a noise floor
        data -= post['noise_level']
        data *= data > 0.
    
        """
        ####Return (obsolete)
        if data[-1,:,:].mean() == 0.0: 
            print rank, tempData
        """    
        
        # rot90 on the last two axis = fliplr+transpose
        data = ce.utils.switch_orientation(data,(True,False,True))
        #data = np.transpose(np.rot90(np.transpose(data)))
        
    
        ######################################################################################################

        ## MPI PUSH to sharp endpoint.
        for ind, d in zip(distributed[parallel.rank],data):
            out_dict = dict(
                num = ind,
                process_id = parallel.rank,
                position = translation[ind][:2],
                data = d,
                mask = None,
            )
            sharpy.push('clean' , **out_dict)

        ####################################################################
        ## Put data into the right spot in the stack           
        dataStack[node_indices]= data
        
        
        
        tcProcess = time.time()-t0
        t0 = time.time()
        log.emit = "Processed %d frames on %d processes in %.2f seconds" % (len(chunk_indices),parallel.size,tcProcess)
    
        tRead += tcRead 
        tProcess += tcProcess 
        
        start = stop 
        
    ## END LOOP ##
    parallel.allreduce(dataStack) 
    
    # let the other processes quit
    if not parallel.master:  
        return 
 
    log.emit =  "Done!"
    

    #if verbose: print "Processed frames at: %.2f Hz" %(float(loadedFrames / nexp) / tProcess)
    log.emit = "Total time per frame : %i ms" %(1000. * (tProcess + tPrep + tRead) / float(expectedFrames) * parallel.size )
            
    millis = tuple(1000. * np.array([tPrep, tRead, tProcess]) / float(expectedFrames) * parallel.size)
    log.emit =  "Distribution (Prep, Load, Process): (%.2f | %.2f | %.2f ) ms" % millis


    ######################################################################################################
    ###Calculate the STXM image and make some arrays for the translations

    
    #####################################################################################################
    ###Remove outliers based on STXM intensities


    ##########################################################
    ####interpolate STXM image onto the same mesh as the reconstructed image


    ######################################################################################################
    ###Generate the CXI file

    
    log.emit = "Writing data to file: %s" % outputPath + cxifile

    ########################################################################################################
    ########################################################################################################

    cxiObj = cxi()
    cxiObj.process = {'json':json.dumps(param)}
    cxiObj.probe = probe
    cxiObj.beamline = '5.3.2.1'
    cxiObj.energy = e * 1.602e-19
    cxiObj.ccddata = dataStack
    cxiObj.probemask = probe_Fmask
    cxiObj.probeRmask = probe_Rmask
    cxiObj.datamean = dataAve
    cxiObj.illuminationIntensities = dataAve
    cxiObj.stxm = np.ones((pts.y,pts.x))
    cxiObj.stxmInterp = 'None'
    cxiObj.xpixelsize = float(sh_pad.x) / float(sh_out.x) * float(ccdp) / 1e6
    cxiObj.ypixelsize = float(sh_pad.y) / float(sh_out.x) * float(ccdp) / 1e6
    cxiObj.corner_x, cxiObj.corner_y, cxiObj.corner_z = corner_position
    cxiObj.translation = translation
    cxiObj.indices = indices
    print [(k,type(v)) for k,v in cxiObj.__dict__.items()]
    writeCXI(cxiObj, fileName = outputPath + cxifile)

    ######################################################################################################
    ######################################################################################################

    log.emit = "Done."

    #os.chmod(outputFile, 0664)
    return outputPath + cxifile
   
class AutoCRC(object):
    
    def __init__(self, out=256, pad=3*256):
        
        self.crow = None
        self.ccol = None
        self._configure(sh_out, sh_pad)

    def _configure(out=256, pad=3*256):
    
        self.pad = pad
        self.out = out 
        
        self.ratio = np.float(pad) / np.float(out)
        self.use_spline = (pad % out != 0) 
        if self.use_spline:
            self.ratio = np.float(pad) / np.float(out)
        else:
            self.ratio = pad / out
        
    def mpi_prepare(data, autocenter=True, mask = None):
        """
        Find center of mass and the corners
        """
        pass
        
def low_pass_filter(data, width, use_fftw3 = False):
    """
    Applies low-pass filter along the last 2 dimensions using fftw3
    
    ** ACTS IN-PLACE **
    
    ** needs overhaul ** 
    
    ** fftw3 acts faulty on phasis **
    """
    rows = np.abs(np.arange(data.shape[-2]) - data.shape[-2] / 2)
    cols = np.abs(np.arange(data.shape[-1]) - data.shape[-1] / 2)
    
    x, y = np.meshgrid(cols,rows)
    
    datafilter = np.exp(-x**2 / 2. / np.float(width)**2 -\
                     y**2 / 2. / np.float(width)**2)
    datafilter = np.fft.fftshift(datafilter).astype("float64")
    
    data = data * (1.0 +0.j)

    #############Do it this way to reuse the plan, requires some data copies
    try:
        import fftw3
        input_array = np.zeros_like(data[0])
        output_array = np.zeros_like(data[0])
        fftForwardPlan = fftw3.Plan(input_array,output_array,'forward')
        fftBackwardPlan = fftw3.Plan(input_array,output_array,'backward')
        for i in range(len(data)):
            np.copyto(input_array, data[i])
            fftw3.guru_execute_dft(fftForwardPlan, input_array, output_array)
            np.copyto(input_array, output_array * datafilter)
            fftw3.guru_execute_dft(fftBackwardPlan, input_array, output_array)
            np.copyto(data[i], output_array)
            
        data =  np.abs(data).astype(np.float32) / (data[0].size) #renormalize after transforms
            
    except ImportError:
        print 'failed to import fftw3, using numpy instead'
        data[:] =  np.fft.ifft2(np.fft.fft2(data) * datafilter)
        
    
class FrameDealer(object):
    """
    This class basically combines multiple exposures and carries out
    the background correction on individual CCD frames
    """
    def __init__(self, dwell = None, 
                       repetitions = 1,
                       dark_total=None,
                       exp_total=None,
                       threshold = 50000, 
                       rows = 480,
                       scols = 192,
                       do_overscan = True,
                       do_spike_removal = True):
        
        self.repetitions = repetitions
        self.threshold = threshold
        self.dwell = dwell if dwell is not None else [1.0]
        self.multi = (dwell is not None and len(dwell) > 1.)
        self.do_overscan = do_overscan
        self.do_spike_removal = do_spike_removal
        self.rows=rows
        self.dark_total = dark_total
        self.exp_total = exp_total
        # set up processor
        self.Pro = ProcessFCCDraw(rows = 480,scols = 192)
            
        
    def set_dark(self):
        """Assumes a flat list of dark frames"""
        self.dark = []
        dark_frames = self.get_darks()
        if self.dark_total is not None:
            dark_frames = dark_frames[:self.dark_total*self.repetitions*len(self.dwell)]
            
        for t,dwell in enumerate(self.dwell):
            bg = dark_frames[t::len(self.dwell)]
            norm = (bg > 0).astype(np.float64)
            norm = norm.sum(0)
            norm[norm == 0.0]=1.0
            res = bg.sum(0) / norm
            self.dark.append(res)

        return len(dark_frames)
        
    def frames_available_flat(self,start=0):
        """
        return all frames available from index `start`
        IMPLEMENT IN SUBCLASS
        """
        raise NotImplementedError
        
    def frames_available(self,start=0):
        """
        Frames available, grouped per scan point
        """
        c= self.repetitions * len(self.dwell)
        M = self.frames_available_flat(start*c)//c
        
        if self.exp_total is not None:
            return min(M,self.exp_total-start)
        else:
            return M 
        
    def get_data(self,indices):
        """
        Returns data using flat indices 
        """
        raise NotImplementedError
        
        
    def get_clean_data(self,node_indices, verbose = True):
        """Accesses data and dark attribute"""
        ### PROCESS THE DATA STACK ###########
        D = len(self.dwell)
        R = self.repetitions
        
        dmask = None
        res = None
        for d,dwell in enumerate(self.dwell):
            
            rsum = None
            smask = None
            inorm = None
            
            for r in range(R):
                flat_indeces = [ind * D * R + D * r + d for ind in node_indices]
                data = self.get_data(flat_indeces)
                
                # missing pixel mask
                imask = data > 10
                
                data -= self.dark[d]
                    
                data = self.Pro.process_stack(data, verbose, do_overscan = self.do_overscan,do_spike_removal = self.do_spike_removal)
                imask = self.Pro.process_stack(imask, False)

                """    
                if parallel.master:
                    from matplotlib import pyplot as plt
                    #plt.imshow(data[0,420:500,570:650])
                    plt.imshow(np.log10(data[0,420:500,465:545]))
                    plt.colorbar()
                    plt.show()
                """
                
                # mask calculation should come before correction. It is here only because of shape issues
                smask = data < self.threshold if smask is None else smask & (data < self.threshold)
                
                rsum = data * imask if rsum is None else rsum + data * imask
                inorm = imask if inorm is None else inorm +imask
                
            inorm[inorm<=1.]=1.
            rsum /= inorm
            
            if dmask is None:
                dmask = smask
                res = rsum  / dwell if D > 1 else rsum / dwell
            else:
                res = dmask * res + (1-dmask) * rsum / dwell 
                dmask |= smask
            
        return res

class TiffDealer(FrameDealer):
    
    def __init__(self, exp_dir,
                       bg_dir,                       
                       gap=70, 
                       dtype= np.uint16,
                       **kwargs):
        
        super(TiffDealer,self).__init__(**kwargs)
            
        self.bg_dir = bg_dir if bg_dir.endswith(os.path.sep) else bg_dir+os.path.sep
        self.exp_dir = exp_dir if exp_dir.endswith(os.path.sep) else exp_dir+os.path.sep
        self.gap = gap
        self.dtype = dtype
        self._cuts = None
        self.cols = None
        
    def get_data(self,indices):
        
        files = [self.tiffs[ii] for ii in indices]
        
        return self.load_and_crop(files)
        
    def frames_available_flat(self,start=None):
        
        for tries in range(4):
            tifFiles = sorted(glob(self.exp_dir+'*.tif'))
            if not tifFiles:
                self.tiffs = []
                self.frame_nums = []
                return 0
            
            indices = np.array([int(f.split('.')[0].split('image')[1]) for f in tifFiles])
            # when jumping a decimal, the sorting doesn't work properly
            k = np.argsort(indices)
            indices=indices[k]
            tifFiles=[tifFiles[j] for j in k]
            frame_offset = indices.min()
            is_complete = indices.max()-frame_offset+1 == len(indices)
            is_contiguous = is_complete if not is_complete else np.allclose(indices,np.arange(frame_offset,indices.max()+1)) 
            if is_contiguous:
                break
        
        if not is_contiguous:
            missing = [ii for ii in np.arange(frame_offset,indices.max()+1) if ii not in indices]
            raise IOError('File list is non-contigouus, aborted after 4 tries. Missing %s' % str(missing) )
            
        self.tiffs = tifFiles
        self.frame_nums = indices
        
        indices = range(len(self.tiffs))
        start = 0 if start is None else int(start)
    
        return len(tifFiles)-start
        
    def _set_crop(self,shape):
        
        N,cols = shape
        self.cols = cols
        self._cuts =[
            N / 2 - self.rows - self.gap / 2,
            N / 2 - self.gap / 2,
            N / 2 + self.gap / 2,
            N / 2 + self.gap / 2 + self.rows 
        ]
        if parallel.master:
            print self._cuts
    
    def _crop(self, one):
        cut1,cut2,cut3,cut4 = self._cuts

        cropped = np.zeros((2* self.rows,self.cols),dtype=self.dtype)
        cropped[0:self.rows ,:]= one[cut1:cut2,:]
        cropped[self.rows :2* self.rows,:]= one[cut3:cut4,:]
        return cropped
            
    def get_darks(self):
        files = sorted(os.listdir(self.bg_dir))
        files = [self.bg_dir+f for f in files if f.count('tif')]
        return self.load_and_crop(files)
        """
        if self.multi:
            ExpList = files[0::2]
            shortExpList = files[1::2]
            npoints = len(shortExpList)
            # in case there is one exposure
            if npoints < len(ExpList):
                wrng = "Warning : not the same number of short and long background exposures"
            else:
                wrng = None
            ExpList = ExpList[:npoints]
        else:
            ExpList = files
            shortExpList = None
            npoints = len(ExpList)
        """
        
    def load_and_crop(self, files, shape=None):

        # load first:
        # The tiffs from the beamline are non-canonical, supress warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = ce.tiffio.imread(files[0]).astype(self.dtype)
        
        sh = shape if shape is not None else f.shape
        if self._cuts is None:
            self._set_crop(sh) 

        frames = [self._crop(f)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for f in files[1:]:
                frames.append(self._crop(ce.tiffio.imread(f).astype(self.dtype)))
            
        return np.asarray(frames)
        
    
def aveDirFrames(bgScanDir, multi = False):

    files = sorted(os.listdir(bgScanDir))
    files = [bgScanDir+f for f in files if f.count('tif')]
    if multi:
        ExpList = files[0::2]
        shortExpList = files[1::2]
        npoints = len(shortExpList)
        # in case there is one exposure
        if npoints < len(ExpList):
            wrng = "Warning : not the same number of short and long background exposures"
        else:
            wrng = None
        ExpList = ExpList[:npoints]
    else:
        ExpList = files
        shortExpList = None
        npoints = len(ExpList)
    
    
    bg = load_and_crop(ExpList, dtype=np.float64)
    norm = (bg > 0).astype(np.float64)
    norm = norm.sum(0)
    norm[norm == 0.0]=1.0
    res = bg.sum(0) / norm
    
    if shortExpList is None:
        return res, None, len(bg) , wrng
    else:
        bg = load_and_crop(shortExpList)
        norm = (bg > 0).astype(np.float64).sum(0)
        norm[norm == 0.0]=1.0
        return res, bg.sum(0) / norm, len(bg), wrng

