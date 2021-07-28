import time, os, sys

from cosmic.utils import dist, shift, circle, e2l

from scipy import random, ndimage, interpolate

from scipy import polyfit, misc, ndimage, mgrid, rand

from processFCCD import ProcessFCCDraw # fitOverscan, processfCCDFrame

import numpy as np

import cosmic.ext as ce

from cosmic.io import cxi,writeCXI, SharpClient

import warnings

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
    
    
def process(param, verbose = True):


    ### IO #####
    log = Log(verbose)
    sharpy = SharpClient()
    
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
    filePrefix = 'image'
    namePrefix = 'NS_'
    
    if bgScanDate == '0': bgScanDate = scanDate
    bg_subtract = True
    bgScanDir = param["bgScanDir"]
    bg_prefix = 'NS_' + bgScanDate + bgScanNumber + '_'
 
    outputPath = param['scanDir'] + param['outputFilename']
    
    ## SETUP sharp client
    if parallel.master:
        sharpy.setup( num_proc_ids = parallel.size )
        
    sh_out = Point(int(param['sh_sample_x']),int(param['sh_sample_y']),dtype=np.int) 
    # size to which the raw data is resampled, usually 256x256
    
    ss = Point(float(param['ssx']),float(param['ssy']))
    # scan step size in X/Y
    
    pixnm = Point(np.single(param['pixnm']),np.single(param['pixnm']), dtype=np.single)
    # requested reconstructed pixel size, usually ~5 nm
    # actually only pixnm.x is used
    
    pts = Point(int(param['xpts']),int(param['ypts']), dtype=np.int)
    # number of scan points in X and Y for raster scans
    
    zd = float(param['zd']) ## zone plate diameter
    dr = float(param['dr']) ## zone plate outer zone width
    e = float(param['e']) #photon energy
    ccdp = float(param['ccdp']) ##physical CCD pixel size, usually 30 microns
    ccdz = float(param['ccdz']) * 1000. ## sample to CCD distance, converted to microns
    nexp = int(param['nexp']) ## number of exposure times per point, 1 or 2
    t_exp = float(param['t_long']),float(param['t_short']) ##exposure times, t_short = 0 for single exposure
    st = float(param['s_threshold']) ## CCD saturation threshold
    nt = float(param['nl']) ##baseline noise level to subtract
    
    cropSize = int(param['sh_crop']) #size to which the raw data is cropped, typically 500-1000
    cropSize =960

    low_pass = int(param['low_pass']) ##flag to truncate the data with a gaussian

    #useBeamstop = int(param['useBeamstop']) ##flag to apply beamstop correction
    
    fw = float(param['filter_width']) ##width of the low pass filter (smoothing) of the data
    df = int(param['lowPassFilter']) ##flag to smooth the data prior to downsampling, filter with a gaussian

    # beamstop calculation
    bsNorm = 1. / float(param['beamStopTransmission']) ##normalization factor for the beamstop correction
    beamstopThickness = int(param['beamstop'])
    bsXshift = int(param['beamstopXshift'])
    bsYshift = int(param['beamstopYshift'])


    pThreshold = float(param['probeThreshold'])  ##intensity threshold on the averaged data for probe calculation, usually 0.1
    defocus = float(param['defocus']) ##amount to defocus the probe by after the initial estimate

    # post process calculations
    removeOutliers = int(param['removeOutliers'])  ##flag to remove outlier data 

    if nexp == 2:
        multi = True
    else: 
        multi = False
    
    cropSize =960
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
    
    
    log.emit = "Scan date: %s, Scan Number: %s, Scan ID: %s" %(scanDate, scanNumber, scanID)
    log.emit = "Output file: %s" %param['dataFile']
    log.emit = "Photon energy: %.2f eV" %e
    log.emit = "Wavelength: %.2f nm" %l
    log.emit = "Zone plate focal length: %.2f mm" %f
    log.emit = "sample-CCD distance: %.2f mm" %(ccdz / 1000.)
    log.emit = "Requested pixel size (nm): %.2f" %(pixnm.x)
    log.emit = "Will pad the data to: %i x %i (HxV)" %(sh_pad.x,sh_pad.y)
    log.emit = "Will downsample data to: %i x %i (HxV)" %(sh_out.x,sh_out.y)
    log.emit = "Probe step in pixels (x,y): %.2f,%.2f" %(ss.x / pixnm.x, ss.y / pixnm.y)
    log.emit = "Intensity threshold percent for probe calculation: %i" %(100. * pThreshold)
    log.emit = "Beamstop transmission: %.4f" %(1. / bsNorm)
    log.emit = "Beamstop normalization factor: %.2f" %bsNorm

    expectedFrames = pts.x * pts.y * nexp  - nexp * multi
    
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

    from glob import glob
    tifFiles = sorted(glob(scanDir+'*.tif'))
    indices = np.array([int(f.split('.')[0].split('image')[1]) for f in tifFiles])
    # when jumping a decimal, the sorting doesn't work properly
    k = np.argsort(indices)
    indices=indices[k]
    tifFiles=[tifFiles[j] for j in k]
    frame_offset = indices.min()
    assert indices.max()-frame_offset+1 == len(indices), 'File list is missing images'
    assert np.allclose(indices,np.arange(frame_offset,indices.max()+1)), 'File list is non-contigouus'
    #print indices - np.arange(frame_offset-1,indices.max())
    
    log.emit = "Starting frame is: %i" %(frame_offset)
    
    if multi: 
        fNorm = 1.

    ######################################################################################################
    ###load and average the background frames
    ###this alternates long and short frames regardless of "multi", should be fixed
    log.emit = "Averaging background frames..."
    
    bg, bg_short, nframes, wrng = aveDirFrames(bgScanDir, multi)
    
    log.emit = wrng
    
    log.emit = "Done. Averaged %i frames." % nframes


    ######################################################################################################
    ## Translation & data preparation
    pixIndex = [(i / pts.x, i % pts.x) for i in range(expectedFrames / nexp)]
    shiftData = [(-(float(pixIndex[i][0]) - float(pts.y - 1) / 2) * ss.y, (float(pixIndex[i][1]) - float(pts.x - 1) / 2) * ss.x) for i in range(expectedFrames / nexp)]
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
    if beamstopThickness == 20:
        bs_r = 31.  ##beamstop radius in pixels
    else: 
        bs_r = 37 * pixnm.x / 15.25 * e / 800.

    bs = 1. + ((bsNorm - 1) * circle(sh_out.x, bs_r, sh_out.x / 2, sh_out.x / 2))
    bs = np.roll(np.roll(bs, bsXshift, axis = 1), bsYshift, axis = 0)
    fNorm = ndimage.filters.gaussian_filter(bs, sigma = 0.5) #we multiply the data by this to normalize
    



    #### MPI DISTRIBUTION #################

    if multi:
        Ntiff = len(tifFiles)
        #assert Ntiff % 2 == 0, 'Help, odd number of tifs (%d) even though its double exposure' % Ntiff
        if Ntiff % 2 != 0:
            tifFiles=tifFiles[1:]
        Ntiff = len(tifFiles)
        ## resorted in tuples
        tiffs = [(tifFiles[i],tifFiles[i+1]) for i in range(0,Ntiff,2)]
    
    indices = range(len(tiffs))
    chunk_size = 4 * parallel.size


    tPrep = time.time() - t0
    t0 = time.time()
    
    tRead = 0
    tProcess = 0

    for start in range(0,len(tiffs),chunk_size):
        
        parallel.barrier()
        
        stop = min(start+chunk_size,len(tiffs)-1)
        log.emit = "Processing frames %d through %d." %(start,stop)  
        # indices to distribute
        chunk_indices = indices[start:stop]
        
        # MPI distribution keys
        parallel.loadmanager.reset()
        distributed = parallel.loadmanager.assign(chunk_indices)
         
        # node specific
        node_indices = [chunk_indices[j] for j in distributed[parallel.rank]]
        node_files = [tiffs[j] for j in node_indices]

        # processes without jobs should better now
        if len(node_files)==0:
            continue
        
        t0 = time.time()
        ### LOADING #########################
        if multi:
            short_files = []
            long_files = []
            ## order is long-short 
            for l,s in node_files:
                short_files.append(s)
                long_files.append(l)
            #print "Process %d loads %d frames" % (parallel.rank,len(long_files))    
            data = load_and_crop(long_files,dtype= np.float)
            data_short = load_and_crop(short_files,dtype= np.float)
        else:
            data = load_and_crop(node_files)
            data_short = None
                

        tcRead = time.time()-t0
        t0 = time.time()
        log.emit = "Read %d frames on %d processes in %.2f seconds" % (len(chunk_indices),parallel.size,tcRead)
    
    
        ### PROCESS THE DATA STACK ###########
       
        ## Assembling the frames ##
        
        # set up processor
        Pro = ProcessFCCDraw()
    
        # subtract background   
        data -= bg
        
        data = Pro.process_stack(data, verbose= verbose, do_overscan = True,do_spike_removal = True)
    
        if data_short is not None:
            
            data_short -= bg_short
            data_short = Pro.process_stack(data_short, verbose= verbose, do_overscan = True,do_spike_removal = True)
            
            smask = data < st
            
            # combine 
            data = data * smask + (1 - smask) * data_short * t_exp[0] / t_exp[1]
        
        ## Cropping / Rebinning ##
        
        if data.shape[-2] != cropSize or data.shape[-2] != cropSize:
            self.emit = "Raw data shape is %d x %d" % data.shape[-2:] + " and should be %d x %d."% (cropSize,cropSize)
            self.emit = "Non-rectangular diffraction frames may lead to unexpected results in the interpolation"
           
        if start == 0:
            ## center detection
            dataAve = data.sum(axis = 0) 
            
            # average dataset across all nodes
            parallel.allreduce(dataAve)
            
            # locate the center of mass of the average and shift to corner
            dataAve = dataAve * (dataAve > 10.) #threshold, 10 is about 1 photon at 750 eV
            
            cen = ce.utils.mass_center(dataAve)
            log.emit = "Found center at %s " % str(cen)
            
            # filter for smoothing the diffraction data
            if df:
                log.emit = "Applying filter on data transform"
                data = ce.utils.gf_2d(data, 2)
            
            
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
            probe_Fmask = (fave > pThreshold * fave.max())
            probe = np.fft.fftshift(np.fft.ifftn(np.sqrt(fave) * probe_Fmask)) 
            N = probe_Fmask.shape[0]
            probe_Rmask = circle(N, N / 4, N / 2, N / 2)
            
            # We can propagate this to some defocus distance if needed.
            """
            if defocus != 0:
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
        if low_pass:
            data *= np.exp(-dist(sh_out.x)**2 / 2. / (sh_out.x / 4)**2)
    
        # subtract a noise floor
        data -= nt
        data *= data > 0.
    
        ####Return (obsolete)
        if data[-1,:,:].mean() == 0.0: 
            print rank, tempData
            
        
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

    
    log.emit = "Writing data to file: %s" % outputPath

    ########################################################################################################
    ########################################################################################################

    cxiObj = cxi()
    cxiObj.process = param
    cxiObj.probe = probe
    cxiObj.beamline = '5.3.2.1'
    cxiObj.energy = e * 1.602e-19
    cxiObj.ccddata = dataStack
    cxiObj.probemask = probe_Fmask
    cxiObj.probeRmask = probe_Rmask
    cxiObj.datamean = dataAve
    cxiObj.illuminationIntensities = dataAve
    cxiObj.stxm = 'None'
    cxiObj.stxmInterp = 'None'
    cxiObj.xpixelsize = float(sh_pad.x) / float(sh_out.x) * float(ccdp) / 1e6
    cxiObj.ypixelsize = float(sh_pad.y) / float(sh_out.x) * float(ccdp) / 1e6
    cxiObj.corner_x, cxiObj.corner_y, cxiObj.corner_z = corner_position
    cxiObj.translation = translation
    cxiObj.indices = indices
    writeCXI(cxiObj, fileName = outputPath)

    ######################################################################################################
    ######################################################################################################

    log.emit = "Done."

    #os.chmod(outputFile, 0664)
    return 1
   
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
        
    
    
def load_and_crop(files,rows=960,gap=70,dtype= np.uint16):

    import warnings
    from tifffile import imread
    
    # load first:
    # The tiffs from the beamline are non-canonical, supress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = imread(files[0]).astype(dtype)
    
    N,cols = f.shape
        
    ##these slices take out the central empty band
    ##every frame should be (X,Y) = (960,1152)
    ##after removing overscan will be 960X960
    
    cut1 = N / 2 - rows/ 2 - gap / 2
    cut2 = N / 2 - gap / 2
    cut3 = N / 2 + gap / 2
    cut4 = N / 2 + gap / 2 + rows / 2
            
            
    def _crop(one):
        cropped = np.zeros((rows,cols),dtype=dtype)
        cropped[0:rows / 2,:]= one[cut1:cut2,:]
        cropped[rows / 2:rows,:]= one[cut3:cut4,:]
        return cropped
        
    frames = [_crop(f)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in files[1:]:
            frames.append(_crop(imread(f).astype(dtype)))
        
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
        
    
def aveDirFrames_David(bgScanDir, multi = False):
    
    from numpy import hstack, fliplr, flipud, zeros, sort, zeros_like
    from imageIO import imload

    n = 960 ##size of the data to extract in the x direction, take all vertically which contains overscan
    m = 68 ##size of the empty vertical band through the center

    bgFileList = sorted(os.listdir(bgScanDir))
    bgFileList = [bgFileList[i] for i in range(len(bgFileList)) if bgFileList[i].count('tif')]
    longExpList = bgFileList[1::2]
    shortExpList = bgFileList[0::2]
    npoints = len(shortExpList)
    nFrames = 0
    for a,b,i in zip(longExpList,shortExpList,xrange(npoints)):
        if i == 0:
            bg = imload(bgScanDir + a).astype('single')
            bgNorm = zeros(bg.shape)
            bgNorm[bg > 0.] = 1.
            if multi:
                bgShort = imload(bgScanDir + b).astype('single')
                bgShortNorm = zeros(bgShort.shape)
                bgShortNorm[bgShort > 0.] = 1.
            nFrames += 1
        else:
            temp = imload(bgScanDir + a).astype('single')
            tempShort = imload(bgScanDir + b).astype('single')
            bg += temp
            bgNorm[temp > 0.] += 1
            if multi:
                bgShort += tempShort
                bgShortNorm += tempShort > 0.
            else:
                bg += tempShort
                bgNorm += tempShort > 0.
            nFrames += 2
    bgNorm[bgNorm == 0.] = 1.
    bg = bg / bgNorm
    if multi:
        bgShortNorm[bgShortNorm == 0.] = 1.
        bgShort = bgShort / bgShortNorm
    else: bgShort = None

    ny, nx = bg.shape
    bgCrop = zeros((n, nx), dtype = 'uint16')
    bgShortCrop = zeros((n, nx), dtype = 'uint16')
        
    ##these slices take out the central empty band
    ##every frame should be (X,Y) = (960,1152)
    ##after removing overscan will be 960X960
    bgCrop[0:n / 2,:]= bg[ny / 2 - n / 2 - m / 2:ny / 2 - m / 2,:]
    bgCrop[n / 2:n,:]= bg[ny / 2 + m / 2:ny / 2 + m / 2 + n / 2,:]
    
    if multi:
        bgShortCrop[0:n / 2,:]= bgShort[ny / 2 - n / 2 - m / 2:ny / 2 - m / 2,:]
        bgShortCrop[n / 2:n,:]= bgShort[ny / 2 + m / 2:ny / 2 + m / 2 + n / 2,:]
    
    return bgCrop.astype('float'), bgShortCrop.astype('float'), nFrames


