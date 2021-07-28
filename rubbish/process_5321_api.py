import time, os, sys

from cosmic.utils import dist, shift, circle, e2l

from scipy import random, ndimage, interpolate

from scipy import polyfit, misc, ndimage, mgrid, rand

from processFCCD import ProcessFCCDraw # fitOverscan, processfCCDFrame

import numpy as np

import cosmic.ext as ce

from cosmic.io import cxi,writeCXI

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
        


def process(param, verbose = True):

    ### IO #####
    verbose = (verbose and parallel.master)
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
 

    sh_sample = int(param['sh_sample_y']),int(param['sh_sample_x']) #size to which the raw data is resampled, usually 256x256
    fw = float(param['filter_width']) ##width of the low pass filter (smoothing) of the data
    ssy, ssx = float(param['ssy']),float(param['ssx']) ##scan step size in X/Y
    xpts = int(param['xpts']) ##number of X scan points
    ypts = int(param['ypts']) ##number of Y scan points
    zd = float(param['zd']) ## zone plate diameter
    dr = float(param['dr']) ## zone plate outer zone width
    e = float(param['e']) #photon energy
    ccdp = float(param['ccdp']) ##physical CCD pixel size, usually 30 microns
    ccdz = float(param['ccdz']) * 1000. ## sample to CCD distance, converted to microns
    nexp = int(param['nexp']) ## number of exposure times per point, 1 or 2
    t_exp = float(param['t_long']),float(param['t_short']) ##exposure times, t_short = 0 for single exposure
    st = float(param['s_threshold']) ## CCD saturation threshold
    xpixnm,ypixnm = np.single(param['pixnm']),np.single(param['pixnm']) ##requested reconstructed pixel size, usually ~5 nm
    pThreshold = float(param['probeThreshold'])  ##intensity threshold on the averaged data for probe calculation, usually 0.1
    removeOutliers = int(param['removeOutliers'])  ##flag to remove outlier data 
    nt = float(param['nl']) ##baseline noise level to subtract
    cropSize = int(param['sh_crop']) #size to which the raw data is cropped, typically 500-1000
    verbose = True
    low_pass = int(param['low_pass']) ##flag to truncate the data with a gaussian
    bsNorm = 1. / float(param['beamStopTransmission']) ##normalization factor for the beamstop correction
    defocus = float(param['defocus']) ##amount to defocus the probe by after the initial estimate
    beamstopThickness = int(param['beamstop'])
    #useBeamstop = int(param['useBeamstop']) ##flag to apply beamstop correction
    df = int(param['lowPassFilter']) ##flag to smooth the data prior to downsampling, filter with a gaussian
    bsXshift = int(param['beamstopXshift'])
    bsYshift = int(param['beamstopYshift'])


    if nexp == 2:
        multi = True
    else: multi = False
    
    ######################################################################################################
    ###calculate the number of pixels to use for raw data (the "crop" size)
    nyn, nxn = sh_sample  ##final size of the data array, usually 256x256
    l = e2l(e) ##wavelength in nm
    f = zd * dr / l #zp focal length microns
    na = zd / f / 2. #zp numerical aperture
    xtheta = l / 2. / xpixnm #scattering angle
    ##this is the X size of the array (before downsampling, physical CCD pixels size) needed to get the requested pixel size
    nx = 2 * np.round(ccdz * (xtheta) / ccdp) 
    ytheta = l / 2. / ypixnm
    ##this is the Y size of the array (before downsampling, physical CCD pixel size) needed to get the requested pixel size
    ny = 2 * np.round(ccdz * (ytheta) / ccdp)
    sh_pad = ny, nx  #number of pixels (at physical CCD pixel size) before downsampling required for the requested pixel size
    cropSize =960
    
    if verbose:
        print "Scan date: %s, Scan Number: %s, Scan ID: %s" %(scanDate, scanNumber, scanID)
        print "Output file: %s" %param['dataFile']
        print "Photon energy: %.2f eV" %e
        print "Wavelength: %.2f nm" %l
        print "Zone plate focal length: %.2f mm" %f
        print "sample-CCD distance: %.2f mm" %(ccdz / 1000.)
        print "Requested pixel size (nm): %.2f" %(xpixnm)
        print "Will pad the data to: %i x %i (HxV)" %(nx,ny)
        print "Will downsample data to: %i x %i (HxV)" %(nxn,nyn)
        print "Probe step in pixels (x,y): %.2f,%.2f" %(ssx / xpixnm, ssy / ypixnm)
        print "Intensity threshold percent for probe calculation: %i" %(100. * pThreshold)
        print "Beamstop transmission: %.4f" %(1. / bsNorm)
        print "Beamstop normalization factor: %.2f" %bsNorm

    expectedFrames = xpts * ypts * nexp  - nexp * multi
    if verbose: 
        print "Expecting %i frames." %(expectedFrames)    
    
    t0=time.time()
    ######################################################################################################
    ###load the list of tiff file names
    """
    fileList = sorted(os.listdir(scanDir))
    tifFiles = [i for i in range(len(fileList)) if fileList[i].count('tif')]
    indices = [int(fileList[tifFiles[i]].split('.')[0].split('image')[1]) for i in range(len(tifFiles))]
    frame_offset = np.array(indices).min()# + 1
    #if nexp == 2: frame_offset = 2 ###JUST TESTING!!!
    """
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
    if verbose: 
        print "Starting frame is: %i" %(frame_offset)
    
    if multi: 
        fNorm = 1.

    ######################################################################################################
    ###load and average the background frames
    ###this alternates long and short frames regardless of "multi", should be fixed
    if verbose: 
        print "Averaging background frames..."
    
    bg, bg_short, nframes = aveDirFrames(bgScanDir, multi)
    
    if verbose: 
        print "Done. Averaged %i frames." % nframes


    ######################################################################################################
    ###set up the frame shift data
    ## SIMPLIFY
    pixIndex = [(i / xpts, i % xpts) for i in range(expectedFrames / nexp)]
    shiftData = [(-(float(pixIndex[i][0]) - float(ypts - 1) / 2) * ssy, (float(pixIndex[i][1]) - float(xpts - 1) / 2) * ssx) for i in range(expectedFrames / nexp)]

    ######################################################################################################
    ###set up the low pass filter and beamstop normalization
    ##beamstop
    if beamstopThickness == 20:
        bs_r = 31.  ##beamstop radius in pixels
    else: 
        bs_r = 37 * xpixnm / 15.25 * e / 800.
    y,x = sh_sample
    bs = 1. + ((bsNorm - 1) * circle(x, bs_r, x / 2, x / 2))
    bs = np.roll(np.roll(bs, bsXshift, axis = 1), bsYshift, axis = 0)
    fNorm = ndimage.filters.gaussian_filter(bs, sigma = 0.5) #we multiply the data by this to normalize
    
    ##filter for smoothing the diffraction data
    if df:
        x,y = np.meshgrid(np.abs(np.arange(cropSize) - cropSize / 2),np.abs(np.arange(cropSize) - cropSize / 2))
        datafilter = np.exp(-x**2 / 2. / np.float(fw)**2 -\
                     y**2 / 2. / np.float(fw)**2)
        datafilter = np.fft.fftshift(datafilter).astype("float64")
    else: 
        datafilter = False

    tPrep = time.time() - t0
    t0 = time.time()

    ######################################################################################################
    ###split up the data stack into chunks and send to the processors
    """
    if comm.Get_size() > 1: 
        nProcessors = comm.Get_size() - 1
    else: 
        print "readALSMPI requires at least 2 MPI processes. Exiting."; return
       
    R = int(expectedFrames / nProcessors) ##ratio of jobs to processors rounded down to nearest integer
    Y = expectedFrames - nProcessors * R ##number of processors with R + 1 jobs
    X = nProcessors * (R + 1) - expectedFrames  ##number of processors with R jobs
              
    i1, i2 = 0, 0    
    for rank in range(1,comm.Get_size()):
        if rank < Y + 1:
            i2 = i1 + R + 1 ##for R+1 jobs, first Y processors
        else:
            i2 = i1 + R  ##for R jobs on all remaining processor
        nSlices = i2 - i1
        indexTuple = i1, i2
        i1 = i2
    """
    #### MPI DISTRIBUTION #################
    if multi:
        Ntiff = len(tifFiles)
        #assert Ntiff % 2 == 0, 'Help, odd number of tifs (%d) even though its double exposure' % Ntiff
        if Ntiff % 2 != 0:
            tifFiles=tifFiles[1:]
        Ntiff = len(tifFiles)
        ## resorted in tuples
        tiffs = [(tifFiles[i],tifFiles[i+1]) for i in range(0,Ntiff,2)]
    
    distributed = parallel.loadmanager.assign(tiffs) # doesn't work for double exposure
    node_hash_indices = distributed[parallel.rank]
    node_files = [tiffs[j] for j in node_hash_indices]
    #node_tif_indices = np.array([int(f.split('.')[0].split('image')[1]) for f in node_files])
    
    #####################################################################################################
    if multi:
        short_files = []
        long_files = []
        ## order is long-short 
        for l,s in node_files:
            short_files.append(s)
            long_files.append(l)
        print "Process %d loads %d frames" % (parallel.rank,len(long_files))    
        data = load_and_crop(long_files,dtype= np.float)
        data_short = load_and_crop(short_files,dtype= np.float)
        if parallel.master:
            print data.shape
    else:
        data = load_and_crop(node_files)
        data_short = None
        
    ###############################################################################################
    ###define the array for the processed data
    
    #dataStack, loadedFrames = loadDirFrames(scanDir, filePrefix, expectedFrames, cropSize, multi, fCCD, frame_offset + iStart, verbose = verbose)
    #data = np.zeros((len(dataStack), cropSize, cropSize))
    parallel.barrier()
    tRead = time.time()-t0
    t0 = time.time()
    if verbose:
        print "Read %d frames on %d processes in %.2f seconds" % (len(tifFiles),parallel.size,tRead)


    ## PROCESS THE DATA STACK ###########
    #data = processFCCDstack(data, data_short, bg, bg_short, t_exp, st,verbose=verbose)
    
    # set up processor
    # subtract background
    Pro = ProcessFCCDraw()
    
    data -= bg
    data = Pro.process_stack(data, verbose= verbose, do_overscan = True,do_spike_removal = True)

    if data_short is not None:
        
        data_short -= bg_short
        data_short = Pro.process_stack(data_short, verbose= verbose, do_overscan = True,do_spike_removal = True)
        
        smask = data < st
        
        # combine 
        data = data * smask + (1 - smask) * data_short * t_exp[0] / t_exp[1]
        
    ###############################################################################################
    #if verbose:
    if verbose: 
        sys.stdout.write("Processing frames...");sys.stdout.flush()


    ###############################################################################################
    ####Low Pass Filter
    #if df:
    if False:
        import fftw3
        if verbose: print "Applying filter on data transform"

        data = data * (1.0 +0.j)

        #############Do it this way to reuse the plan, requires some data copies
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

        data = np.abs(data).astype(np.float32) / (data[0].size) #renormalize after transforms

    dxn = np.int(np.float(nxn) / np.float(nx) * np.float(cropSize))
    dyn = dxn
    Nframes,sh0,sh1 = data.shape

    ##this is the downsampled datashape.  Still need to crop after recentering.
    
    interpolated = []
    
    ##bivariate spline interpolation
    x, y = np.linspace(0, sh0, sh0),np.linspace(0, sh1, sh1)
    yn, xn = np.meshgrid(np.linspace(0, sh1, dyn), np.linspace(0, sh0, dxn))
    
    for d in data:
        spline = interpolate.RectBivariateSpline(x, y, d)
        interpolated.append(spline.ev(xn.flatten(), yn.flatten()).reshape(dyn,dxn))

    ## new interpolated data
    data = np.asarray(interpolated)
    
    ####truncate with a gaussian if requested
    if low_pass:
        data *= np.exp(-dist(dyn)**2 / 2. / (dyn / 4)**2)

    ###subract a noise floor
    data -= nt
    data *= data > 0.

    ####Return
    if data[-1,:,:].mean() == 0.0: 
        print rank, tempData
        
    ## ok what is happening here??
    data = np.transpose(np.rot90(np.transpose(data)))
    
    ######################################################################################################
    ###retrieve the chunks back and put them into the right spot in the stack (re-initialized to the new smaller size)
        
    nPoints = xpts * ypts
    dataStack = np.zeros((nPoints, dyn, dxn))
        
    dataStack[distributed[parallel.rank]]= data
    parallel.allreduce(dataStack) 
    
    # let the other processes quit
    if not parallel.master:  
        return 
 
    tProcess = time.time() - t0
    if verbose: print "Done!"
    

    #if verbose: print "Processed frames at: %.2f Hz" %(float(loadedFrames / nexp) / tProcess)
    if verbose: 
        print "Total time per frame : %i ms" %(1000. * (tProcess + tPrep + tRead) / float(expectedFrames) * parallel.size)
            
    if verbose: 
        millis = tuple(1000. * np.array([tPrep, tRead, tProcess]) / float(expectedFrames) * parallel.size)
        print "Distribution (Prep, Load, Process): (%.2f | %.2f | %.2f ) ms" % millis


    ######################################################################################################
    ###Calculate the STXM image and make some arrays for the translations
    ihistMask = dataStack.sum(axis = 0)
    ihistMask = ihistMask > 0.1 * ihistMask.max()
    ihist = np.array([(dataStack[i] * ihistMask).sum() for i in range(nPoints)])
    indx = np.arange(nPoints)
    stxmImage = np.reshape(ihist,(ypts,xpts))[::-1,:]
    
    #####################################################################################################
    ###Remove outliers based on STXM intensities
    x = np.array(shiftData)[:,1]
    y = np.array(shiftData)[:,0]

    """
    if removeOutliers:
        #gy, gx = gradient(stxmImage)
        gy = stxmImage - ndimage.filters.gaussian_filter(stxmImage, sigma = 0.25)
        gy = gy[::-1, :].flatten()  ##puts it in the same ordering as ccddata, starting lower left
        delta = 8. * gy.std()
        badIndices = np.where(gy < (gy.mean() - delta))[0]  ##the min Y gradient is one row below the bad pixel
        ihistMask = 1 - ihistMask
        ihist = np.array([(dataStack[i] * ihistMask).sum() for i in range(nPoints)])
        noiseIndices = np.where(ihist > (ihist.mean() + 2. * ihist.std()))
        badIndices = np.unique(np.append(badIndices,noiseIndices))
        stxmImage = stxmImage[::-1, :].flatten()
        k = 0
        if len(badIndices) > 0:
            for item in badIndices:
                stxmImage[item] = (stxmImage[item + 1] + stxmImage[item - 1]) / 2.
                if indx[item] > 0:
                    indx[item] = 0
                    indx[item + 1:nPoints] = indx[item + 1:nPoints] - 1
                else:
                    indx[item] = 0
                x = np.delete(x, item - k)
                y = np.delete(y, item - k)
                dataStack = np.delete(dataStack, item - k, axis=0)
                k += 1
        stxmImage = np.reshape(stxmImage, (ypts, xpts))[::-1, :]
        if verbose:
            print "Removed %i bad frames." % (len(badIndices))
    """
    translationX, translationY = (x - x.min()) * 1e-9, (y - y.min()) * 1e-9

    ######################################################################################################
    ###Calculate the probe, center of mass and probe mask
    if verbose: 
        print "Calculating probe and mask"
    
    dataAve = dataStack.sum(axis = 0) #average dataset
    #locate the center of mass of the average and shift to corner
    dataAve = dataAve * (dataAve > 10.) #threshold, 10 is about 1 photon at 750 eV
    dataSum = dataAve.sum() #sum for normalization
    xIndex = np.arange(dataAve.shape[1])
    yIndex = np.arange(dataAve.shape[0])

    xc = np.int(np.round((dataAve.sum(axis = 0) * xIndex).sum() / dataSum))
    yc = np.int(np.round((dataAve.sum(axis = 1) * yIndex).sum() / dataSum))
    if verbose:
        print "Center positions: %.2f, %.2f" %(xc,yc)

    ##Calculate the probe from the average diffraction pattern
    dataStack = np.roll(np.roll(dataStack, dataStack.shape[1] / 2 - yc, axis = 1), dataStack.shape[2] / 2 - xc, axis = 2)  ##center the data
    ##zero the center
    #j,k,l = dataStack.shape
    #dataStack = dataStack * (1. - circle(k, 9, k / 2, k / 2))
    dataAve = np.roll(np.roll(dataAve, -yc, axis = 0), -xc, axis = 1) ##center the average data
    #dataAve = dataAve * (1. - circle(k, 9, k / 2, k / 2))
    pMask = (dataAve > pThreshold * dataAve.max())
    p = np.sqrt(dataAve) * pMask
    p = np.fft.ifftshift(np.fft.ifftn(p)) #we can propagate this to some defocus distance if needed.
    ###put propagation code here
    if defocus != 0:
        from propagate import propnf
        p = propnf(p, defocus * 1000. / xpixnm, l / xpixnm)

    ######################################################################################################
    ###Apply a mask to the data
    # ny,nx = dataStack.shape[1:3]
    # m = np.ones((ny,nx))# - circle(nx, 2, nx / 2, nx / 2)
    # m[:,190:nx] = 0.
    # m[175:205,140:nx] = 0.
    # dataStack *= m

    ##########################################################
    ####interpolate STXM image onto the same mesh as the reconstructed image
    if verbose: print "Interpolating STXM image"
    yr,xr = ssy * ypts, ssx * xpts
    y,x = np.meshgrid(np.linspace(0,yr,ypts),np.linspace(0,xr,xpts))
    y,x = y.transpose(), x.transpose()
    yp,xp = np.meshgrid(np.linspace(0,yr, ypts * (ssy / ypixnm)),np.linspace(0,xr, xpts * (ssx / xpixnm)))
    yp,xp = yp.transpose(), xp.transpose()
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[0,1] - x0
    dy = y[1,0] - y0
    ivals = (xp - x0)/dx
    jvals = (yp - y0)/dy
    coords = np.array([ivals, jvals])
    stxmImageInterp = ndimage.map_coordinates(stxmImage.transpose(), coords)
    stxmImageInterp = stxmImageInterp * (stxmImageInterp > 0.)

    stride_x = 1
    stride_y = 1
    start_x = 0
    start_y = 0
    end_x = xpts
    end_y = ypts

    ######################################################################################################
    ###Generate the CXI file
    translation = np.column_stack((translationX, translationY, np.zeros(translationY.size))) #(x,y,z) in meters
    corner_position = [float(sh_pad[1]) * ccdp / 2. / 1e6, float(sh_pad[0]) * ccdp / 2. / 1e6, ccdz / 1e6] #(x,y,z) in meters

    outputFile = param['scanDir'] + param['outputFilename']
    if verbose: print "Writing data to file: %s" %outputFile

    ########################################################################################################
    ########################################################################################################

    cxiObj = cxi()
    cxiObj.process = param
    cxiObj.probe = p
    cxiObj.beamline = '5.3.2.1'
    cxiObj.energy = e * 1.602e-19
    cxiObj.ccddata = dataStack
    cxiObj.probemask = pMask
    cxiObj.probeRmask = circle(pMask.shape[0], pMask.shape[0] / 4, pMask.shape[0] / 2, pMask.shape[0] / 2)
    cxiObj.datamean = dataAve
    cxiObj.illuminationIntensities = dataAve
    cxiObj.stxm = stxmImage
    cxiObj.stxmInterp = stxmImageInterp
    cxiObj.xpixelsize = float(sh_pad[1]) / float(nxn) * float(ccdp) / 1e6
    cxiObj.ypixelsize = float(sh_pad[0]) / float(nxn) * float(ccdp) / 1e6
    cxiObj.corner_x, cxiObj.corner_y, cxiObj.corner_z = corner_position
    cxiObj.translation = translation
    cxiObj.indices = indx
    writeCXI(cxiObj, fileName = outputFile)

    ######################################################################################################
    ######################################################################################################

    if verbose: print "Done."

    #os.chmod(outputFile, 0664)
    return 1



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
            print "Warning : not the same number of short and long background exposures"
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
        return res, None, len(bg)
    else:
        bg = load_and_crop(shortExpList)
        norm = (bg > 0).astype(np.float64).sum(0)
        norm[norm == 0.0]=1.0
        return res, bg.sum(0) / norm, len(bg)
        
    
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


