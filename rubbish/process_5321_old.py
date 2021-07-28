import time, os, sys

from cosmic.utils import dist, shift, circle, e2l

from scipy import random, ndimage, interpolate

from scipy import polyfit, misc, ndimage, mgrid, rand

from processFCCD import fitOverscan, processfCCDFrame

import numpy as np

import cosmic.ext as ce

from cosmic.io import cxi,writeCXI

parallel = ce.utils.parallel

#imread = ce.tifffile.imread

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
        

    if nexp == 2:
        multi = True
    else: multi = False

def process(param, verbose = True):

    ### IO #####
   
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
 
    """
    ######################################################################################################
    ###setup some of the file name prefixes
    if not(int(param['sim'])):
        scanDir = param['scanDir']
        if fCCD:
            filePrefix = 'image'
            namePrefix = 'NS_'
        else:
            if param['bl'] == 'ns':
                namePrefix = 'NS_'
                filePrefix = namePrefix + scanDate + scanNumber + '_'
            if param['bl'] == 'mes':
                namePrefix = '11_'
                filePrefix = namePrefix + scanDate + scanNumber + '_'
            if param['bl'] == 'K':
                namePrefix = 'K'
                filePrefix = namePrefix + scanDate + scanNumber + '_'
    else:
        scanDir = dataPath
        if not(int(param['saveFile'])): 
            saveFile = param['dataFile']
        else: 
            saveFile = param['saveFile']
        
        filePrefix = param['filePrefix']

    if bgScanDate == '0': 
        bgScanDate = scanDate
    if bgScanNumber != '0':
        bg_subtract = True
        bgScanDir = param['bgScanDir']
        bg_prefix = 'NS_' + bgScanDate + bgScanNumber + '_'
    else: 
        bg_subtract = False
    """
    
    """
    ######################################################################################################
    
    interferometer = int(param['interferometer'])
    xaxis = param['xaxis']
    yaxis = param['yaxis']
    xaxisNo = str(int(scanID) - 1)
    yaxisNo = str(int(scanID) - 1)
    
    ###load interferometer data if requested
    if interferometer:
        xaxisDataFile = namePrefix + scanDate + scanNumber + '_' + xaxis + xaxisNo + '.xim'
        yaxisDataFile = namePrefix + scanDate + scanNumber + '_' + yaxis + yaxisNo + '.xim'
        print "X Axis data file: ", xaxisDataFile
        print "Y Axis data file: ", yaxisDataFile
        xdataFile = dataPath + scanDate + '/' + xaxisDataFile
        ydataFile = dataPath + scanDate + '/' + yaxisDataFile
        try: xshiftData = readXIM(xdataFile)
        except:
            print "Could not open interferometer X data file: %s" %xdataFile
            interferometer = 0
        else: xshiftData = (xshiftData - xshiftData.mean()) * 1000. #convert to nanometers
        try: yshiftData = readXIM(ydataFile)
        except:
            print "Could not open interferometer Y data file: %s" %ydataFile
            interferometer = 0
        else: yshiftData = (yshiftData - yshiftData.mean()) * 1000. #convert to nanometers

    """

    """
    p_sh_sample = int(param['sh_sample_y']),int(param['sh_sample_x']) #size to which the raw data is resampled
    filter_width = float(param['filter_width'])
    ss = float(param['ssy']),float(param['ssx'])
    xpts = int(param['xpts'])
    ypts = int(param['ypts'])
    zd = float(param['zd'])
    dr = float(param['dr'])
    e = float(param['e']) #photon energy
    ccdp = float(param['ccdp'])
    ccdz = float(param['ccdz']) * 1000. #convert to microns
    nexp = int(param['nexp'])
    bin = int(param['bin'])
    xc = int(param['xcenter'])
    yc = int(param['ycenter'])
    t_exp = float(param['t_long']),float(param['t_short'])
    saturation_threshold = float(param['s_threshold'])
    nProcesses = int(param['nProcesses'])
    bl = param['bl']
    xpixnm,ypixnm = np.single(param['pixnm']),np.single(param['pixnm'])

    pThreshold = float(param['probeThreshold'])
    removeOutliers = int(param['removeOutliers'])
    dfilter = int(param['lowPassFilter'])
    nl = float(param['nl'])
    cropSize = int(param['sh_crop']) #size to which the raw data is cropped
    noise_threshold = nl
    process = 15
    verbose = False
    fstep = 1
    tiff = 0
    logscale = 1
    indexd = None
    indexo = None
    gpu = int(param['processGPU'])
    fv = int(param['fv'])
    fh = int(param['fh'])
    tr = int(param['transpose'])
    fCCD_version = int(param['fCCD_version'])
    low_pass = int(param['low_pass'])
    multiCore = True
    bsNorm = 1. / float(param['beamStopTransmission'])
    defocus = float(param['defocus'])
    bsXshift = int(param['beamstopXshift'])
    bsYshift = int(param['beamstopYshift'])
    beamstopThickness = int(param['beamstop'])


    if nexp == 2:
        multi = True
    else: 
        multi = False
    """


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
    xpixnm,ypixnm = single(param['pixnm']),single(param['pixnm']) ##requested reconstructed pixel size, usually ~5 nm
    pThreshold = float(param['probeThreshold'])  ##intensity threshold on the averaged data for probe calculation, usually 0.1
    removeOutliers = int(param['removeOutliers'])  ##flag to remove outlier data 
    nt = float(param['nl']) ##baseline noise level to subtract
    cropSize = int(param['sh_crop']) #size to which the raw data is cropped, typically 500-1000
    verbose = True
    low_pass = int(param['low_pass']) ##flag to truncate the data with a gaussian
    bsNorm = 1. / float(param['beamStopTransmission']) ##normalization factor for the beamstop correction
    defocus = float(param['defocus']) ##amount to defocus the probe by after the initial estimate
    beamstopThickness = int(param['beamstop'])
    useBeamstop = int(param['useBeamstop']) ##flag to apply beamstop correction
    df = int(param['lowPassFilter']) ##flag to smooth the data prior to downsampling, filter with a gaussian
    bsXshift = int(param['beamstopXshift'])
    bsYshift = int(param['beamstopYshift'])
    ######################################################################################################
    ###calculate the number of pixels to use for raw data (the "crop" size)
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
    if verbose: print "Expecting %i frames." %(expectedFrames)    
    """
    nyn, nxn = p_sh_sample
    ssy, ssx = ss
    data_ave = np.zeros((nyn,nxn))
    l = e2l(e)
    f = zd * dr / l #zp focal length microns
    na = zd / f / 2. #zp numerical aperture
    xtheta = l / 2. / xpixnm #scattering angle
    nx = 2 * round(ccdz * (xtheta) / ccdp)
    ytheta = l / 2. / ypixnm
    ny = 2 * round(ccdz * (ytheta) / ccdp)
    sh_pad = ny, nx  #size to which the raw data is zero padded

    if parallel.master:
        verbose = True

    if verbose:
        if fCCD: print "Using fastCCD version: %s" %fCCD_version
        else: print "Not using fastCCD"
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

    zpw = round(2. * na * ccdz / ccdp)
    stxmImage = np.zeros((ypts, xpts))
    stxmMask = circle(nxn,zpw * float(nxn) / float(nx),nxn/2,nxn/2)

    expectedFrames = xpts * ypts * bin * nexp# - nexp * fCCD
    if verbose: print "Expecting %i frames." %(expectedFrames)
    """
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
    if bg_subtract:
        if verbose: print "Averaging background frames..."
        if param['bg_filename'] == '':
            bg, bg_short, nframes = aveDirFrames(bgScanDir, multi, fCCD, fCCD_version)
        else:
            from cosmic.ext.piutils import readSpe
            
            bg = np.asarray(readSpe(scanDir + param['bg_filename'])['data'])
            nframes = len(bg) / nexp
            bg_short = bg[1::2].mean(axis = 0)
            bg = bg[2::2].mean(axis = 0)
        if verbose: print "Done. Averaged %i frames." %nframes

        osmask = np.zeros(bg.shape)
        sh = bg.shape
        if fCCD_version == 1:
            osmask[:,0::12] = 1
            osmask[:,1::12] = 1
        elif fCCD_version == 2:
            #osmask[:,0::12] = 1
            #osmask[:,11::12] = 1
            osmask[0:sh[0]/2,0::12] = 1
            osmask[0:sh[0]/2,11::12] = 1
            osmask[sh[0]/2:sh[0],0::12] = 1
            osmask[sh[0]/2:sh[0],11::12] = 1
            #osmask = vstack((flipud(fliplr(osmask[:,1152:2304])),osmask[:,0:1152]))
        #bg = vstack((flipud(fliplr(bg[:,1152:2304])),bg[:,0:1152]))
        osmask = np.vstack((osmask[1:467,:],osmask[534:1000,:]))
        #bg = np.vstack((bg[1:467,:],1. * bg[534:1000,:]))
        dmask = 1 - osmask
        indexd = np.where(dmask)
        indexo = np.where(osmask)
        #bg = reshape(bg[indexd],(932,960))
        os_bkg = 0.
    else:
        bg = None
        bg_short = None
        if verbose: print "No background data submitted!"

    t0 = time.time()


    ######################################################################################################
    ###set up the frame shift data
    pixIndex = [(i / xpts, i % xpts) for i in range(expectedFrames / nexp)]
    if interferometer:
        shiftData = [(-yshiftData[i / xpts, i % xpts] * 1000., xshiftData[i / xpts, i % xpts] * 1000.) for i in range(xpts * ypts)]
    else:
        shiftData = [(-(float(pixIndex[i][0]) - float(ypts - 1) / 2) * ssy, (float(pixIndex[i][1]) - float(xpts - 1) / 2) * ssx) for i in range(expectedFrames / nexp)]

    ######################################################################################################
    ###set up the low pass filter and beamstop normalization
    fsh = cropSize, cropSize
    if beamstopThickness == 20:
        bs_r = 31.
    else: bs_r = 40 #was 10 at BL901

    bs = 1. + (bsNorm * circle(cropSize, bs_r, cropSize / 2, cropSize / 2) - (bsNorm + 0.5) * circle(cropSize, 10, cropSize / 2, cropSize / 2))
    bs = 1. + ((bsNorm - 1) * circle(cropSize, bs_r, cropSize / 2, cropSize / 2))
    fNorm = shift(ndimage.filters.gaussian_filter(bs, sigma = 0.5), bsYshift, bsXshift) #we multiply the data by this to normalize

    
    if dfilter:
        x,y = np.meshgrid(abs(np.arange(fsh[1]) - fsh[1] / 2),abs(np.arange(fsh[0]) - fsh[0] / 2))
        filter = np.exp(-x**2 / 2. / float(fw)**2 -\
                     y**2 / 2. / float(fw)**2)
        filter = np.fft.fftshift(filter).astype("float64")
    else: 
        filter = False

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
    distributed = parallel.loadmanager.assign(tifFiles) # doesn't work for double exposure
    node_hash_indices = distributed[parallel.rank]
    node_files = [tifFiles[j] for j in node_hash_indices]
    node_tif_indices = np.array([int(f.split('.')[0].split('image')[1]) for f in node_files])
    #####################################################################################################

    ny, nx = sh_pad
    nyn, nxn = p_sh_sample
    df = dfilter
    nl = noise_threshold

    ###############################################################################################
    ###define the array for the processed data
    from tifffile import imread
    dataStack = np.array([imread(f).astype('uint16')[500:1500,:] for f in node_files])
    
    #dataStack, loadedFrames = loadDirFrames(scanDir, filePrefix, expectedFrames, cropSize, multi, fCCD, frame_offset + iStart, verbose = verbose)
    #data = np.zeros((len(dataStack), cropSize, cropSize))
    parallel.barrier()
    tRead = time.time()-t0
    t0 = time.time()
    if parallel.master:
        print "Read %d frames on %d processes in %.2f seconds" % (len(tifFiles),parallel.size,tRead)

    ######################################################################################################
    ###Generate a mask to cover bad regions of the detectors
    dataMask = np.ones_like(dataStack[0])
    if fCCD:
        dataMask[500:1000,0:150] = 0.
    dataStack = dataStack * dataMask

    ###############################################################################################
    ###for fastCCD data launch the fCCD pre-processor for overscan removal and background subtraction
    data=[]
    if fCCD:        
        if multi:
            for i in range(len(dataStack)):
                data.append(processfCCDFrame(dataStack[i,:,:], dataStack[i + 1,:,:], bg, bg_short, os_bkg,\
                    indexd, indexo, t_exp, saturation_threshold,noise_threshold, xc,yc,fNorm,i,cropSize,fCCD_version))
                i += 1
        else:
            for i in range(len(dataStack)):
                data.append(processfCCDFrame(dataStack[i,:,:], None, bg, bg_short, os_bkg, indexd, indexo, t_exp, \
                    saturation_threshold, noise_threshold, xc, yc, fNorm, i, cropSize, fCCD_version))
    
    else:
        data = dataStack
    del(dataStack)
    
    data = np.asarray(data)
    nSlices = len(data)
    ###############################################################################################
    #if verbose:
    if verbose: 
        sys.stdout.write("Processing frames...");sys.stdout.flush()

    ###############################################################################################
    ###background subtraction is already done so just remove the final "noise level" offset
    if fCCD:
        ###redefine the center since the fastCCD preprocessor centers the data
        y,x = data[0].shape
        xc,yc = data[0].shape[1] / 2, data[0].shape[0] / 2

    ##############################################################################################
    ###Background subtract and merge for non-fastCCD data
    elif multi:
        nSlices = nSlices / 2
        #data = dataStruct[0]
        data[0::2,:,:] -= bg ##subtract background from long exposures
        data[0::2,:,:] *= (data[0::2,:,:] > nl)
        data[1::2,:,:] -= bg_short ##subtract background from short exposures
        data[1::2,:,:] *= (data[1::2,:,:] > nl)
        mask = (data[0] > saturation_threshold)
        mask = 1. - mask / mask.max()
        data[0::2,:,:] = mask * data[0::2,:,:] + (1 - mask) * data[1::2,:,:] * t_exp[0] / t_exp[1]
        data = data[0::2,:,:]

    else:
        #data = dataStruct[0]
        data -= (bg + nl)
        data *= data > 0.

    dy, dx = float(nyn) / float(ny),float(nxn) / float(nx)
    dyn, dxn = int(float(y) * dy), int(float(x) * dx)
    data -= nl
    data = data * (data > 0.)

    ###############################################################################################
    ####Low Pass Filter
    if df:
        import fftw3
        if verbose: print "Applying filter on data transform"

        data = data * (1+0j)

        #############Do it this way to reuse the plan, requires some data copies
        input_array = np.zeros_like(data[0])
        output_array = np.zeros_like(data[0])
        fftForwardPlan = fftw3.Plan(input_array,output_array,'forward')
        fftBackwardPlan = fftw3.Plan(input_array,output_array,'backward')
        for i in range(len(data)):
            np.copyto(input_array, data[i])
            fftw3.guru_execute_dft(fftForwardPlan, input_array, output_array)
            np.copyto(input_array, output_array * filter)
            fftw3.guru_execute_dft(fftBackwardPlan, input_array, output_array)
            np.copyto(data[i], output_array)

        data = abs(data) / (data[0].size) #renormalize after transforms

    ###############################################################################################
    ###RESAMPLE and ZERO PAD
    if dyn < dxn:
        q = dxn
    else: 
        q = dyn

    dataStack = np.zeros((nSlices,nyn, nxn))
    yc = int(float(yc) * float(q) / float(y)) + 1
    xc = int(float(xc) * float(q) / float(x)) + 1

    p1 = nyn / 2 - yc + 1
    p2 = p1 + q
    p3 = nxn / 2 - xc + 1
    p4 = p3 + q

    if p1 < 0 or p3 < 0 or p2 > nyn or p4 > nxn:
        print "Zero-padding failed, try increasing pixels per scan step."
        return

    ##remove stripe noise from readout
    #for i in range(nSlices):
    #	data[i] = data[i] - data[i,0:10,:].mean(axis = 0)

    ##downsample using bivariate spline interpolation
    for i in range(nSlices):
        x, y = np.linspace(0, data[i].shape[0], data[i].shape[0]),np.linspace(0, data[i].shape[1], data[i].shape[1])
        sp = interpolate.RectBivariateSpline(x, y, data[i])
        yn, xn = np.meshgrid(np.linspace(0, data[i].shape[1], q), np.linspace(0, data[i].shape[0], q))
        dataStack[i, p1:p2, p3:p4] = sp.ev(xn.flatten(), yn.flatten()).reshape(q,q)

    if low_pass:
        dataStack *= np.exp(-dist(nyn)**2 / 2. / (nyn / 8)**2)

    ###Zero data points below the noise floor
    dataStack *= (dataStack > nl)

    #dataStack[dataStack > 35000.] = 0.
    #dataStack *= 1. - circle(256,3,130,130)    
    
    ####Return
    if fv: dataStack = dataStack[:,::-1,:]
    if fh: dataStack = dataStack[:,:,::-1]
    if tr: dataStack = np.transpose(dataStack, axes = (0,2,1))
    if fCCD: dataStack = np.transpose(np.rot90(np.transpose(dataStack)))
    if verbose: 
        print "Sending data from Rank %i" % parallel.rank
        #comm.send((i1,i2), dest = 0, tag = 12)
        
    if not parallel.master:
        parallel.send(np.ascontiguousarray(dataStack), dest = 0, tag = 12)
        return

    ######################################################################################################
    ###retrieve the chunks back and put them into the right spot in the stack (re-initialized to the new smaller size)
    nPoints = xpts * ypts
    dataStack = np.zeros((nPoints, nyn, nxn))
    for rank in range(1, parallel.size):
        indeces = distributed[rank]
        """
        i1,i2 = indeces[0], indices[
        
        comm.recv(source = rank, tag = 12)     ##these are wrong for double exposure mode
        if multi: 
            i1, i2 = i1 / 2, i2 / 2
        """
        dataStack[indeces] = parallel.receive(source = rank, tag = 12)
        
    tProcess = time.time() - t0
    if verbose: print "Done!"
    

    #if verbose: print "Processed frames at: %.2f Hz" %(float(loadedFrames / nexp) / tProcess)
    if verbose: 
        print "Total time per frame : %i ms" %(1000. * (tProcess + tPrep + tRead) / float(expectedFrames) * parallel.size)
            
    if verbose: 
        millis = tuple(1000. * np.array([tPrep, tRead, tProcess]) / float(expectedFrames) * parallel.size)
        print "Distribution (Prep, Load, Process): (%.2f | %.2f | %.2f ) ms" % millis

    #dataStack[:,118,128:132] = 0.
    probe_noise = dataStack[0]
    probe_noise = probe_noise * (probe_noise < 0.1 * probe_noise.max())#np.load('/global/groups/cosmic/Data/probe_intensities.npy')
    #dataStack = dataStack - probe_noise
    #dataStack = dataStack * (dataStack > 0.)

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
        print "Removed %i bad frames." % (len(badIndices))

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
    cxiObj.beamline = bl
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



def loadDirFrames(scanDir, filePrefix, expectedFrames, cropSize, multi = False, fCCD = True, \
                  frame_offset = 0, verbose = True):
                      
    from imageIO import imload
    from numpy import zeros

    if multi: 
        nexp = 2
    else: 
        nexp = 1
    numJobs = 0
    i = 0
    if fCCD: 
        ny, nx = cropSize, cropSize
    while (i < expectedFrames):
        sys.stdout.write("Loading frame %i \r" %i)
        sys.stdout.flush()
        frame_num = i + frame_offset
        if fCCD:
            filename = scanDir + filePrefix + str(frame_num) + ".tif"
        else:
            frame_num += 1
            if frame_num < 10:
                fileNumStr = '00' + str(frame_num)
                filename = scanDir + filePrefix + fileNumStr + ".tif"
            elif frame_num < 100:
                fileNumStr = '0' + str(frame_num)
                filename = scanDir + filePrefix + fileNumStr + ".tif"
            else:
                fileNumStr = str(frame_num)
                filename = scanDir + filePrefix + fileNumStr + ".tif"
        if fCCD:
            data = imload(filename).astype('uint16')[500:1500,:]
        else:
            data = imload(filename).astype('uint16')
        if i == 0:
            if fCCD: 
                ny, nx = cropSize, cropSize
            else: 
                ny, nx = data.shape
            if verbose: 
                print "Frames are %i x %i pixels (H x V)" %(nx,ny)
            dataStack = zeros((expectedFrames, data.shape[0],data.shape[1]), dtype = 'uint16')
    

        dataStack[i,:,:]= data
        i += 1
        loadedFrames = i

    return dataStack, loadedFrame

def aveDirFrames(bgScanDir, multi = False, fCCD = True, fCCD_version = 2):
    
    from numpy import hstack, fliplr, flipud, zeros, sort, zeros_like
    from imageIO import imload

    if fCCD:
        bgFileList = sorted(os.listdir(bgScanDir))
        bgFileList = [bgFileList[i] for i in range(len(bgFileList)) if bgFileList[i].count('tif')]
        longExpList = bgFileList[1::2]
        shortExpList = bgFileList[0::2]
        npoints = len(shortExpList)
        nFrames = 0
        for a,b,i in zip(longExpList,shortExpList,xrange(npoints)):
            if i == 0:
                bg = imload(bgScanDir + a).astype('single')[500:1500,:]
                #if fCCD_version == 1: bg = hstack((bg[534:1000,:], fliplr(flipud(shift(bg[0:500-34,:],-1,0)))))
                #elif fCCD_version == 2: bg = hstack((bg[534:1000,:], fliplr(flipud(shift(bg[0:500-34,:],0,0)))))
                bgNorm = zeros(bg.shape)
                bgNorm[bg > 0.] = 1.
                if multi:
                    bg_short = imload(bgScanDir + b).astype('single')[500:1500,:]
                    if fCCD_version == 1: bg_short = hstack((bg_short[534:1000,:], fliplr(flipud(shift(bg_short[0:500-34,:],-1,0)))))
                    elif fCCD_version == 2: bg_short = hstack((bg_short[534:1000,:], fliplr(flipud(shift(bg_short[0:500-34,:],0,0)))))
                    bgShortNorm = zeros(bg_short.shape)
                nFrames += 1
            else:
                temp = imload(bgScanDir + a).astype('single')[500:1500,:]
                #if fCCD_version == 1: temp = hstack((temp[534:1000,:], fliplr(flipud(shift(temp[0:500-34,:],-1,0)))))
                #elif fCCD_version == 2: temp = hstack((temp[534:1000,:], fliplr(flipud(shift(temp[0:500-34,:],0,0)))))
                temp_short = imload(bgScanDir + b).astype('single')[500:1500,:]
                #if fCCD_version == 1: temp_short = hstack((temp_short[534:1000,:], fliplr(flipud(shift(temp_short[0:500-34,:],-1,0)))))
                #elif fCCD_version == 2: temp_short = hstack((temp_short[534:1000,:], fliplr(flipud(shift(temp_short[0:500-34,:],0,0)))))
                bg += temp
                bgNorm[temp > 0.] += 1
                if multi:
                    bg_short += temp_short
                    bgShortNorm += temp_short > 0.
                else:
                    bg += temp_short
                    bgNorm += temp_short > 0.
                nFrames += 2
        bgNorm[bgNorm == 0.] = 1.
        bg = bg / bgNorm
        if multi:
            bgShortNorm[bgShortNorm == 0.] = 1.
            bg_short = bg_short / bgShortNorm
        else: bg_short = None
    else:
        i = 0
        for item in sort(os.listdir(bgScanDir)):
            if item.split('.')[-1::][0] == 'tif':
                if i == 0:
                    bg = imload(bgScanDir + item).astype('single')
                    if multi: bg_short = zeros_like(bg)
                else:
                    if multi:
                        if (i % 2) == 0:
                            bg += imload(bgScanDir + item).astype('single')
                        else:
                            bg_short += imload(bgScanDir + item).astype('single')
                    else:
                        bg += imload(bgScanDir + item).astype('single')
                i += 1
        if i == 0:
            if verbose: print "Did not find any tif files in the directory: %s" %bgScanDir
        else:
            if multi:
                bg = bg / (i / 2)
                bg_short = bg_short / (i / 2)
            else:
                bg = bg / i
                bg_short = None
        nFrames = i

    return bg, bg_short, nFrames


