import time, os, sys, json

from cosmic.utils import dist, shift, circle, e2l, eval_hdr, ScanInfo

from scipy import random, ndimage, interpolate

from .processFCCD import ProcessFCCDraw  # fitOverscan, processfCCDFrame

import numpy as np

import cosmic.ext as ce

from cosmic.io import cxi, write_CXI_metadata, write_CXI_data, SharpClient

import warnings
import h5py

from glob import glob

parallel = ce.utils.parallel


# imread = ce.tifffile.imread

class AttrDict(object):
    """
    Wrapper around dictionary for convenient attribute access
    """

    def __init__(self, dct=None):
        self.set_dict(dct)

    def get_dict(self):
        return self.__dict__

    def set_dict(self, dct=None):
        if dct is not None:
            self.set_dict(dct)


class Point(object):

    def __init__(self, x=0, y=0, dtype=np.float):
        self.arr = np.array([y, x], dtype=dtype)

    @property
    def x(self):
        return self.arr[1]

    @x.setter
    def x(self, val):
        self.arr[1] = val

    @property
    def y(self):
        return self.arr[0]

    @y.setter
    def y(self, val):
        self.arr[0] = val


class Log(object):

    def __init__(self, verbose=True):
        self.verbose = verbose and parallel.master

    @property
    def emit(self):
        # non mpi
        print(self.msg)

    @emit.setter
    def emit(self, msg):
        self.msg = msg
        if self.verbose:
            print(msg)


from cosmic.ext.ptypy.utils.validator import ArgParseParameter
from io import StringIO

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
    
    [scan.dark_num_total]
    default = 0
    help = Total number of dark frames expected.
               
    [scan.exp_num_total]
    default = 0
    help = Total number of exposed frames expected.
                            
    [scan.number]
    default = None
    help = Number of scan
    
    [scan.index]
    default = 1
    help = Index of scan in multi energy series
                            
    [scan.date]
    default = None
    help = Date of scan
                            
    [scan.id]
    default = None
    help = Identifying string for scan
                            
    [scan.dwell] 
    default = None
    help = Dwell times 
    
    [scan.repetition]
    default = 1
    help = Exposure repetition per scan point
    
    [scan.double_exposure]
    default = False
    help = Inidicates double exposure mode
    
    [process]
    default = None
    help = Processing paramaters for diffraction images
    
    [process.frames_per_rank_and_block]
    default = 10
    help = Number of frames per rank and block of data
    
    [process.adu_per_ev]
    default = 0.004
    help = How much ADU are generated on average per photon ev. Depends
      on gain setting. 
    
    [process.adu_offset]
    default = 0.35
    help = Manual offset correction (in ADU)
    
    [process.gap]
    default = 70
    help = Vertical gap in the diffraction images. 
                
    [process.threshold] 
    default = 2000
    help = Fast CCD per pixel saturation threshold
    alias = s_threshold
    
    [process.crop]
    default = 960
    help = Raw data will be cropped to a pixel areas of this size
    alias = crop_size
    
    [process.scols] 
    default = 192
    help = Total number of supercolumns
                            
    [process.do_overscan]
    default = True
    help = Use overscan areas additional background substraction.
                            
    [process.do_spike_removal]
    default = True
    help = Remove spikes from cross channel crosstalk.
    
    [process.precount]
    default = -1
    help = Column precount, vary to shift the overscan pixels appropriately
    
    [geometry]
    default = None
    help = Information about the imaging geometry. Maybe filled
    
    [geometry.resolution]
    default = 5e-9
    help = Target resolution value in nm
    
    [geometry.rebin]
    default = 2
    help = Rebin diffraction this many pixels for both axis.
    
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
    
    [geometry.shape]
    default = 256
    help = Target diffraction pattern shape after interpolation. Set to None for no interpolation. 
    
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
    
    [post.orientation]
    default = 7
    help = From [0-7] (+1 for fliplr, +2 for flipud, +4 for transpose)
    
    [post.low_pass]
    default = 0.
    help = RMS of low pass applied filter to diffraction data if >0
    alias = filter_width 
    
    [post.noise_level]
    default = 0.5
    help = Baseline noise level to subtract
    alias = nl
    
    [post.adu_per_ev]
    default = 0.004
    help = How much adu are generated on average for a 1000 eV photon. 
    
    [post.probe_threshold]
    default = 0.1
    help = Threshold for annulus to estimate probe from diffraction patterns.
    alias = pthreshold
    
    [post.defocus]
    default = 0.0
    help = Amount to defocus the probe by after the initial estimate if >0
    """))


def json_to_param(param, jload):
    scan = jload
    scan['dwell'] = (scan['dwell1'], scan['dwell2'])
    # sideload scan specific stuff
    for k, v in scan.items():
        if k in param['scan']:
            param['scan'][k] = v
        elif k in param['geometry']:
            param['geometry'][k] = v
        else:
            if 'info' not in param:
                param['info'] = {}
            param['info'][k] = v


class Processor(object):
    PROCESSED_FRAMES_PER_NODE = 4

    def __init__(self, param=None, verbose=True):

        self.log = Log(verbose)
        self.sharpy = SharpClient()
        self.param = DEFAULT.make_default(depth=10) if param is None else param
        self.state = AttrDict()
        self.state.frame_center = None
        # other state variables
        self._ip = None

        ## SETUP sharp client
        if parallel.master:
            self.sharpy.setup(num_proc_ids=parallel.size)

    def load_scan_definition(self, scanfile):
        """
        Loads scan-specific information. Overwrites defaults held in
        internal default dictionary `param`
        """
        param = self.param
        if hasattr(scanfile, 'items'):
            json_to_param(param, dict(scanfile))
        elif scanfile.endswith('json'):
            with open(scanfile, 'r') as f:
                scan = json.load(f)
                json_to_param(param, scan)
                f.close()
        elif scanfile.endswith('.hdr'):
            d = eval_hdr(scanfile)
            basedir, fname = os.path.split(scanfile)
            index = fname[3:-4]
            exp_dir_num = 1
            dark_dir_num = 2
            sdef = d['ScanDefinition']
            param['geometry']['energy'] = float(sdef['StackAxis']['Min'])
            param['scan']['exp_dir'] = basedir + '/' + index + '/%03d' % exp_dir_num
            param['scan']['darkdir'] = basedir + '/' + index + '/%03d' % dark_dir_num
            param['scan']['dwell'] = (float(sdef['Dwell']), float(sdef['Dwell2']))
        elif scanfile.endswith('scan_info.txt'):
            SI = ScanInfo()
            # SI.read_file(scanfile)
            # json_to_param(param,SI.to_json())
            json_to_param(param, SI.read_file(scanfile))
        else:
            self.log.emit = 'Scanfile %s is not understood' % scanfile
            return
        # fill the rest
        scan = param['scan']
        trunk, idx = os.path.split(scan['exp_dir'])
        scan['index'] = int(idx)
        scan['base'], scan['id'] = os.path.split(trunk)
        scan['number'] = int(scan['id'][6:])
        import datetime
        scan['date'] = scan['id'][:6]  # datetime.datetime.strptime(scan['id'][:6],'%y%m%d')


    # def get_zoneplate_info(self, geo):
    #     """
    #     Zoneplate specific things. Not essential
    #     """
    #     # zp focal length microns
    #     f = geo['zp_diameter'] * geo['zp_outer_width'] / geo['lambda']
    #     # zp numerical aperture
    #     na = geo['zp_diameter'] / f / 2.
    #     self.log.emit = "Zone plate focal length: %.2f mm" % f
    #     return na, f


    def calculate_geometry(self):
        """
        Uses geometric information to calculate the array padding
        """
        self.log.emit = "--- Calculating geometry ----"

        geo = self.param['geometry']

        # wavelength in nm
        geo['lambda'] = e2l(geo['energy'])

        # Nmber of pixels (at physical CCD pixel size) before downsampling 
        ccdp = geo['psize'] * geo['rebin']  ##physical CCD pixel size, usually 30 microns
        ccdz = geo['distance'] * 1000.  ## sample to CCD distance, converted to microns

        sh = geo['shape']
        resol = geo['resolution']

        if resol is not None:
            # adapt shape
            pixnm = Point(resol * 1e9, resol * 1e9, dtype=np.single)
            sh_pad = Point(dtype=np.int)
            sh_pad.arr[:] = 2 * np.round(ccdz * geo['lambda'] / 2 / pixnm.arr / ccdp)
            pixnm.arr[:] = ccdz * geo['lambda'] / ccdp / sh_pad.arr  # rounded resolution.
            if sh is None:
                sh_out = Point(dtype=np.int)
                sh_out.arr[:] = sh_pad.arr
            else:
                # size to which the raw data is resampled, usually 256x256
                sh_out = Point(sh, sh, dtype=np.int)
        else:
            # adapt resolution
            if sh is not None:
                sh_out = Point(sh, sh, dtype=np.int)
                sh_pad = Point(dtype=np.int)
                sh_pad.arr[:] = sh_out.arr
                pixnm = Point(dtype=np.single)
                pixnm.arr[:] = ccdz * geo['lambda'] / ccdp / sh_pad.arr
            else:
                # both None.
                warnings.warn('`shape` and `resolution` cannot both be None. Setting `shape` to 128')
                geo['shape'] = 128
                self.calculate_geometry()
                return

        sh_crop = Point()
        sh_crop.arr[:] = sh_pad.arr * geo['rebin']

        # resulting detector pixel size
        psize_out = Point()
        psize_out.arr = sh_pad.arr.astype(float) / sh_out.arr * ccdp / 1e6

        self._do_interpolate = ((sh_out.arr - sh_pad.arr) != 0).any()

        log = self.log
        log.emit = "Photon energy: %.2f eV" % geo['energy']
        log.emit = "Wavelength: %.2f nm" % geo['lambda']
        log.emit = "Sample-CCD distance: %.2f mm" % geo['distance']
        log.emit = "Requested pixel size (nm): %.2f" % (pixnm.x)
        log.emit = "Will crop the data to: %i x %i (HxV)" % (sh_crop.x, sh_crop.y)
        log.emit = "Will rebin the data to: %i x %i (HxV)" % (sh_pad.x, sh_pad.y)
        if self._do_interpolate:
            log.emit = "Will interpolate data to: %i x %i (HxV)" % (sh_out.x, sh_out.y)
        else:
            log.emit = "Will not interpolate data"

        # save this info
        self.state.sh_pad = sh_pad
        self.state.sh_crop = sh_crop
        self.state.sh_out = sh_out
        self.state.pixnm = pixnm
        self.state.psize_out = psize_out

    def calculate_translation(self):
        # funny enough the translation for cxi are
        # [fastaxis, slowaxis, {0}]
        # fast axis is horizontal direction in microscope
        # This is because the camera is tilted.
        # However if the readout of the CCD were (1152 x 2000) instead
        # of (2000 x 1152) this axis inversion would not be needed.
        s = self.state
        info = self.param['info']
        self.log.emit = "--- Calculating rectangular scan translations ----"

        shifts = info.get('translations')
        if shifts is None:
            # This is old code and only for reference. Shoud not occur anymore
            # Confusing anyway         
            ss = Point(info['exp_step_y'] * 1000, info['exp_step_x'] * 1000)

            # number of scan points in X and Y for raster scans 
            pts = Point(info['exp_num_x'], info['exp_num_y'], dtype=np.int)
            expectedFrames = pts.x * pts.y
            ## Translation & data preparation
            pixIndex = [(i / pts.x, i % pts.x) for i in range(expectedFrames)]
            shiftData = [(-(float(pixIndex[i][0]) - float(pts.y - 1) / 2) * ss.y,
                          (float(pixIndex[i][1]) - float(pts.x - 1) / 2) * ss.x)
                         for i in range(expectedFrames)]
            x = np.array(shiftData)[:, 1]
            y = np.array(shiftData)[:, 0]
            translationX, translationY = (x - x.min()) * 1e-9, (y - y.min()) * 1e-9
            # (x,y,z) in meters
            s.translation = np.column_stack((translationX, translationY, np.zeros(translationY.size)))
            self.log.emit = "Probe step in pixels (x,y): %.2f,%.2f" % (ss.x / s.pixnm.x, ss.y / s.pixnm.y)
        else:
            self.log.emit = "Found array of shifts / translations in scan info.\nShift calculation skipped."
            # (x,y,z) in meters
            shifts = np.asarray(shifts)
            s.translation = np.zeros((len(shifts), 3))
            s.translation[:, 0] = shifts[:, 1] * 1e-6
            s.translation[:, 1] = (-shifts[:, 0] + shifts.max(0)[0]) * 1e-6

        s.nPoints = len(s.translation)
        s.indices = np.arange(s.nPoints)

        # this is stupid but necessary
        s.stxm = np.ones((info['exp_num_y'], info['exp_num_x']))

        self.log.emit = "Expecting %i frames." % (s.nPoints)

    def make_cxi_path(self):
        scan = self.param['scan']
        outputPath = scan['base'] + os.path.sep
        return outputPath + 'NS_' + scan['id'] + '_%03d' % scan['index'] + '.cxi'

    def write_cxi(self):
        """
        This one writes a non-canonical cxi.
        """
        if parallel.master:
            s = self.state
            geo = self.param['geometry']
            cxipath = self.make_cxi_path()
            self.log.emit = "Writing data to file: %s" % cxipath

            cxiObj = cxi()

            cxiObj.process = {'json': json.dumps(self.param)}
            cxiObj.probe = s.probe
            cxiObj.beamline = '5.3.2.1'
            cxiObj.energy = geo['energy'] * 1.602e-19
            cxiObj.probemask = s.fmask
            cxiObj.probeRmask = s.mask
            cxiObj.datamean = s.diff_avg
            cxiObj.illuminationIntensities = s.diff_avg
            cxiObj.stxm = s.stxm
            cxiObj.stxmInterp = 'None'
            cxiObj.xpixelsize = s.psize_out.x
            cxiObj.ypixelsize = s.psize_out.y
            # (x,y,z) in meters (this seems wrong and might only work because everythings squared)
            corner_position = [s.psize_out.x * s.sh_out.x / 2.,
                               s.psize_out.y * s.sh_out.y / 2.,
                               geo['distance'] * 1e-3]
            """
            corner_position = [float(s.sh_pad.x) * s.ccdp / 2. / 1e6, 
                               float(s.sh_pad.y) * s.ccdp / 2. / 1e6, 
                               s.ccdz / 1e6] 
            """
            cxiObj.corner_x, cxiObj.corner_y, cxiObj.corner_z = corner_position
            cxiObj.translation = s.translation
            cxiObj.indices = s.indices
            write_CXI_metadata(cxiObj, cxipath)
            # self.sharpy.file_ready(cxipath)
            self.log.emit = "Done."

    def prepare(self, dark_dir=None, exp_dir=None, dealer=None, **dealer_kwargs):
        """
        Prepares the sharp

        kwargs beyond defaults are piped to the Dealer class
        """
        # local references
        geo = self.param['geometry']
        proc = self.param['process']
        scan = self.param['scan']
        s = self.state

        t0 = time.time()
        print(tuple(s.psize_out.arr))
        ## PREPARE sharp client
        if parallel.master:
            self.sharpy.prepare(
                energy=geo['energy'] * 1.602e-19,
                distance=geo['distance'] * 1e-3,
                num_frames=s.nPoints,
                shape=tuple(s.sh_out.arr.astype(np.int32)),
                pixelsize=tuple(s.psize_out.arr.astype(np.float32)),
                run_file=self.make_cxi_path(),
            )

            ## Ptyd base
            s.meta = dict(
                # Assuming  square data arrays
                shape=s.sh_out.arr,
                # Joule to kiloelectronvolt
                energy=geo['energy'] / 1000.,
                distance=geo['distance'] / 1000.,
                # assuming rectangular pixels,
                psize=s.psize_out.arr,
                propagation='farfield',
                num_frames=s.nPoints
            )

            s.ptydpath = '%(base)s/%(id)s/ptyd/NS_%(id)s_%(index)03d.ptyd' % scan
            print(s.ptydpath)

        ## Frame Dealer
        dwell = scan['dwell'] if scan['double_exposure'] else [1.0]  # may be switched

        dark_dir = scan['dark_dir'] if dark_dir is None else dark_dir
        exp_dir = scan['exp_dir'] if exp_dir is None else exp_dir
        ## Decide here between a StreamDealer or a TifDealer 
        dealer = dealer if dealer is not None else TiffDealer
        self.log.emit = "Using as %s frame dealer" % type(dealer)
        self.dealer = dealer(exp_dir=exp_dir,
                             dark_dir=dark_dir,
                             exp_total=scan['exp_num_total'],
                             dark_total=scan['dark_num_total'],
                             gap=proc['gap'],
                             dtype=np.float,
                             dwell=dwell,
                             repetitions=scan['repetition'],
                             threshold=proc['threshold'],
                             rows=proc['crop'] // 2,
                             scols=proc['scols'],
                             precount=proc['precount'],
                             do_overscan=proc['do_overscan'],
                             do_spike_removal=proc['do_spike_removal'],
                             **dealer_kwargs)

    def stack_process(self, data,
                      center=None,
                      cthreshold=10,
                      Ncrop=None,
                      Nbin=None,
                      orientation=None
                      ):
        if center is None:
            if self.state.frame_center is None:
                dataAve = data.mean(0)
                # locate the center of mass of the average and shift to corner
                # threshold, 10 is about 1 photon at 750 eV
                dataAve = dataAve * (dataAve > cthreshold)

                # calculate frame center
                center = ce.utils.mass_center(dataAve)
                self.state.frame_center = center
                self.log.emit = "Found center at %s " % str(center)
            else:
                center = self.state.frame_center
        else:
            self.state.frame_center = center

        # crop
        crop = np.array([Ncrop, Ncrop]) if Ncrop is not None else self.state.sh_crop.arr
        data, center = ce.utils.crop_pad_symmetric_2d(data, crop, center)

        # rebin
        shbin = Nbin if Nbin is not None else self.state.sh_pad.x
        rebin = data.shape[-1] // shbin
        data, center = ce.utils.rebin_2d(data, rebin), center / rebin

        # switch data
        orientation = self.param['post']['orientation'] if orientation is None else orientation
        data, center = ce.utils.switch_orientation(data, orientation, center)

        return data, center

    def interpolate_2d(self, data, center, lsh_out=None, lsh_pad=None):

        # interpolate
        if self._ip is None:
            _ = AttrDict()
            _.center = center
            sh = data.shape[-2:]
            out = lsh_out if lsh_out is not None else self.state.sh_out.x
            pad = lsh_pad if lsh_pad is not None else self.state.sh_pad.x

            _.row = np.linspace(0, sh[0] - 1, sh[0]) - center[0]
            _.col = np.linspace(0, sh[1] - 1, sh[1]) - center[1]
            _.out = out
            symspace = np.linspace(0, pad, out) - pad / 2
            _.evcol, _.evrow = np.meshgrid(symspace, symspace)
            _.scale = np.prod(self.state.sh_pad.arr) / np.prod(self.state.sh_out.arr)
            _.center = np.array([pad / 2, pad / 2])
            self._ip = _
            self._ip.func = lambda f: self._smoothed_spline(f)

        return self._ip.func(data), self._ip.center

    def _smoothed_spline(self, data, low_var=None):
        _ = self._ip
        fw = self.param['post']['low_pass'] if low_var is None else low_var
        if fw > 0.:
            data = ce.utils.gf_2d(data, fw)

        ret = []
        rows = _.evrow.flatten()
        cols = _.evcol.flatten()
        for d in data:
            spline = interpolate.RectBivariateSpline(_.row, _.col, d)
            ret.append(spline.ev(rows, cols).reshape(_.out, _.out))  # *_.scale)

        return np.asarray(ret)

    def probe_guess(self, avg, threshold=None):
        """
        Calculate a probe guess from the average diffraction pattern.
        """
        th = threshold if threshold is not None else self.param['post']['probe_threshold']
        fave = np.fft.fftshift(avg)
        fmask = (fave > th * fave.max())
        guess = np.fft.fftshift(np.fft.ifftn(np.sqrt(fave) * fmask))
        N = fmask.shape[0]
        mask = circle(N, N / 4, N / 2, N / 2)

        # We can propagate this to some defocus distance if needed.
        """
        if post['defocus'] != 0:
            from propagate import propnf
            probe = propnf(probe, defocus * 1000. / pixnm.x, l / pixnm.x)
        """
        # maybe set cxi directly here
        self.state.mask = mask
        self.state.probe = guess
        self.state.fmask = fmask
        self.state.diff_avg = avg

        return guess, fmask, mask

    def process(self, start=0):
        while not self.process_init(start=start):
            self.log.emit = "Waiting for frames"
            time.sleep(0.1)

        while not self.end_of_scan:
            msg = self.process_loop()
            if msg == 'break': break
            elif msg == 'wait':
                time.sleep(0.1)
                continue

        self.log.emit = "Done!"
        self.print_benchmarks()

    def print_benchmarks(self):
        s = self.state
        # if verbose: print "Processed frames at: %.2f Hz" %(float(loadedFrames / nexp) / t_process)
        avg_time = 1000. * (s.t_process + s.t_prep + s.t_read) / float(s.nPoints)
        self.log.emit = "Avg time per frame : (All cores %.2f | Single core %.2f ) ms" % (avg_time, avg_time * parallel.size)
        millis = tuple(1000. * np.array([s.t_read, s.t_process]) / float(s.nPoints))
        self.log.emit = "      Distribution : (Load %.2f | Process %.2f ) ms" % millis

    def process_init(self, start=0):

        if self.dealer.frames_available() <= 0:
            return False

        cxipath = self.make_cxi_path()

        if parallel.master: 
            h5py.File(cxipath, "w")  # here we only truncate the file, in case it exists, to guarantee we overwrite it

        s = self.state
        s.cxipath = cxipath
        log = self.log


        t0 = time.time()
        #log.emit = "Starting frame is: %i" % (Dealer.frame_nums[0])

        ######################################################################################################
        ###load and average the background frames
        ###this alternates long and short frames regardless of "multi", should be fixed
        log.emit = "Averaging background frames..."

        ndarks = self.dealer.set_dark()

        log.emit = "Done. Averaged %d frames." % ndarks

        self.calculate_translation()

        #### MPI DISTRIBUTION #################
        s.t_prep = time.time() - t0
        self.start = start
        s.chunk_num = 0
        s.last_available = 0
        s.t_read = 0
        s.t_process = 0

        self.end_of_scan = False

        return True
    
    def process_loop(self, chunk_size=None):

        s = self.state
        start = self.start
        param = self.param
        log = self.log
        
        if not chunk_size:
            chunk_size = param['process']['frames_per_rank_and_block'] * parallel.size
        # for tries in range(1000): # maximum wait time is 1000sec

        if self.end_of_scan: 
            return 'break'

        parallel.barrier()

        stop = self.start + chunk_size
        is_last_chunk = stop >= s.nPoints

        # indices to distribute
        frames_available = self.dealer.frames_available(start)

        stream_stagnates = frames_available == s.last_available

        log.emit = "%d frames available." % frames_available

        ## Decision tree
        if frames_available >= chunk_size:
            # enough frames, can be in the middle or at the end
            if is_last_chunk: self.end_of_scan = True
        else:
            # too little frames means either end-of scan or pause
            if not stream_stagnates or (stream_stagnates and s.chunk_num == 0):
                s.last_available = frames_available
                return 'wait'
            else:
                s.last_available = 0
                stop = min(start + frames_available, s.nPoints)
                self.end_of_scan = True

        # abort conditions

        # we got a stack
        stack_indices = list(range(start, stop))
        chunk = np.zeros((len(stack_indices), s.sh_out.y, s.sh_out.x))

        t0 = time.time()
        log.emit = "Reading frames %d through %d." % (start, stop - 1)

        # MPI distribution keys
        parallel.loadmanager.reset()
        distributed = parallel.loadmanager.assign(stack_indices)
        chunk_indices = distributed[parallel.rank]

        # node specific portion of global indices
        node_indices = [stack_indices[j] for j in chunk_indices]

        if not node_indices and self.end_of_scan:
            # You may leave now
            return 'break'

        # load and crop from indices
        try:
            data = self.dealer.get_clean_data(node_indices)
        except:
            # once more
            data = self.dealer.get_clean_data(node_indices)

        # discretize data to photon hits.
        # scale to single photon hits
        data -= param['process']['adu_offset']
        data /= param['process']['adu_per_ev'] * param['geometry']['energy']

        # Todo: this ain't ideal
        data = np.floor(data)

        s.t_read += time.time() - t0
        t0 = time.time()
        log.emit = "Read %d frames on %d processes in %.2f seconds" % (len(stack_indices), parallel.size, s.t_read)

        ## Check crop ##
        crop_size = param['process']['crop']
        if data.shape[-2] != crop_size or data.shape[-2] != crop_size:
            log.emit = "Raw data shape is %d x %d" % data.shape[-2:] + " and should be %d x %d." % (
            crop_size, crop_size)
            log.emit = "Non-rectangular diffraction frames may lead to unexpected results in the interpolation"

        post = param['post']

        """
        from matplotlib import pyplot as plt
        print data[-1].min()
        plt.imshow(data[-1]-data[-1].min())
        plt.show()
        """
        if s.chunk_num == 0:
            ## center detection
            data_avg = data.sum(axis=0)

            """
            from matplotlib import pyplot as plt
            plt.imshow(np.log10(data_avg+1000))
            plt.show()
            """
            # average dataset across all nodes
            parallel.allreduce(data_avg)

            sh = data_avg.shape

            data_avg, center = self.stack_process(data_avg.reshape((1,) + sh))

            # interpolate
            if self._do_interpolate:
                data_avg, center = self.interpolate_2d(data_avg, center)

            # calculate probe from data_avg
            log.emit = "Calculating probe and mask"

            data_avg[data_avg < 0.] = 0.
            init, fmask, mask = self.probe_guess(data_avg[0])

            ## SHARP ENDPOINT access ###
            if parallel.master:
                self.sharpy.update(
                    probe_init=init,
                    probe_fmask=fmask,
                    probe_mask=mask,
                    positions=s.translation[:, :2]
                )

                ## Ptyd creation
                s.meta['center'] = center
                rwrwr = (4 + 2 + 1) * 2 ** 6 + (4 + 2 + 1) * 2 ** 3 + 4 + 1

                if os.path.exists(s.ptydpath):
                    os.remove(s.ptydpath)
                ce.io.h5write(s.ptydpath, meta=s.meta, info=self.param)
                os.chmod(os.path.split(s.ptydpath)[0], rwrwr)

        # interpolate
        data, center = self.stack_process(data)

        # interpolate
        if self._do_interpolate:
            data, center = self.interpolate_2d(data, center)

        # truncate with a gaussian if requested
        if post['truncate']:
            data *= np.exp(-dist(s.sh_out.x) ** 2 / 2. / (s.sh_out.x / 4) ** 2)

        # subtract a noise floor
        data -= post['noise_level']
        data *= data > 0.

        """
        ## MPI PUSH to sharp endpoint. Each process has separate push
        for ind, d in zip(node_indices,data):
            out_dict = dict(
                num = ind,
                process_id = parallel.rank,
                position = s.translation[ind][:2],
                data = d,
                mask = None,
            )
            self.sharpy.push('clean' , **out_dict)
        """
        ## Put data into the right spot in the stack
        chunk[chunk_indices] = data
        parallel.allreduce(chunk)

        ## ptyd chunk generation

        if parallel.master:

            write_CXI_data(chunk, self.state.cxipath)

            # ptypy
            ch = dict(
                data=chunk,
                indices=stack_indices,
                positions=s.translation[stack_indices, 1::-1],
            )

            h5address = 'chunks/%d' % s.chunk_num
            hddaddress = self.state.ptydpath + '.part%03d' % s.chunk_num
            log.emit = "Writing %s" % hddaddress

            ce.io.h5write(hddaddress, ch)
            with h5py.File(self.state.ptydpath, 'r+') as f:
                f[h5address] = h5py.ExternalLink(hddaddress, '/')
                f.close()

            log.emit = "Done writing .ptyd."

            log.emit = "Using streaming endpoint"
            # sharp endpoint push from master node only
            for ind, d in zip(stack_indices, chunk):
                out_dict = dict(
                    num=ind,
                    process_id=parallel.rank,
                    position=s.translation[ind][:2],
                    data=d,
                    mask=None,
                )
                self.sharpy.push('clean', **out_dict)
            log.emit = "Done streaming."


        s.t_process += time.time() - t0
        log.emit = "Processed %d frames on %d processes in %.2f seconds" % (
        len(stack_indices), parallel.size, s.t_process)

        s.chunk_num += 1
        self.start = stop

        if parallel.master:
            return self.state.ptydpath, hddaddress, s.chunk_num-1
        else:
            return None, None, None

class FrameDealer(object):
    """
    This class basically combines multiple exposures and carries out
    the background correction on individual CCD frames

    It assumes the script has been started with MPI.
    """

    def __init__(self,
                 exp_dir=None,
                 dark_dir=None,
                 dwell=None,
                 repetitions=1,
                 dark_total=None,
                 exp_total=None,
                 threshold=2000,
                 rows=960,
                 scols=192,
                 gap=70,
                 dtype=np.uint16,
                 precount=0,
                 do_overscan=True,
                 do_spike_removal=True):

        self.repetitions = repetitions
        self.threshold = threshold
        self.dwell = dwell if dwell is not None else [1.0]
        self.multi = (dwell is not None and len(dwell) > 1.)
        self.do_overscan = do_overscan
        self.do_spike_removal = do_spike_removal
        self.rows = rows
        self.gap = gap
        self._cuts = None
        self.cols = None
        self.dark_total = dark_total
        self.exp_total = exp_total
        self.dark_processed = []
        self.exp_paths = [] # A list of all paths to exposed frames
        self.exp_dir = {} if exp_dir is None else exp_dir
        self.dark_dir = {} if dark_dir is None else dark_dir
        self.dtype=dtype
        # set up processor
        self.Pro = ProcessFCCDraw(rows=rows, scols=scols, precount=precount)

    def get_darks(self):
        paths = self.darks_available_flat()
        dark_frames = self.load_and_crop(paths)
        return dark_frames

    def set_dark(self):
        """Assumes a flat list of dark frames"""
        #self.dark = []

        dark_frames = np.asarray(self.get_darks())

        if self.dark_total is not None:
            dark_frames = dark_frames[:self.dark_total * self.repetitions * len(self.dwell)]

        for t, dwell in enumerate(self.dwell):
            bg = dark_frames[t::len(self.dwell)]
            norm = (bg > 0).astype(np.float64)
            norm = norm.sum(0)
            norm[norm == 0.0] = 1.0
            res = bg.sum(0) / norm
            #self.dark.append(res)
            self.dark_processed.append(
                self.Pro.process_one_frame(res, do_overscan=self.do_overscan, do_spike_removal=self.do_spike_removal))

        return len(dark_frames)

    def darks_available_flat(self):
        """
        return all darks frame paths
        """
        return list(self.dark_dir.keys())


    def get(self, paths):
        """
        Swap a list of paths with a list of data frames
        """
        return [self.exp_dir[p] for p in paths]

    def frames_available_flat(self):
        """
        return all frame paths available in ascending flat index
        """
        raise list(self.exp_dir.keys())

    def frames_available(self, start=0):
        """
        Frames available, grouped per scan point.
        Updates all paths to exposures internally.
        """
        paths = self.frames_available_flat()
        self.exp_paths = paths
        c = self.repetitions * len(self.dwell)
        M = len(paths) - (start * c) // c

        if self.exp_total is not None:
            return min(M, self.exp_total - start)
        else:
            return M

    def get_paths(self, flat_indices):
        """
        Returns data using flat indices
        """
        paths = [self.exp_paths[ii] for ii in flat_indices]
        return paths

    def get_clean_data(self, node_indices, verbose=True):
        """Accesses data and dark attribute"""
        ### PROCESS THE DATA STACK ###########
        D = len(self.dwell)
        R = self.repetitions

        short_exp_first = (D > 1 and self.dwell[0] == min(self.dwell))
        dmask = None
        res = None
        for d, dwell in enumerate(self.dwell):

            rsum = None
            smask = None
            inorm = None

            for r in range(R):
                flat_indeces = [ind * D * R + D * r + d for ind in node_indices]
                paths = self.get_paths(flat_indeces)
                data = np.asarray(self.load_and_crop(paths))
                # missing pixel mask, float conversion necessary as it goes through the stack process
                imask = (data > 10).astype(np.float)

                # data -= self.dark[d]

                data = self.Pro.process_stack(data, verbose, do_overscan=self.do_overscan,
                                              do_spike_removal=self.do_spike_removal)
                imask = self.Pro.process_stack(imask, False)

                data -= self.dark_processed[d]

                """
                if parallel.master:
                    from matplotlib import pyplot as plt
                    #plt.imshow(data[0,420:500,570:650])
                    #plt.imshow(np.log10(data[0,400:500,400:545]))
                    plt.imshow(np.log10(data[0]+1))
                    #plt.imshow(data[0] * (data[0] > 2000))
                    #plt.imshow(data[0] * (data[0] < 100) * (data[0] > 0 ))
                    #plt.imshow(data[0,400:550,350:545])
                    plt.title(str(dwell))
                    plt.colorbar()
                    plt.show()
                """

                # mask calculation should come before correction. It is here only because of shape issues
                smask = data < self.threshold if smask is None else smask & (data < self.threshold)

                rsum = data * imask if rsum is None else rsum + data * imask
                inorm = imask if inorm is None else inorm + imask

            inorm[inorm <= 1.] = 1.
            rsum /= inorm
            """
            if dmask is None:
                dmask = smask
                res = rsum  / dwell if D > 1 else rsum / dwell
            else:
                res = dmask * res + (1-dmask) * rsum / dwell 
                dmask |= smask
            """
            if D == 1:
                res = rsum
                mask = smask
            else:
                if short_exp_first:
                    if d == 0:
                        # first dwell
                        res = rsum * max(self.dwell) / dwell
                        dmask = smask
                    else:
                        # second dwell
                        res = rsum * smask + (1 - smask) * res
                        mask = dmask | smask
                else:
                    if d == 0:
                        dmask = smask
                        res = rsum
                    else:
                        res = dmask * res + (1 - dmask) * rsum * max(self.dwell) / dwell
                        mask = dmask | smask
        return res

    def _set_crop(self, shape):

        N, cols = shape
        self.cols = cols
        self._cuts = [
            N // 2 - self.rows - self.gap // 2,
            N // 2 - self.gap // 2,
            N // 2 + self.gap // 2,
            N // 2 + self.gap // 2 + self.rows
        ]

    def _crop(self, one):
        cut1, cut2, cut3, cut4 = self._cuts

        cropped = np.zeros((2 * self.rows, self.cols), dtype=self.dtype)
        cropped[0:self.rows, :] = one[cut1:cut2, :]
        cropped[self.rows:2 * self.rows, :] = one[cut3:cut4, :]
        return cropped

    def load_and_crop(self, paths, shape=None):

        # load first:
        f = self.get([paths[0]])[0].astype(self.dtype)

        sh = shape if shape is not None else f.shape
        if self._cuts is None:
            self._set_crop(sh)

        frames = [self._crop(f)]

        if len(paths) > 1:
            frames += [self._crop(f.astype(self.dtype)) for f in self.get(paths[1:])]

        return frames

class FrameDealerMPI(FrameDealer):

    def frames_available(self, start=0):
        if parallel.master:
            m = super().frames_available(start=start)
        else:
            m = None
        m = parallel.bcast(m)
        self.exp_paths = parallel.comm.bcast(self.exp_paths)
        return m

    def get_darks(self):
        if parallel.master:
            paths = self.darks_available_flat()
            dark_frames = super().load_and_crop(paths)
        else:
            dark_frames = None

        dark_frames = parallel.bcast(np.array(dark_frames))

        return dark_frames

    def load_and_crop(self, paths, shape=None):
        # collect all paths at rank 0
        dct = dict.fromkeys(paths, parallel.rank)
        pathrank = parallel.gather_dict(dct)
        if parallel.master:
            frames = super().load_and_crop(list(pathrank.keys()))
            dct = dict(zip(pathrank.keys(),frames))
        else:
            dct = {}
        dct = parallel.bcast_dict(dct, keys=paths)
        return list(dct.values())

class _TiffIOMixin(object):

    def darks_available_flat(self):
        bg_dir = self.dark_dir
        bg_dir = bg_dir if bg_dir.endswith(os.path.sep) else bg_dir + os.path.sep
        paths = sorted(os.listdir(bg_dir))
        paths = [bg_dir + f for f in paths if f.count('tif')]
        return paths

    def frames_available_flat(self):
        exp_dir = self.exp_dir if self.exp_dir.endswith(os.path.sep) else self.exp_dir + os.path.sep
        # these operations are quite expensive for large lists
        paths = [exp_dir + f for f in sorted(os.listdir(exp_dir)) if f.endswith('.tif')]
        return paths

    def get(self, paths):
        """
        Swap a list of paths with a list of data frames
        :param paths:
        :return:
        """
        from tifffile import imread
        # The tiffs from the beamline are non-canonical, supress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = [imread(f).astype(np.uint16) for f in paths]

        return frames

class TiffDealer(_TiffIOMixin, FrameDealer):
    pass

class TiffDealerMPI(_TiffIOMixin, FrameDealerMPI):
    pass

class IAServerDealer(FrameDealerMPI):
    """
    Basically uses the interaction Server's object dict as file system
    """
    def __init__(self, iadict = None, **kwargs):
        super().__init__(**kwargs)
        self._store = iadict if iadict is not None else {}

    def darks_available_flat(self):
        """
        return all darks frame paths
        """
        return sorted([k for k in self._store.keys() if k.startswith(self.dark_dir) and 'image' in k])


    def get(self, paths):
        """
        Swap a list of paths with a list of data frames
        """
        return [self._store[p] for p in paths]

    def frames_available_flat(self):
        """
        return all frame paths available in ascending flat index
        """
        return sorted([k for k in self._store.keys() if k.startswith(self.exp_dir) and 'image' in k])

class ZMQDealer(FrameDealer):
    """
    This Dealer assumes that the Frames are held externally
    """
    def __init__(self, host,
                 port,
                 exp_path,
                 bg_path,
                 **kwargs):
        pass



class AutoCRC(object):

    def __init__(self, out=256, pad=3 * 256):

        self.crow = None
        self.ccol = None
        self._configure(sh_out, sh_pad)

    def _configureself(self, out=256, pad=3 * 256):

        self.pad = pad
        self.out = out

        self.ratio = np.float(pad) / np.float(out)
        self.use_spline = (pad % out != 0)
        if self.use_spline:
            self.ratio = np.float(pad) / np.float(out)
        else:
            self.ratio = pad / out

    def mpi_prepare(data, autocenter=True, mask=None):
        """
        Find center of mass and the corners
        """
        pass
