__author__ = 'davidshapiro'

import h5py
import datetime
import numpy as np
import scipy as sc
from ..preprocess.defaults import DEFAULT

__all__ = ['cxi','readCXI','write_CXI_metadata','write_CXI_data']
class cxi(object):

    def __init__(self, cxiFile = None):

        self.beamline = None
        self.facility = None
        self.energy = None
        self.ccddata = None
        self.probe = None
        self.imageProbe = None
        self.probemask = None ##Fourier mask
        self.probeRmask = None ##Real space mask
        self.illuminationIntensities = None
        self.datamean = None
        self.stxm = None
        self.stxmInterp = None
        self.xpixelsize = None
        self.ypixelsize = None
        self.translation = None
        self.image = None
        self.startImage = None
        self.bg = None
        self.beamstop = None
        self.corner_x = None
        self.process = None
        self.time = None
        self.indices = None
        self.goodFrames = None
        self.corner = None
        self.indices = None
        self.error = None
        self.__delta = 0.0001

        if cxiFile == None:
            self.process = DEFAULT.copy()
        else:
            try: f = h5py.File(cxiFile,'r')
            except IOError:
                print("readCXI Error: no such file or directory")
                #return
            else:
                print("Loading file contents...")
                try: self.beamline = f['entry_1/instrument_1/name'].value
                except KeyError: self.beamline = None
                try: self.facility = f['entry_1/instrument_1/source_1/name'].value
                except KeyError: self.facility = None
                try: self.energy = f['entry_1/instrument_1/source_1/energy'].value
                except KeyError: self.energy = None
                try: self.ccddata = f['/entry_1/instrument_1/detector_1/data'].value
                except KeyError:
                    print("Could not locate CCD data!")
                    self.ccddata = None
                # try: self.imageProbe = f['entry_1/instrument_1/image_1/probe'].value
                # except KeyError: self.imageProbe = None
                try: self.probemask = f['entry_1/instrument_1/detector_1/Probe Mask'].value
                except:
                    try: self.probemask = f['entry_1/instrument_1/detector_1/probe_mask'].value
                    except KeyError: self.probemask = None
                try: self.probeRmask = f['entry_1/instrument_1/detector_1/probe_Rmask'].value
                except KeyError: self.probeRmask = None
                try: self.datamean = f['entry_1/instrument_1/detector_1/Data Average'].value
                except KeyError: self.datamean = None
                try: self.stxm = f['entry_1/instrument_1/detector_1/STXM'].value
                except KeyError: self.stxm = None
                try: self.stxmInterp = f['entry_1/instrument_1/detector_1/STXMInterp'].value
                except KeyError: self.stxmInterp = None
                try: self.ccddistance = f['entry_1/instrument_1/detector_1/distance'].value
                except KeyError: self.ccddistance = None
                try: self.xpixelsize = f['entry_1/instrument_1/detector_1/x_pixel_size'].value
                except KeyError: self.xpixelsize = None
                try: self.ypixelsize = f['entry_1/instrument_1/detector_1/y_pixel_size'].value
                except KeyError: self.ypixelsize = None
                try: self.translation = f['entry_1/instrument_1/detector_1/translation'].value
                except KeyError: self.translation = None
                try: self.illuminationIntensities = f['entry_1/instrument_1/detector_1/illumination_intensities'].value
                except KeyError: self.illuminationIntensities = None

                entryList = [str(e) for e in list(f['entry_1'])]
                if 'image_latest' in entryList:
                    image_offset = 1
                else: image_offset = 0
                try: currentImageNumber = str(max(loc for loc, val in enumerate(entryList) if not(val.rfind('image'))) - image_offset)
                except:
                    print("Could not locate ptychography image data.")
                    self.image = None
                    try: self.probe = f['entry_1/instrument_1/detector_1/probe'].value
                    except KeyError: self.probe = None
                    self.imageProbe = self.probe
                    self.bg = None
                else:
                    print("Found %s images" %(int(currentImageNumber)))
                    self.image = []
                    for i in range(1,int(currentImageNumber) + 1): 
                        print("loading image: %s" %(i))
                        self.image.append(f['entry_1/image_' + str(i) + '/data'].value)
                    try: self.imageProbe = f['entry_1/image_' + currentImageNumber + '/process_1/final_illumination'].value
                    except:
                        try: self.imageProbe = f['entry_1/instrument_1/detector_1/probe'].value
                        except KeyError:
                            try: self.imageProbe = f['entry_1/instrument_1/detector_1/Probe'].value
                            except KeyError:
                                self.imageProbe = None
                    try: self.bg = f['entry_1/image_' + currentImageNumber + '/process_1/final_background'].value
                    except: self.bg = None
                    self.probe = self.imageProbe.copy()

                try: self.dataProbe = f['entry_1/instrument_1/source_1/data_illumination'].value
                except KeyError: self.dataProbe = None
                try: self.startImage = f['entry_1/image_1/startImage'].value
                except KeyError: self.startImage = None
                try: self.beamstop = f['entry_1/instrument_1/detector_1/Beamstop'].value
                except KeyError: self.beamstop = None
                try: self.corner_x,self.corner_y,self.corner_z = f['/entry_1/instrument_1/detector_1/corner_position'].value
                except KeyError: self.corner_x,self.corner_y,self.corner_z = None, None, None

                self.process = Param().param
                if 'entry_1/process_1/Param' in f:
                    for item in list(f['entry_1/process_1/Param']):
                        self.process[str(item)] = str(f['/entry_1/process_1/Param/'+str(item)][()])
                try: self.time = f['entry_1/start_time'].value
                except KeyError: self.time = None
                try: self.indices = f['entry_1/process_1/indices'].value
                except KeyError: self.indices = None
                try: self.goodFrames = f['entry_1/process_1/good frames'].value
                except KeyError: self.goodFrames = None
                if 'entry_1/image_1/probe' in f:
                    self.probe = f['entry_1/image_1/probe'].value
                if '/entry_1/instrument_1/detector_1/corner_position' in f:
                    self.corner = f['/entry_1/instrument_1/detector_1/corner_position'].value
                else: self.corner = None
                f.close()

    def help(self):
        print("Usage: cxi = readCXI(fileName)")
        print("cxi.beamline = beamline name")
        print("cxi.facility = facility name")
        print("cxi.energy = energy in joules")
        print("cxi.ccddata = stack of diffraction data")
        print("cxi.probe = current probe")
        print("cxi.dataProbe = probe estimated from the data")
        print("cxi.imageProbe = probe calculated from the reconstruction")
        print("cxi.probemask = probe mask calculated from diffraction data")
        print("cxi.datamean = average diffraction pattern")
        print("cxi.stxm = STXM image calculated from diffraction data")
        print("cxi.stxmInterp = STXM image interpolated onto the reconstruction grid")
        print("cxi.xpixelsize = x pixel size in meters")
        print("cxi.ypixelsize = y pixel size in meters")
        print("cxi.translation = list of sample translations in meters")
        print("cxi.image = reconstructed image")
        print("cxi.bg = reconstructed background")
        print("cxi.process = parameter list used by the pre-processor")
        print("cxi.time = time of pre-processing")
        print("cxi.indices = array of STXM pixel coordinates for each dataset")
        print("cxi.goodFrames = boolean array indicating good frames")
        print("cxi.startImage = image which started the iteration")
        print("cxi.corner_x/y/z = positions of the CCD corner")

    def pixm(self):

        l = (1239.852 / (self.energy / 1.602e-19)) * 1e-9
        NA = np.sqrt(self.corner_x**2 + self.corner_y**2) / np.sqrt(2.) / self.corner_z
        #NA = np.arctan(self.corner_x / self.corner_z) ##assuming square data
        return l / 2. / NA

    def removeOutliers(self, sigma = 3):
    
        nPoints = len(self.translation)
        indx = self.indices
        ny, nx = self.stxm.shape
        gy, gx = np.gradient(self.stxm)
        gy = self.stxm - sc.ndimage.filters.gaussian_filter(self.stxm, sigma = 0.25)    
        gy = gy[::-1,:].flatten()  ##puts it in the same ordering as ccddata, starting lower left
        delta = 8. * gy.std()
        badIndices = np.where(gy < (gy.mean() - delta))[0] ##the min Y gradient is one row below the bad pixel
        
        #  badIndices = []
        # for i inrange(1,len(gy)-1):
        
        self.stxm = self.stxm[::-1,:].flatten()
        k = 0
        if len(badIndices) > 0:
            for item in badIndices:
                self.stxm[item] = (self.stxm[item + 1] + self.stxm[item - 1]) / 2.
                if indx[item] > 0:
                    indx[item] = 0
                    indx[item+1:nPoints] = indx[item+1:nPoints] - 1
                else: indx[item] = 0
                self.translation = np.delete(self.translation, item - k, axis = 0)
                self.ccddata = np.delete(self.ccddata, item - k, axis = 0)
                k += 1
        self.stxm = np.reshape(self.stxm,(ny,nx))[::-1,:]
        print("Removed %i bad frames." %(len(badIndices)))

    def imageShape(self):

        ny, nx = self.ccddata[0].shape
        pixm = self.pixm()
        y,x = np.array((self.translation[:,1], self.translation[:,0]))
        y = (y / pixm).round() + ny / 2
        x = (x / pixm).round() + nx / 2
        pPosVec = np.column_stack((y,x))

        dx = pPosVec[:,1].max() - pPosVec[:,1].min() + 2
        dy = pPosVec[:,0].max() - pPosVec[:,0].min() + 2

        return dy + ny, dx + nx

    def pixelTranslations(self):

        pixm = self.pixm()
        ny, nx = self.ccddata[0].shape
        y,x = np.array((self.translation[:,1], self.translation[:,0]))
        y = (y / pixm).round() + ny / 2
        x = (x / pixm).round() + nx / 2
        return np.column_stack((y,x))

    def dataShape(self):

        return self.ccddata.shape

    def illumination(self):

        """
        Public function which computes the overlap from a stack of probes.  This is equivalent to the total illumination profile
        Input: Stack translation indices and the probe
        Output: Illumination profile
        """
        #TODO: verify that this is correct for multimode
        # isum = np.zeros(self.oSum.shape)
        # for k in range(self.oModes):
        #     for i in range(self.__nFrames):
        #         j = self.__indices[i]
        #         isum[k,j[0]:j[1],j[2]:j[3]] = isum[k,j[0]:j[1],j[2]:j[3]] + np.reshape(abs(self.probe.sum(axis = 1)), (self.ny, self.nx))
        # return isum
        qnorm = self.QPH(self.QP(np.ones(self.ccddata.shape, complex)))
        return self.QH(np.abs(qnorm)) + self.__delta**2

    def QP(self, o):

        """
        Private function which multiplies the frames by the probe
        Input: stack of frames
        Output: stack of frames times the probe
        """

        return o * self.probe

    def QPH(self, o):

        """
        Private function which multiplies the frames by the conjugate probe.
        Input: stack of frames
        Output: stack of frames times the conjugate probe
        """

        return o * self.probe.conjugate()

    def QH(self, ovec):

        """
        Private function which computes the overlap from stack of frames.
        Input: Stack translation indices and the stack of frames
        Output: Total object image
        """
        self.ny, self.nx = self.probe.shape
        self.__indices = []
        i = 0
        for p in self.pixelTranslations():
            x1 = p[1] - self.nx / 2.
            x2 = p[1] + self.nx / 2.
            y1 = p[0] - self.ny / 2.
            y2 = p[0] + self.ny / 2.
            self.__indices.append((y1,y2,x1,x2))
            i += 1
        
        osum = np.zeros(self.imageShape())  ##this is 3D, (oModes, oSizeY, oSizeX)
        
        for i in range(len(self.ccddata)):
            j = self.__indices[i]
            ##sum the oVec over the probe modes and then insert into the oSum maintaining separation
            ##of the object modes
            osum[j[0]:j[1], j[2]:j[3]] = osum[j[0]:j[1], j[2]:j[3]] + ovec[i,:,:]
        
        return osum

def readCXI(cxiFile):

    cxiObj = cxi(cxiFile)

    return cxiObj

def placeProbe(cxiFile, probe):

    """
    Places probe in detector_1 group of the cxiFile
    """
    f = h5py.File(cxiFile, 'a')
    del f['entry_1/instrument_1/detector_1/probe']
    dset = f.create_dataset('entry_1/instrument_1/detector_1/probe', data = probe)


    f.close()

def placeIntensities(cxiFile, intensities):

    """
    Places probe in detector_1 group of the cxiFile
    """
    f = h5py.File(cxiFile, 'a')
    del f['entry_1/instrument_1/detector_1/illumination_intensities']
    dset = f.create_dataset('entry_1/instrument_1/detector_1/illumination_intensities', data = intensities)


    f.close()


def write_CXI_metadata(cxiObj, fileName):

    print("Writing metadata to cxi...")

    f = h5py.File(fileName, "a")
    f.create_dataset("cxi_version",data = 130)
    entry_1 = f.require_group("entry_1")
    entry_1.create_dataset("start_time",data=datetime.date.today().isoformat())

    instrument_1 = entry_1.require_group("instrument_1")
    instrument_1.create_dataset("name",data=cxiObj.beamline)
    source_1 = instrument_1.create_group("source_1")
    source_1.create_dataset("energy", data=cxiObj.energy) # in J
    source_1.create_dataset("name",data="ALS")
    source_1.create_dataset("probe_mask", data = cxiObj.probemask)
    source_1.create_dataset("probe", data = cxiObj.probe)
    source_1.create_dataset("data_illumination", data = cxiObj.probe)

    detector_1 = instrument_1.require_group("detector_1")

    detector_1['data'].attrs['axes'] = "translation:y:x"
    detector_1.create_dataset("probe", data = cxiObj.probe)
    detector_1.create_dataset("data_illumination", data = cxiObj.probe)
    detector_1.create_dataset("illumination_intensities", data = cxiObj.illuminationIntensities)
    detector_1.create_dataset("probe_mask", data = cxiObj.probemask)
    detector_1.create_dataset("probe_Rmask", data = cxiObj.probeRmask)
    detector_1.create_dataset("Data Average", data = cxiObj.datamean)
    detector_1.create_dataset("STXM", data = cxiObj.stxm)
    detector_1.create_dataset("STXMInterp", data = cxiObj.stxmInterp)
    detector_1.create_dataset("distance", data=cxiObj.corner_z) # in meters
    detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    detector_1.create_dataset("x_pixel_size", data = cxiObj.xpixelsize) # in meters
    detector_1.create_dataset("y_pixel_size", data = cxiObj.ypixelsize) # in meters
    detector_1.create_dataset("corner_position", data = (cxiObj.corner_x, cxiObj.corner_y, cxiObj.corner_z)) # in meters
    sample_1 = entry_1.create_group("sample_1")
    geometry_1 = sample_1.create_group("geometry_1")
    geometry_1.create_dataset("translation", data = cxiObj.translation)

    data_1 = entry_1.create_group("data_1")
    data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
    data_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

    process_1 = entry_1.create_group("process_1")
    process_1.create_dataset("command",data=" ")
    process_1.create_dataset("indices", data = cxiObj.indices)
    paramGroup = process_1.create_group('Param')
    dsets = []
    for item in cxiObj.process:
        dsets.append(paramGroup.create_dataset(item, data = cxiObj.process[item]))
    f.close()
    return 0

def write_CXI_data(data_chunk, fileName):

  
    print(("Writing a " + str(data_chunk.shape) + " data chunk to cxi..."))


    data_path = "/entry_1/instrument_1/detector_1/data"

    f = h5py.File(fileName, "a")

    #if data exist we add a new chunk
    if  data_path in f:

        all_data = f[data_path]
        all_data.resize(all_data.shape[0] + data_chunk.shape[0], axis=0)
        all_data[all_data.shape[0] - data_chunk.shape[0]:] = data_chunk.astype('float32')

    #otherwise, we create the group and add the first chunk
    else:    
        f.create_dataset(data_path, data = data_chunk.astype('float32'), maxshape=(None,) + data_chunk.shape[-2:], chunks=True, compression='gzip')

    print(("Accumulated data size = " + str(f[data_path].shape)))

    f.close()

    return 0

