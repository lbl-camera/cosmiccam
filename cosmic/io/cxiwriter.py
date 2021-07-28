import h5py

class CXIWriter:
    """A class for writing CXI files.

    Example
    -------
    TODO
    """
    def __init__(self, filename, version=140):
        """Opens a new CXI file and creates a tree.

        Parameters
        ----------
        filename : str
             Name of CXI file.
        version : int
             Version of CXI file format, default = 140
        """
        # Open file
        self._filename = filename
        self._f = h5py.File(filename, "w")

        # Create CXI tree
        self._f.create_dataset("cxi_version",data=140)
        self._entry_1      = self._f.create_group("entry_1")
        self._instrument_1 = self._entry_1.create_group("instrument_1")
        self._source_1     = self._instrument_1.create_group("source_1")
        self._detector_1   = self._instrument_1.create_group("detector_1")
        self._sample_1     = self._entry_1.create_group("sample_1")
        self._geometry_1   = self._sample_1.create_group("geometry_1")
        self._data_1       = self._entry_1.create_group("data_1")

        # Create soft links
        self._detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
        self._data_1["data"]            = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
        self._data_1["translation"]     = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

    def write_metadata(self, photon_energy, detector_distance, x_pixel_size, y_pixel_size, corner_position=None):
        """Writes meta data into CXI file.

        Parameters
        ----------
        photon_energy : float
            The photon energy of the experiment [J]
        detector_distance : float
            The distance between sample and detector [m]
        x_pixel_size : float
            The size of a detector pixel in x [m]
        y_pixel_size : float
            The size of a detector pixel in y [m]
        corner_position : Optional[[float, float, float]]
            X,Y,Z coordinates of the detector corner [m]
        """
        self._source_1.create_dataset("energy", data=photon_energy)
        self._detector_1.create_dataset("distance", data=detector_distance)
        self._detector_1.create_dataset("x_pixel_size", data=x_pixel_size)
        self._detector_1.create_dataset("y_pixel_size", data=y_pixel_size)
        if corner_position is not None:
            self._detector_1.create_dataset("corner_position", data=corner_position)

    def write_illumination(self, data=None, **kwargs):
        """Writes illumination into CXI file.
        
        Parameters
        ----------
        data : Optional[array_like]
             Illumination function as a complex valued ndarray.
        """
        if data is not None:
            self._source_1.create_dataset("illumination", data=data, **kwargs)

    def write_Fillumination_mask(self, data=None, **kwargs):
        """Writes Fillumination mask into CXI file.
        
        Parameters
        ----------
        data : Optional[arary_like]
             Fillumination mask as boolean ndarray.
        """
        if data is not None:
            self._detector_1.create_dataset("Fillumination_mask", data=data, dtype=bool, **kwargs)

    def write_Fillumination_intensities(self, data=None, **kwargs):
        """Writes Fillumination intensities into CXI file.
        
        Parameters
        ----------
        data : Optional[arary_like]
             Fillumination intensities as floating ndarray.
        """
        if data is not None:
            self._detector_1.create_dataset("Fillumination_intensities", data=data, **kwargs)

    def write_initial_image(self, data=None, **kwargs):
        """Writes initial image to CXI file.

        Parameters
        ----------
        data : Optional[array_like]
            Initial image as a complex valued ndarray.
        """
        if data is not None:
            self._detector_1.create_dataset("initial_image",data=data, **kwargs)

    def write_dataframes(self, data=None, shape=None, **kwargs):
        """Writes data frames to CXI file.

        Parameters
        ----------
        data : Optional[array_like]
            Stack of data frames, if None empty dataset with given shape is created.
        shape : Optional[tuple of int]
            Shape of data frames in the form (nframes, ny, nx).
        """
        if data is not None:
            self.frames = self._detector_1.create_dataset("data", data=data, **kwargs)
        else:
            self.frames = self._detector_1.create_dataset("data", shape, chunks=True, **kwargs)
        self.frames.attrs['axes'] = "translation:y:x"

    def insert_dataframe(self, i, data):
        """Insert data frame into CXI file.

        Parameters
        ----------
        i : int
            Index of data frame.
        data : array_like
            Data frame as float array of shape (ny,nx).
        """
        self.frames[i] = data
        
    def write_darkframes(self, data=None, shape=None, **kwargs):
        """Writes dark frames to CXI file.

        Parameters
        ----------
        data : Optional[array_like]
            Stack of dark frames, if None empty dataset with given shape is created.
        shape : Optional[tuple of int]
            Shape of dark frames in the form (nframes, ny, nx).
        """
        if data is not None:
            self.dark = self._detector_1.create_dataset("data_dark", data=data, **kwargs)
        else:
            self.dark = self._detector_1.create_dataset("data_dark", shape, chunks=True, **kwargs)
        self.dark.attrs['axes'] = "translation:y:x"

    def insert_darkframe(self, i, data):
        """Insert dark frame into CXI file.

        Parameters
        ----------
        i : int
            Index of dark frame.
        data : array_like
            Dark frame as float array of shape (ny,nx).
        """
        self.dark[i] = data

    def write_translations(self, data, **kwargs):
        """Writes translations to CXI file.
        
        Parameters
        ----------
        data : Optional[array_like]
             Array of translation with shape (nframes, 3) where slow axis are x,y,z positions [m].
        """
        self._geometry_1.create_dataset("translation", data=data, **kwargs)

    def reopen(self):
        """Reopen the file to append some data"""
        self._f = h5py.File(self._filename, 'a')

    def flush(self):
        self._f.flush()
        
    def close(self):
        """Close the CXI file."""
        self._f.close()
