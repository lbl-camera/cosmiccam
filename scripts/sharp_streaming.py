import os, sys

os.environ["IPYTHONDIR"] = "/tmp"

#if ("OMPI_COMM_WORLD_LOCAL_RANK" in os.environ):
  #print "Local Rank: ", os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
  #os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

import zmq
import h5py
import time
import pickle

import numpy as np
import afnumpy as af
import afnumpy.fft as fft
import sharp

from mpi4py import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

isLeader = (rank == 0)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#print(os.path.dirname(__file__))

class Solver:
    """
    Wrap around the SHARP solver. Initializes the reconstruction.
    Starts the engine.
    """
    def __init__(self, engine,communicator,strategy):
        self.m_engine = engine
        self.m_communicator = communicator
        self.m_opt = sharp.Options.getOptions()
        self.strategy = strategy

    def initialize(self, frames, mean_bg, translations, reciprocal_size,
                   illumination, illumination_mask, illumination_intensities, frame_weights):
        """Initialize the reconstruction. """
        self.m_engine.setTotalFrameCount(frames.shape[0])
        self.m_engine.setFrames(frames)
        self.m_engine.setMeanBackground(mean_bg)
        self.m_engine.setReciprocalSize(reciprocal_size)
        self.m_engine.setTranslations(translations[self.strategy.myFrames(),:])
        self.m_engine.setAllTranslations(translations)
        self.m_engine.setMyFrameWeights(frame_weights)
        self.m_engine.setIllumination(illumination)
        self.m_engine.setIlluminationMask(sharp.af2thrust(illumination_mask))
        #self.m_engine.setIlluminationIntensities(sharp.af2thrust(illumination_intensities))
        self.m_engine.setCommunicator(self.m_communicator)
        return self.m_engine.initialize()

    def run(self, iterations):
        """Run the engine for a given nr. of iterations."""
        tic = time.clock()
        self.m_engine.iterate(iterations)
        toc = time.clock()
        if self.m_communicator.isLeader():
            if self.m_opt.time_solver:
                print("Iteration time: %g us\n", toc-tic)

    def writeImage(self, prefix):
        """Write the results to a CXI file."""
        if self.m_communicator.isLeader():
            if len(self.m_opt.output_file) > 0:
                self.m_input_output.writeCXI(self.m_engine,self.m_opt.output_file)
            else:
                self.m_input_output.writeCXI(self.m_engine)

class Engine(sharp.CudaEngine):
    """The reconstruction engine. Defines what happens in every iteration of the reconstruction."""
    def __init__(self):
        sharp.CudaEngine.__init__(self)
        self.overlapResidual = 0.
        self.dataResidual  = 0.
        self.stop = False

    def init_shapes(self):
        """Defines shapes of frames, images."""
        self.frames_shape = (self.m_nframes, self.m_frame_width, self.m_frame_height)
        self.crop = self.m_frame_width/2

    def init_objects(self):
        """Create afnumpy objects."""
        self.frames_data    = af.zeros(self.frames_shape, dtype=af.complex64)
        self.image_object   = sharp.range2af(self.m_image)
        self.frames_iterate = sharp.range2af(self.m_frames_iterate)
        self.frames_object  = af.copy(self.frames_iterate)
        self.illumination   = sharp.range2af(self.m_illumination)
        self.frames_temp    = sharp.range2af(self.m_frames)
        self.frame_weights  = sharp.range2af(self.m_frame_weights)
        self.m_frames_norm  = 1.

    def iterate(self, steps):
        """Runs the engine for a given nr. of iteration steps."""

        for i in range(steps):
            
            # Stop the engine on request
            if self.stop:
                break

            # True if residials should be calculated
            calculateResidual = (self.m_iteration % self.m_output_period == 0)

            # Data Projector
            if calculateResidual:
                self.dataResidual = self.dataProjector(self.m_frames_iterate, self.frames_data, True)
                frames_norm = af.sqrt(af.sum(af.abs(self.frames_temp * (self.frame_weights).reshape((self.m_nframes,1,1)))))
                #print rank, self.dataResidual / frames_norm
                if frames_norm:
                    self.dataResidual /= frames_norm
            else:
                self.dataProjector(self.m_frames_iterate, self.frames_data)

            # First update (RAAR)
            self.axpbypcz(self.frames_iterate, self.frames_object, self.frames_data, self.frames_iterate, self.m_beta, -self.m_beta,1.-2.* self.m_beta)

            # True if synchronization should be done
            do_sync = (((self.m_iteration % self.m_global_period) == 0) or ((i == steps - 1)))

            # OverlapProjector
            if calculateResidual:
                self.overlapResidual = self.overlapProjector(self.frames_data,
                                                        self.frames_object,
                                                        self.image_object,
                                                        do_sync, True)
                #print rank, self.overlapResidual / frames_norm
                if frames_norm:
                    self.overlapResidual /= frames_norm
            else:
                self.overlapProjector(self.frames_data,
                                      self.frames_object,
                                      self.image_object,
                                      do_sync)

            # Refine Illumination
            if(self.m_iteration > 0 and self.m_illumination_refinement_period > 0 and
               (self.m_iteration % self.m_illumination_refinement_period) == 0):
               self.refineIllumination(self.frames_data,
                                       self.image_object)

            # Second update (RAAR)#
            self.axpby(self.frames_iterate, self.frames_object, self.frames_iterate, 1.0, 2.* self.m_beta)

            # Print Resdiuals
            if self.m_comm.isLeader() and calculateResidual:
                if self.m_has_solution:
                    tmp1 = af.array(self.m_illuminated_area.real * self.solution.conj() * self.image_object)
                    phi = af.sum(tmp1)
                    tmp2 = af.array(self.m_illuminated_area.real * af.abs(self.image_object)**2)
                    image_norm_2 = af.sum(tmp2)
                    imageResidual = (self.m_image_norm_2 * image_norm_2 - (af.abs(phi)**2)) / (self.m_image_norm_2 * image_norm_2)
                    if imageResidual <= 0:
                        imageResidual = 0
                    msg = "iter = %d, data = %06g, overlap = %06g, image = %06g"
                    print(msg %(self.m_iteration, self.dataResidual, self.overlapResidual, imageResidual))
                else:
                    msg = "iter = %d, data = %06g, overlap = %06g"
                    print(msg %(self.m_iteration, self.dataResidual, self.overlapResidual))
                    pass

            # Update iteration counter
            # ------------------------
            self.m_iteration += 1



class Reconstructor:
    def __init__(self):
        self.energy = None
        self.distance = None
        self.num_frames = None
        self.dhalf = None
        self.shape = None
        self.pixelsize = None
        self.run_file = None
        self.wavelength = None

        self.probe = None
        self.probemask = None
        self.probeRmask = None
        self.positions = None
        self.translations = None
  
        self.opt = sharp.Options.getOptions()
        self.engine = Engine()

        self.communicator = None
        self.io = None
        self.illumination = None
        self.Fillumination_mask = None
        self.Fillumination_intensities = None

        self.scanNx = None 
        self.scanStep = None

        self.flipYpositions = False

        self.positions = None

        self.crop = None
        self.iterations = None
        self.reciprocal_size = None

        self.mean_background = None
        self.std_background  = None

        self.strategy = None
        self.myframes = None

        self.frames   = None
        self.frame_weights = None
        self.validframes   = None

        self.initialized = False
        self.counter = 0


    def initialize(self):

        if self.initialized:
            return
        
        print("Initializing service...")
        options = ['']

        #if self.sharpIlluminationRefine:
        #    options.append('-r %d' %self.sharpIlluminationRefine)

        #if self.sharpFourierMask:
        #    options.append('-M')

        options.append('-T')
        options.append('2')
        #options.append('-I')
        options.append('-D')
        options.append('test.cxi')

        print("Arguments: ", options)

        self.opt.parse_args(options) # parse arguments
        self.engine.setWrapAround(self.opt.wrapAround)

        # initialize communicator and io
        self.communicator = sharp.Communicator(options, self.engine)
        self.io = sharp.InputOutput(options, self.communicator)
        self.engine.setInputOutput(self.io)

        # set illuminations
        self.illumination = af.array(self.probe, dtype=af.complex64) 
        self.Fillumination_mask = af.array(self.probemask, dtype=af.complex64)
        #self.Fillumination_intensities = af.array(np.ascontiguousarray(array[2].real))
        
        # positions
        self.positions_gpu = af.array(np.ascontiguousarray(self.positions), dtype=af.float64)

        # preparation
        self.crop = self.shape[0]/2
        self.iterations = self.opt.iterations

        # background
        self.mean_background = af.zeros(self.shape, dtype=af.float32)
        self.std_background  = af.zeros(self.shape, dtype=af.float32)

        # strategy
        self.strategy = sharp.Strategy(self.communicator)
        self.strategy.setTranslation(self.positions_gpu)
        self.strategy.calculateDecomposition()
        self.myframes = self.strategy.myFrames()
        print("rank %d has " %rank, self.myframes)

        # initialize a gpu array for frames
        self.frames   = af.zeros((len(self.myframes), self.shape[0], self.shape[1]), dtype=af.float32)
        self.frame_weights = af.zeros(len(self.myframes), af.complex64)
        self.validframes   = np.zeros(len(self.myframes)).astype(np.bool)

        # initialize solver
        self.solver = Solver(self.engine, self.communicator, self.strategy)
        if self.solver.initialize(self.frames, self.mean_background, self.positions_gpu,
                                  self.reciprocal_size, self.illumination,
                                  self.Fillumination_mask, self.Fillumination_intensities,
                                  self.frame_weights) == -1:
            raise RuntimeError('Solver/Engine has failed to initialize')

        # initialize engine shapes and objects
        self.engine.init_shapes()
        self.engine.init_objects()
        self.initialized = True

    def result(self):
        image = np.ascontiguousarray(self.engine.image_object[self.crop:-self.crop, self.crop:-self.crop])
        illumination = np.array(self.engine.illumination).reshape((self.shape[0], self.shape[0]))
        out_dict = {}
        out_dict["data"] = np.abs(image) # need to prepare vis for complex arrays
        out_dict["illumination"] = np.abs(illumination) # need to prepare vis for complex arrays
        return out_dict

    """
    def send_recons(self):
        if isLeader:
            image = np.ascontiguousarray(self.engine.image_object[self.crop:-self.crop, self.crop:-self.crop])
            illumination = np.array(self.engine.illumination).reshape((self.arraySize, self.arraySize))
            self.send_socket.send_recons(image, illumination, str(self.engine.m_iteration), 
                                         str(self.engine.dataResidual), 
                                         str(self.engine.overlapResidual))

    def continue_reconstruction(self):
        received_total = comm.allreduce(self.received, op=MPI.SUM)
        if (received_total % self.update):
            return False
        if self.iteration >= self.iterTotal:
            return True
        else:
            self.engine.calculateImageScale()
            self.solver.run(self.iterStep)
            self.iteration += self.iterStep
            self.send_recons()
            return False
    """

    def prepare(self, datax):
        if datax["command"] != "prepare":
            return

        print("preparing...")
        data = datax["data"]

        self.energy = data["energy"]
        self.distance = data["distance"]
        self.num_frames = data["num_frames"]
        self.dhalf = data["dhalf"]
        self.shape = data["shape"]
        self.pixelsize = data["pixelsize"]
        self.run_file = data["run_file"]
        self.wavelength = 1.98644521e-25/self.energy
        self.reciprocal_size = (self.pixelsize[0] * self.shape[0]/(self.distance * self.wavelength), self.pixelsize[1] * self.shape[0]/(self.distance * self.wavelength))
        print(self.energy, self.distance, self.num_frames, self.dhalf, self.shape, self.pixelsize, self.wavelength, self.reciprocal_size)

        """
        cxi_obj.xpixelsize = pixelsize[0]
        cxi_obj.ypixelsize = pixelsize[1]
        cxi_obj.energy = energy
        cxi_obj.corner_x = dhalf
        cxi_obj.corner_y = dhalf
        cxi_obj.corner_z = distance
        """
        pass

    def update(self, datax):
        if datax["command"] != "update":
            return

        print("updating...")
        data = datax["data"]
        self.probe = data["probe_init"]
        self.probemask = data["probe_fmask"]
        self.probeRmask = data["probe_mask"]
        self.positions = data["positions"]
        #self.translations = data["translations"]

        #print self.positions.shape
        print(self.positions)
        #print self.positions * self.reciprocal_size
        #print "translations", self.translations
        self.initialize()
        """
        cxi_obj.probe = data["probe_init"]
        cxi_obj.probemask = data["probe_fmask"]
        cxi_obj.probeRmask = data["probe_mask"]
        positions = data["positions"]
        cxi_obj.translation = data["translations"]
        """

    def frame(self, datax):
        if datax["command"] != "frame":
            return
        data = datax["data"]
        index = int(data["num"])
        if index in self.myframes:
            print("rank %d got frame " %rank, index)
            self.frames[self.myframes.index(index),:,:] = af.array(data["data"], dtype=af.float32)
            self.frame_weights[self.myframes.index(index)] = 1.
        if not (self.counter % 5):
            self.engine.setMyFrameWeights(self.frame_weights)
            self.engine.calculateImageScale()
            self.solver.run(10)
        self.counter += 1
        
    def completed(self, datax):
        if datax["command"] == "completed":
            return

        """
        data = np.array(ccddata)
        cxi_obj.ccddata = data
        cxi_obj.datamean = data.mean(0)
        cxi_obj.illuminationIntensities = data.mean(0)
        cxi_obj.indices = np.arange(len(data))

        out_path = "/global/home/hkrish/sim.cxi"
        writeCXI(cxi_obj, fileName = out_path)

        import os
        cmd = "/global/home/hkrish/camera/sharp-wrapper/run_script.sh %s" % out_path
        os.system(cmd)
        f1 = h5py.File(out_path,'r')
        image_prefix = "/entry_1/image_1"
        data = np.abs(np.array(f1[image_prefix + "/data"]))
        """
        print("completed")

def main():
    recon = Reconstructor()
    ctx = zmq.Context()

    sub_socket = ctx.socket(zmq.SUB)
    #sub_socket.connect("tcp://phasis.lbl.gov:9000")
    sub_socket.connect("tcp://localhost:15000")
    sub_socket.setsockopt(zmq.SUBSCRIBE, "")

    result_socket = ctx.socket(zmq.PUB)
    result_socket.connect("tcp://localhost:9500")

    while True:
        msg = sub_socket.recv()
        data_pickle = pickle.loads(msg)
        print(rank, data_pickle["command"])

        if data_pickle["command"] == "setup":
            print(data_pickle)
        
        if data_pickle["command"] == "prepare":
            recon.prepare(data_pickle)

        if data_pickle["command"] == "update":
            recon.update(data_pickle)

        if data_pickle["command"] == "frame":
            recon.frame(data_pickle)

        if data_pickle["command"] == "completed":
            recon.completed(data_pickle)

        if rank == 0 and recon.initialized:
            res = recon.result()
            data_pickle["command"] = "recon"
            data_pickle["data"] = res
            pickle_string = pickle.dumps(data_pickle)
            result_socket.send(pickle_string)
      
if __name__ == "__main__":
    main()
