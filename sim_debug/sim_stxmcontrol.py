import os
import time
import datetime
import socket
import struct
import logging
import datetime
import sys, os
import numpy as np
from math import ceil
from cosmic.camera import fccd

"""
NOTES

Double exposure is not working properly yet.
"""

today = datetime.date.fromtimestamp(time.time())

PARAMS = dict(
    # parameters for the frontend
    nstream=dict(
        step=0.03,  # mum
        num=15,  # number of
        bnum=5,  # num of dark points for each axes
        dwell=(100, 100),  # msec
        energy=800,  # ev
    ),
    comm=dict(
        udp_ip='127.0.0.1',
        udp_port=49203,
        cinc_ip='127.0.0.1',
        # cinc_ip = '10.0.0.16',
        cinc_port=8880,
        #basepath='/tmp/tmpscan/'
        basepath='/Users/benders/Globus/'
    )
)
# QT kill switch
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

DELAY_100Hz = 1.8e-5
DELAY_100Hz = 1.8e-5


class DummySignal(object):

    def __init__(self, tpe=str):
        self.tpe = tpe

    def emit(self, msg='signal'):
        print(self.tpe(msg))


try:
    from PyQt4 import QtCore

    __has_qt4 = True
    Thread = QtCore.QThread
    Signal = QtCore.pyqtSignal
except:
    __has_qt4 = False
    import threading

    Thread = threading.Thread
    Signal = DummySignal


def eV2mum(eV):
    """\
    Convert photon energy in eV to wavelength (in vacuum) in micrometers.
    """
    wl = 1. / eV * 4.1356e-7 * 2.9998 * 1e6

    return wl


def abs2(X):
    return np.abs(X) ** 2


class FccdSimulator(Thread):
    """Simulates STXM control and FCCD camera."""
    statusMessage = Signal(str)
    simulationDone = Signal()

    def __init__(self, params=None, delay_udp=1e-7, add_random_dark=False):
        super(FccdSimulator, self).__init__()

        # private stuff
        self._status = "undefinded"
        self._status_color = "orange"
        self._stop = False
        self._do_scan = True

        # these could be part of input but not entirely important
        self.seed = 1983
        self.udp_packet_size = 4104
        self.udp_header_size = 8
        #self.shape = (1920, 960)  # (1940,1152)
        #self.fCCD = fccd.FCCD()
        self.shape = (980, 960)  # (1940,1152)
        self.fCCD = fccd.FCCD(nrows=self.shape[0]//2)
        self.psize = 30
        self.offset = 10000
        self.io_noise = 5
        self.photons_per_sec = 2e7
        self.resolution = 0.005
        self.dist = 80000
        self.adu_per_photon = 34
        self.zp_dia_outer = 30  # pixel  on screen
        self.zp_dia_inner = 12  # pixel on screen
        self.nanoball_rad = 5  # nanoball radius  (pixel)

        # DEADFOOD
        self.end_of_frame_msg = b"\xf1\xf2" + b"\xde\xad\xf0\x0d" + b"\x00\x00"

        # load parameters
        if params is not None:
            self.p = params
        else:
            from copy import deepcopy
            self.p = deepcopy(PARAMS)

        self.energy = self.p['nstream']['energy']

        self.photons = [self.photons_per_sec * d / 1000 for d in self.p['nstream']['dwell']]

        # Configuration
        self.udp_address = (self.p['comm']['udp_ip'], self.p['comm']['udp_port'])

        # Create stxm controller
        cin_address = (self.p['comm']['cinc_ip'], self.p['comm']['cinc_port'])
        base_path = self.p['comm']['basepath']
        self.stxm_control = STXMControlComm(cin_address, base_path)

        self.delay = delay_udp
        self.add_random_dark = add_random_dark

        # calculate sim shape to resolution
        a = np.int(self.dist * eV2mum(self.energy) / (self.psize * self.resolution))
        assert a < np.min(self.shape), 'Too many pixel, choose larger resolution'
        for i in [256, 384, 512, 640, 768, 896, 1024]:
            if i > a:
                a = i
                break

        a = 384
        self.sim_shape = (a, a)
        self.status = "Simulation resolution is %d x %d" % self.sim_shape

        # make test frame
        X, Y = np.indices(self.shape)
        Y = (Y // 144) * 10
        Y[self.shape[0] // 2:, :] *= -1
        Y += Y.min()
        self.testframe = Y.astype(np.uint16)

        self.darkframes = None
        self.dataframes = None

    def make_ptycho_data(self):
        self.status = "Preparing ptycho data .."
        self.create_darks()
        self.status = "Prepared dark data .."
        self.create_data()
        self.status = "Prepared ptycho data .."

    def create_darks(self):
        N = self.p['nstream']['bnum'] ** 2
        self.darkframes = self._draw(np.zeros((N,) + self.shape).astype(int))

    def create_data(self):
        """ makes a raster ptycho scan """
        # seed the random generatot to fixed value
        np.random.seed(self.seed)

        sh = self.sim_shape

        # positions
        num = self.p['nstream']['num']
        step = self.p['nstream']['step']

        pos = np.array([(step * k, step * j) for j in range(num) for k in range(num)])
        pixelpos = np.round(pos / self.resolution).astype(int)
        pixelpos -= pixelpos.min()
        pixelpos += 5

        # make object
        self.status = "Preparing exit waves .."

        osh = pixelpos.max(0) + np.array(sh) + 10
        nb = self.nanoball_object(osh, rad=self.nanoball_rad, num=400)
        nb /= nb.max()
        # nb = np.resize(nb,osh)
        self.ob = np.exp(0.2j * nb - nb / 2.)
        # from matplotlib import pyplot as plt
        # plt.imshow(np.angle(self.ob), cmap='gray')
        # plt.colorbar()
        # plt.show()

        pr = self.stxm_probe(sh, outer=self.zp_dia_outer, inner=self.zp_dia_inner)
        pr /= np.sqrt(abs2(pr).sum())
        self.pr = pr
        # from matplotlib import pyplot as plt
        # plt.imshow(np.angle(pr), cmap='hsv')
        # plt.show()

        a, b = sh
        exits = np.array([self.pr * self.ob[pr:pr + a, pc:pc + b] for (pr, pc) in pixelpos])
        # fs = lambda e : np.fft.fftshift(e,(-2,-1))
        fs = lambda e: np.fft.fftshift(np.fft.fft2(np.fft.fftshift(e))) / np.sqrt(sh[0] * sh[1])

        self.status = "Propagating waves .."
        stack = np.array([abs2(fs(e)) for e in exits])

        self.diffstack = np.random.poisson(stack * self.photons[0]) * self.adu_per_photon

        self.diffstack2 = np.random.poisson(stack * self.photons[1]) * self.adu_per_photon
        # this was too slow
        """
        
        I = np.zeros((len(stack),)+self.shape).astype(int)
        off = [(a-b)/2 for (a,b) in zip(self.shape,sh)]
        I[:,off[0]:off[0]+sh[0],off[1]:off[1]+sh[1]] = stack
        self.dataframes = self._draw(I)
        """
        self.status = "Frame waves to larger detector shape .."
        self.dataframes = [self._draw(self._embed_frame(frame)) for frame in self.diffstack]

    def _embed_frame(self, frame):
        sh = frame.shape
        out = np.zeros(self.shape).astype(int)
        off = [(a - b) // 2 for (a, b) in zip(self.shape, sh)]
        out[off[0]:off[0] + sh[0], off[1]:off[1] + sh[1]] = frame
        return out

    def _draw(self, frames):
        return frames + np.random.normal(loc=self.offset, scale=self.io_noise, size=frames.shape).astype(int)

    def _convert(self, frame):
        frame[frame > 63000] = 63000
        return frame.astype(np.uint16).byteswap().tobytes()
        # return frame.astype(np.uint16).tostring()

    @staticmethod
    def stxm_probe(shape, inner=8, outer=25):

        # d = np.float(np.min(shape))
        X, Y = np.indices(shape).astype(float)
        X -= X.mean()
        Y -= Y.mean()
        R = (np.sqrt(X ** 2 + Y ** 2) < outer).astype(complex)
        r = (np.sqrt(X ** 2 + Y ** 2) > inner).astype(complex)
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(R * r)))

    @staticmethod
    def nanoball_object(shape, rad=5, num=40):
        """ creates nanoballs as transmission """

        def cluster_coords(shape, rad=5, num=40):
            sh = shape

            def pick():
                return np.array([np.random.uniform(0, sh[0] - 1), np.random.uniform(0, sh[1] - 1)])

            coords = [np.array(
                [np.random.randint(sh[0] / 3, 2 * sh[0] / 3 - 1), np.random.randint(sh[0] / 3, 2 * sh[0] / 3 - 1)])]
            # np.rand.uniform(0,1.,tuple(sh)):
            for ii in range(num - 1):
                noresult = True
                for k in range(10000):
                    c = pick()
                    dist = np.sqrt(np.sum(abs2(np.array(coords) - c), axis=1))
                    if (dist < 2 * rad).any():
                        continue
                    elif (dist >= 2 * rad).any() and (dist <= 3 * rad).any():
                        break
                    elif (0.001 + np.sum(8 / (dist ** 2)) > np.random.uniform(0, 1.)):
                        break

                coords.append(c)
            return np.array(coords)

        sh = shape
        out = np.zeros(sh)
        xx, yy = np.indices(sh)
        coords = cluster_coords(sh, rad, num)
        for c in coords:
            h = rad ** 2 - (xx - c[0]) ** 2 - (yy - c[1]) ** 2
            h[h < 0] = 0.
            out += np.sqrt(h)

        return out

    def listen_for_greeting(self):
        """Open a socket for sending UDP packets to the framegrabber."""

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.udp_address)
        self.status = "Socket open on (%s) on %d " % self.udp_address

        self.status = "Awaiting Handshake"
        data, addr = self.sock.recvfrom(255)

        self.status = "Received %s from (%s) on %d,  " % (str(data), addr[0], addr[1])
        self.fg_addr = addr

        self.sock.connect(addr)

    def make_header(self, port, length, frame_number, packet_number):
        """Returns header."""
        header = struct.pack('!BBHHH', packet_number, 0, port, length, frame_number)
        return header

    def udpsend(self, packet):
        # self.sock.sendto(packet,self.fg_addr)
        self.sock.send(packet)

    def send_frame_in_udp_packets(self, frame, frame_number):
        """Chop frame into small UDP packets and send them out through connected socket."""
        # frame = self._convert(self.fCCD.scramble(frame))
        c = self.fCCD

        # create a frame in byte stream. Cut off a bit at the end and attach ending message
        frame = self._convert(c._rawXclock(c._clockXrow(c._rowXccd(frame))))[:-200]
        frame += self.end_of_frame_msg
        psize = self.udp_packet_size - self.udp_header_size
        print(frame_number, (len(frame) // psize + 1))
        ip, port = self.udp_address
        try:
            for i in range(len(frame) // psize + 1):
                time.sleep(self.delay)
                h = self.make_header(port, psize + self.udp_header_size, frame_number, i % 256)
                packet = h + frame[i * psize:(i + 1) * psize]
                bytes_sent = self.udpsend(packet)  # , (self.ip, self.port))
                # print frame_number, bytes_sent, packet[:8]

            # optional blurbs / hickups
            packet = self.make_header(port, 48, frame_number, 0) + 32 * b"\x00" + self.end_of_frame_msg
            bytes_sent = self.udpsend(packet)  # , (self.ip, self.port))
            bytes_sent = self.udpsend(packet)  # , (self.ip, self.port))
        except socket.error:
            self.status = "Connection error. Restarting ..."
            self.sock.close()
            self.listen_for_greeting()

    def run(self):
        """This triggers the event loop."""
        scan_num = 0
        j = 0
        t0 = datetime.datetime.now()
        STXM = self.stxm_control
        conn = self.listen_for_greeting()
        # Start the event loop
        while not self._stop:

            # Stop producing data when total nr. of frames is reached
            """
            if j >= self.ntotal:
                self._stop = True
                self.simulationDone.emit()
            """
            if not self._do_scan or not (j % 20 == 0):
                time.sleep(0.5)
                self.send_frame_in_udp_packets(self.testframe, j)
            else:
                # pause for moving in the detector
                self.status = "Moving in CCD detector"
                time.sleep(3)

                j = 0
                self.status = "Closing shutter"
                time.sleep(.2)

                # dark frames
                self.status = "Taking dark frames"
                dr = STXM.get_next_dir_name(scan_num=scan_num)
                os.makedirs(dr)
                time.sleep(.5)

                STXM.comm_sendScanInfo(self.p['nstream'])
                STXM.comm_turnOnOffFastCCDCamera(True)
                STXM.comm_StartRegion(self.p['nstream']['dwell'])
                for k, frame in enumerate(self.darkframes):
                    self.send_frame_in_udp_packets(frame, j + k)

                j += k
                # actual frames
                self.status = "Opening shutter"
                time.sleep(.2)

                self.status = "Taking exp frames"
                dr = STXM.get_next_dir_name(scan_num=scan_num)
                os.makedirs(dr)
                time.sleep(.5)

                STXM.comm_turnOnOffFastCCDCamera(True)
                STXM.comm_StartRegion(self.p['nstream']['dwell'])
                if self.dataframes is not None:
                    for k, frame in enumerate(self.dataframes):
                        self.send_frame_in_udp_packets(frame, j + k)
                else:
                    for k, frame in enumerate(self.diffstack):
                        self.send_frame_in_udp_packets(
                            self._draw(self._embed_frame(frame)), j + k)

                STXM.comm_turnOnOffFastCCDCamera(False)
                scan_num += 1

                self.status = "Moving detector out"
                time.sleep(3)
            # Update counters
            j += 1

            """
            time.sleep(0.1)
            
            # Update status (speed)
            if not (j % 100):
                self.status = "Sending at %.2fHz (frame %d)" %(float(j) / (datetime.datetime.now() - t0).total_seconds(), j)
            """

    def stop(self):
        """Stop the event loop."""
        self.status_emit("Fccd is stopping the event loop", "red")
        self._stop = True

    def status_emit(self, status, color=None):
        """Change and emit status signal `status. If `color` is None use default"""
        c = color if color is not None else self._status_color
        self._status = status
        self.statusMessage.emit("<font color=\"" + c + "\">" + status + "</font>")
        logging.info(status)
        print(status)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self.status_emit(status)


class STXMControlComm(object):
    """ simple class to emulate commands send from stxm control """

    MSGLEN = 256
    MSGEND_RECV = b"\n\r"
    MSGEND_SEND = b"\r\n"  # THIS IS ONE WEIRD INCONSISTENCY

    def __init__(self, address=("127.0.0.1", 8888), basepath='/tmp/Data/'):

        self.addr = address
        self.basepath = basepath

    def get_next_dir_name(self, path=None, datefmt='%y%m%d', scan_num=0):
        """ 
        Mimics automatic path creatio, although I do not quite understand
        teh code in STXM control
        
        """
        s = os.path.sep
        if path is None:
            dt = today.strftime(datefmt)

            path = self.basepath + s + dt + s + dt + '%03d' % scan_num + s
        else:
            if path.endswith(s):
                path += s

        if not os.path.exists(path):
            os.makedirs(path)

        lst = [l for l in os.listdir(path) if os.path.isdir(path + l)]
        self.path = path + '%03d' % (1 + len(lst))
        return self.path

    def stxm_comm(self, stxmsocket, msg):
        """ lazy helper function to send and receive """

        stxmsocket.send(bytes(msg,'UTF-8') + self.MSGEND_SEND)
        # Maybe we would want to flush here probably.

        # join response
        piece = stxmsocket.recv(self.MSGLEN)
        answer = b"" + piece
        while piece.find(self.MSGEND_RECV) < 0:
            piece = stxmsocket.recv(self.MSGLEN)
            answer += piece

        # return response line ending \r and \n printed out 
        return repr(answer.decode('UTF-8'))

    def comm_turnOnOffFastCCDCamera(self, bTurnOn, dwell=100):
        """ A Python remake of the original in scan.cpp """
        """ the original function includes dwell time, but I don't know why """
        on = bool(bTurnOn)

        s = socket.create_connection(self.addr)
        comm = lambda x: self.stxm_comm(s, x)

        if on:
            msg = 'setCapturePath ' + self.path
        else:
            msg = 'setCapturePath '

        print(comm(msg))

        if on:
            msg = 'setCaptureMode continuous'
        else:
            time.sleep(.1)
            time.sleep(dwell / 100. + 0.1)
            msg = 'setCaptureMode single'

        print(comm(msg))

        if on:
            print(comm('setCapturePath ' + self.path))
            print(comm('setExternalTriggerMode'))
            print(comm('startCapture'))
        else:
            print(comm('setInternalTriggerMode'))
            print(comm('stopCapture'))
            print(comm('setDoubleExpCount 0'))

        s.close()

    def comm_sendScanInfo(self, simdict):
        """ Mimics communication in sendScanInfo in CCDscan.cpp """
        s = socket.create_connection(self.addr)

        step, num, bnum, dwell, energy = (simdict[k] for k in ('step', 'num', 'bnum', 'dwell', 'energy'))
        out = "sendScanInfo "
        out += "pos_x %.6g, pos_y %.6g, " % (0.0, 0.0)
        out += "step_size_x %.5g, step_size_y %.5g, " % (step, step)
        out += "num_pixels_x %d, num_pixels_y %d, " % (num, num)
        out += "background_pixels_x %d, background_pixels_y %d, " % (bnum, bnum)
        out += "dwell1 %.3g, dwell2 %0.3g, " % dwell
        out += "energy %.5g," % energy
        out += "isDoubleExp 0" #% (len(dwell)-1)

        print(self.stxm_comm(s, out))

        s.close()
        # format string was in .cpp ended with '\n' .. here we will us '\n\r' 
        return out

    def comm_StartRegion(self, dwell, iExpCount=0):
        """ Mimics communication in StartRegion in CCDscan.cpp """
        s = socket.create_connection(self.addr)
        comm = lambda x: self.stxm_comm(s, x)

        print(comm('setExp %.3g' % dwell[0]))
        print(comm('setExp2 %.3g' % dwell[1]))
        print(comm('setDoubleExpCount %d' % iExpCount))  # No double exposure allowed
        print(comm('resetCounter'))

        s.close()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # from ptypy import utils as u
    S = FccdSimulator()
    # plt.ion()

    # plt.show()
    S.make_ptycho_data()
    # plt.figure();plt.imshow(u.imsave(S.pr))
    # plt.figure();plt.imshow(u.imsave(S.ob,vmin=0.))
    # plt.figure();plt.imshow(np.log10(S.dataframes[0]));plt.colorbar()
    # plt.figure();plt.imshow(np.log10(S.dataframes.sum(0)));plt.colorbar()
    # plt.figure();plt.imshow(S.darkframes[0]);plt.colorbar()
    # plt.show()
    # print type(FccdSimulator)

    S.start()

    if __has_qt4:
        # add the missing clock machine
        QA = QtCore.QCoreApplication([])
        QA.exec_()
