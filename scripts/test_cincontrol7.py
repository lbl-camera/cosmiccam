from cosmic.ext.ptypy.utils.validator import ArgParseParameter
import time
from PyQt5 import QtCore, QtNetwork
# from cosmic.camera.framegrabber_nozmq import Framegrabber
# from cosmic.camera.fccd import FCCD
from cosmic.camera.cin import CIN
from cosmic.ext.ptypy.io import interaction
import numpy as np
import os
from threading import Thread
from urllib.parse import splitport
from collections import deque, OrderedDict
import tifffile
from cosmic.camera.fccd import FCCD
from cosmic.utils import ScanInfo
import json
from io_control import write_data, write_metadata

import gc
import zmq

term = {
    'default': lambda x: x,
    'red': lambda x: '\x1b[0;31;40m %s \x1b[0m' % str(x),
    'green': lambda x: '\x1b[0;32;40m %s \x1b[0m' % str(x),
    'yellow': lambda x: '\x1b[0;33;40m %s \x1b[0m' % str(x),
    'blue': lambda x: '\x1b[0;34;40m %s \x1b[0m' % str(x),
}

scaninfo_translation = {
    'num_pixels_x': 'exp_num_x',
    'num_pixels_y': 'exp_num_y',
    'step_size_x': 'exp_step_x',
    'step_size_y': 'exp_step_y',
    'background_pixels_x': 'dark_num_x',
    'background_pixels_y': 'dark_num_y',
}

import logging, warnings

logging.getLogger().setLevel(20)

import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

global lastbuf

class Region(object):
    
    def __init__(self, num, buffer=1000):
        self.frames = deque(maxlen=buffer)
        self.is_full = False
        self.num_processed = 0
        self.num = num
        self.dir = ''
        

class Scan(object):
    """
    This class saves all information available at the beamline.
    
    It has its own methods to
    save frames and info to disk or potentially transmit its content to
    another socket (NOT IMPLEMENTED).
    
    Frames are stored as stored as Python buffers in the attributes 
    ``.dark_frames`` and ``.exp.frames`` which are lists for thread
    safety reasons.
    """

    def __init__(self, shape=(1940, 1152), num_dark=400, num_exp=60000, max_frames=30000):
        """
        """
        self.name = ''
        self.stack_idx = ''
        self.meta_dir = ''
        self.info = None
        self.info_raw = ''
        self.dark = Region(num_dark)
        self.exp = Region(num_exp, max_frames)
        self.num_rows = shape[0] // 2
        self.num_adcs = shape[1] // 6
        self.shape = shape  # (2 * num_rows, num_adcs * 6)
        self.fbuffer = None
        self._set_scheme('image%06d')
        self.CCD = FCCD(nrows=self.num_rows)
        # privat
        self._num_offset = 0
        self._thread = None

    def save_meta(self, meta_dir=None):
        mdir = meta_dir if meta_dir is not None else self.meta_dir
        mdir = mdir if mdir else os.path.split(self.exp.dir)[0]
        if not mdir:
            raise AttributeError
        else:
            self.info['dark_dir'] = self.dark.dir
            self.info['exp_dir'] = self.exp.dir
            print("self.name: %s" %self.name)
            print("self.stack_idx: %s" %self.stack_idx)
            print("mdir: %s" %mdir)
            return self.store_obj(mdir+'/%s_%s_info' % (self.name, self.stack_idx), self.info)

    def _set_scheme(self, scheme):

        self.scheme = scheme if scheme.startswith(os.path.sep) else os.path.sep + scheme

    def frame_to_image(self, frame):

        num, buf = frame
        npbuf = np.frombuffer(buf, '<u2')
        npbuf = npbuf.reshape((12 * self.num_rows, self.num_adcs))
        image = self.CCD.assemble2(npbuf.astype(np.uint16))

        return num, image

    def process_region(self, region):

        path = region.dir + self.scheme

        processed = {}
        tries = 0
        for i in range(region.num):
            while True:
                try:
                    d = region.frames.popleft()
                    tries = 0
                    break
                except IndexError:
                    d = None
                    tries += 1
                    time.sleep(0.1)
                    if region.is_full and tries > 3:
                        break

            if region.is_full and d is None:
                break
            else:
                num, im = self.frame_to_image(d)
                self.store_image(path % i, im)
                region.num_processed += 1
                processed[i] = num

        self.store_obj(os.path.split(path)[0] + '/original_indices', processed)
        return processed

    def store_obj(self, path, obj):
        fname = path+'.json'
        print("fname: %s" %fname)
        f = open(fname, 'w')
        json.dump(obj, f)
        f.close()
        return fname

    def store_image(self, path, im):
        fname = path+'.tif'
        tifffile.imsave(fname, im)
        return fname

    def save_data(self):
        self.process_region(self.dark)
        self.process_region(self.exp)

    def auto_save_data(self):
        if not self._thread:
            self._thread = Thread(target=self.save_data)
            self._thread.daemon = True
            self._thread.start()

class ClientScan(Scan):

    def __init__(self, address, **kwargs):
        host, port = splitport(address)
        self.client = interaction.Client(address=host, port=port)
        self.client.activate()

        super().__init__(**kwargs)

        while not self.client.connected:
            time.sleep(0.1)


    def store_image(self, path, im):
        self.client.set(path, im)
        return path

    def store_obj(self, path, obj):
        self.client.set(path, obj)
        return path

    def save_data(self):
        self.process_region(self.dark)
        self.process_region(self.exp)
        self.client.wait()
        self.client.stop()

from io import StringIO

DEFAULT = ArgParseParameter(name='Grabber')
DEFAULT.load_conf_parser(StringIO(
    """[shape]
    default = (1040,1152)
    help = (H,W) of the detector in pixel including overscan pixels
    
    [tcp_addr]
    default = "131.243.73.179:8880"
    help = IP:Port address of command tcp server.
    
    [statusterm]
    default = "/dev/pts/1"
    help = Second terminal pointer to display status updates.
    
    [iaserver]
    default = None
    help = Interaction server parameters
    
    [framegrabber]
    default = None
    help = Framegrabber init parameters
    """))

# DEFAULT_framegrabber = ArgParseParameter(name='framegrabber')
DEFAULT.children['framegrabber'].load_conf_parser(StringIO(
    """[udp_addr]
    default = "10.0.5.207:49203"
    help = ip:port type address of the udp frame server
    
    [pub_addr]
    default = "127.0.0.1:49206"
    help = ip:port the framegrabber publishe frames. [NOT IMPLEMENTED]
    
    [read_addr]
    default = "10.0.0.16:49207"
    help = ip:port type address the grabber contacts the udp server from.
    
    [timeout]
    default = 50
    help = Blocking timeout on receive in ms. If set to zero, the framegrabber
      will receive and discard an additional frame. Otherwise chose a timeout
      that is smaller than the waiting time in between scans to avoid overlap.
    """))

DEFAULT.children['iaserver'].load_conf_parser(StringIO(
    """[ia_addr] 
    default = None
    help = Default address for primary connection of shape "host:port"
    """
))

def splitaddr(addr):
    host, port = splitport(addr)
    if port is None:
        return host, port
    else:
        return host,int(port)

from zmq.utils.monitor import recv_monitor_message

def wait_for_n_subscribers(pub_socket: zmq.Socket, n_subscribers: int):
    """
    blocks until pub_socket had n_subscribers connected to it
    """
    connections = 0
    events_socket = pub_socket.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)  # only accept this event
    while connections < n_subscribers:
        print("Waiting for a subscriber")
        recv_monitor_message(events_socket)  # this will block until a handshake was successful
        connections += 1

class Grabber(object):

    ## Parameters

    def __init__(self, pars=None, mode = "disk"):

        default = DEFAULT.make_default(depth=10)
        pars = default if pars is None else pars
        self.pars = pars
        # self.scans = deque(maxlen=2)
        self.shape = pars['shape']
        self.scan = Scan(self.shape)
        self.save_dir = '' 
        self.fname = ''
        self.fname_h5 = ''
        self.dnum_max = None
        self.enum_max = None
        self.dnum = 0
        self.enum = 0
        self._status = ''
        self._thread = None
        self.scan_is_dark = True
        self.scan_stopped = True
        self.scan_info = ''
        self.scan_path = ''
        self.QA = QtCore.QCoreApplication([])
        self.SI = ScanInfo()
        self.cin = CIN()

        self.frames_buffer = []
        self.index_list = []

        self.buffer_ready = False

        self.mode = mode #disk, inmem, socket
        self.current_dataset = "dark_frames" #this will be either exp_frames or dark_frames

        self.metadata = ""

        self.current_total = 0 
        self.dark_total = 0
        self.exp_total = 0

        self.dark_frames_offset = 0 #we use this to 0-index the exposure frames after mesure the dark ones

        self.send_socket = None
        self.send_addr = splitaddr("127.0.0.1:50000")

        try:
            self.statusterm = open(pars['statusterm'], 'a')
        except IOError as e:
            print(e)
            self.statusterm = None

        fsize = self.shape[0] * self.shape[1] * 2

        print(self.shape)

        shape= self.shape
        self.num_rows = shape[0] // 2
        self.num_adcs = shape[1] // 6

        self.ccd = FCCD(nrows=self.num_rows)

        if self.mode == "socket":
            self.createSendFrameSocket()
            wait_for_n_subscribers(self.send_socket, 1)

    def frame_to_image(self, frame):

        npbuf = np.frombuffer(frame, '<u2')
        npbuf = npbuf.reshape((12 * self.num_rows, self.num_adcs))
        image = self.ccd.assemble2(npbuf.astype(np.uint16))

        return image

    def prepare(self):
        ip, port = splitport(self.pars['tcp_addr'])
        self.createReceiveCommandsSocket_qtcpserver(ip=ip, port=int(port))

        # context = zmq.Context()
        p = self.pars['framegrabber']
        # self.frame_socket = context.socket(zmq.SUB)
        # self.frame_socket.setsockopt(zmq.SUBSCRIBE, '')
        # self.frame_socket.set_hwm(2000)
        # self.frame_socket.bind('tcp://%s' % p['read_addr'])

    def createSendFrameSocket(self):

        context = zmq.Context()
        self.send_socket = context.socket(zmq.PUB)
        self.send_socket.bind('tcp://%s:%d' % self.send_addr)
        self.send_socket.set_hwm(10000)
        print("Output frames will be sent to ip %s port %d" % self.send_addr)

    def send_frames(self):

        if self.mode == "disk": 
            write_data(self.fname_h5, self.current_dataset, np.array(self.frames_buffer), np.array(self.index_list), self.current_total)
        elif self.mode == "inmem":
            self.buffer_ready = True
            while self.buffer_ready:
                pass #we wait here for someone to consume this data and turn off buffer_ready
        elif self.mode == "socket":
            for i in range(len(self.frames_buffer)):
                print("Sending frame " + str(self.index_list[i]) + " to socket")

                self.send_socket.send_multipart([b'%d' % self.index_list[i], self.frames_buffer[i]])

        self.frames_buffer = []
        self.index_list = []

    def send_metadata(self, metadata):
        print("Sending metadata to socket")
        self.send_socket.send_string(json.dumps(metadata))

    def zmq_receive(self, pub_address=None, no_control=False):

        n_processed = 0

        buffer_max_size = min(100, self.dark_total)  #this limits how many frames are stored in memory before consuming them

        self.current_total = self.dark_total

        self.frames_buffer = []
        self.index_list = []

        self.print_status("Connecting zmq reading Thread", 'blue')
        print("Connecting zmq reading Thread")
        addr = 'tcp://%s' % self.pars['framegrabber']['pub_addr']
        timeout = int(self.pars['framegrabber']['timeout'])
        context = zmq.Context()
        frame_socket = context.socket(zmq.SUB)
        frame_socket.setsockopt(zmq.SUBSCRIBE, b'')
        frame_socket.set_hwm(2000)
        frame_socket.connect(addr)
        self.print_status("Running zmq reading Thread", 'blue')
        print("Running zmq reading Thread")

        # This could probablu have been done cleaner with a Poller
        # But I rather have the GIL released with sleep instead
        # of polling all the time.
        slp = timeout / 1000.

        first_frame = True
        first_id = 0

        while not self.scan_stopped:
            try:
                number, frame = frame_socket.recv_multipart(flags=zmq.NOBLOCK)  # blocking
                print("2: Received frame " + str(number))
                print(self.shape)
            except zmq.ZMQError:
                time.sleep(slp)
                continue

            if no_control and n_processed == self.dark_total:
                print("Reading exposure frames now")
                self.dark_frames_offset = self.dark_total
                self.current_dataset = "exp_frames" #this will be either exp_frames or dark_frames
                self.current_total = self.exp_total

            if first_frame:
                first_id = int(number)
                first_frame = False
                if self.mode == "socket":
                #We have to send first the raw frames shapes in order to deserialize later
                    self.send_metadata({"raw_frame_shape" : self.frame_to_image(frame).shape})
            
            self.frames_buffer.append(self.frame_to_image(frame))
            self.index_list.append(int(number) - self.dark_frames_offset - first_id)

            n_processed +=1

            if len(self.frames_buffer) == buffer_max_size:
                self.send_frames()

            #In principle this is only used with a simulation
            if n_processed == (self.dark_total + self.exp_total):
                print("All expected frames collected, stopping the scan now.")
                self.scan_stopped = True       

        #We need to save the rest of the frames on the buffer at the end
        if self.frames_buffer != []:
            self.send_frames()

        self.print_status("Stopped reading frames", 'blue')
        frame_socket.disconnect(addr)

        # self.frame_socket.close()

    def createReceiveCommandsSocket_qtcpserver(self, ip="127.0.0.1", port=8880):
        # A socket for receiving commands from the frontend using QTCP server
        self.qtcpserver = QtNetwork.QTcpServer()
        self.qtcpserver.newConnection.connect(self.acceptRemoteConnection)
        hostAddress = QtNetwork.QHostAddress(ip)
        self.qtcpserver.listen(hostAddress, port)
        self.status = "CINController is listening to commands on %d, using QTCPServer" % port

    def acceptRemoteConnection(self):
        self.qtcpclient = self.qtcpserver.nextPendingConnection()
        self.qtcpclient.readyRead.connect(self.answer_command_qtcp)
        self.status = "Accepted connection : %s:%d" % (
        self.qtcpclient.peerAddress().toString(), self.qtcpclient.peerPort())

    def answer_command_qtcp(self):
        newline = "\n\r"
        while self.qtcpclient.canReadLine():
            cmd = bytes(self.qtcpclient.readLine()).decode('UTF-8')
            # cmd = str(cmd) #str(QtCore.QString(cmd))
            # print(type(cmd))
            self.print_status('Getting: ' + cmd[:-1], 'green')
            if len(cmd) == 0:
                return self.qtcpclient.send(self.response(""))

            # sub command
            # cmd = str(QtCore.QString(cmd))
            resp_command = cmd
            #print("Got command: %s" %resp_command)

            if "stopCapture" in resp_command:

                self.on_finished_scan()

            elif "startCapture" in resp_command:
                """
                if self.scan_stopped:
                    self.scan_stopped = False
                    self.scan_is_dark = True
                else:
                    self.scan_is_dark = False
                """
                msg = 'dark' if self.scan_is_dark else 'exp'
                self.on_start_capture(msg)
                self.status = "Capturing '%s'." % msg

            elif "setCapturePath" in resp_command:

                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                newpath = cmd[index + 1:].strip()
                print("Got new path: %s" %newpath)
                if len(newpath) > 0 and newpath != self.scan_path:
                    self.status = "Save output to: " + newpath
                    self.scan_path = newpath
                    # self.newParam_str.emit('cxiwrite', 'saveDir', values.strip())
                    # self.startNewRun.emit()
                    self.on_new_path(newpath)

            elif "sendScanInfo" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                values = cmd[index + 1:].strip()

                self.scan_info = values.strip()
                self.print_status("New scan :\n %s" % '\n'.join(self.scan_info.split(",")))
                self.on_new_info(self.scan_info)

            elif "setExp2" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = float(cmd[index + 1:].strip()) + 3
                self.print_status(
                    "Setting alternate exposure to %.2f.\n This often corrsponds to a shutter time of %.2f\n" % (
                    exp, exp - 10.))
                self.cin.setAltExpTime(exp)

            elif "setExp" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = float(cmd[index + 1:].strip()) + 3
                self.print_status("Setting exposure to %.2f.\n" % exp)
                self.cin.setExpTime(exp)

            elif "setDoubleExpCount" in resp_command:
                index = cmd.index(" ")
                resp_command = cmd[0:index].strip()
                exp = int(cmd[index + 1:].strip())
                if exp == 0:
                    self.print_status("Setting to single exposure mode.")
                    self.cin.set_register("8050","0000",1)
                elif exp in range(1, 8):
                    self.print_status("Setting to double exposure mode (type %d)." % exp)
                    self.cin.set_register("8050","8%d00" % (exp-1),1)
                else:
                    self.print_status("Double exposure mode %d ignored." % exp)

            elif "resetCounter" in resp_command:
                self.cin.set_register("8001", "0106", 0)
                time.sleep(0.002)

            result = self.response(resp_command)
            self.qtcpclient.write(result)
            self.qtcpclient.flush()

    def on_new_info(self, info_raw):
        # This is needed for batched scans
        self.scan.exp.is_full = True

        self.status = "New info %s" % info_raw
        info = self.SI.read_tcp(info_raw)

        # this could maybe be part of read_tcp
        fac = info.get('repetition', 1)
        fac *= info.get('isDoubleExp', 0) + 1
        self.dark_total = info.get('dark_num_total', 0) * fac
        self.exp_total = info.get('exp_num_total', 0) * fac

        ia_addr = self.pars['iaserver']['ia_addr']
        if ia_addr is not None:
            scan = ClientScan('tcp://%s' % ia_addr, shape=self.shape, num_dark=self.dark_total, num_exp=self.exp_total)
            self.print_status("Flushing data to ZMQ Interaction Server %s" % ia_addr)
        else:
            self.print_status("Using disk to store frames")
            scan = Scan(shape=self.shape, num_dark=self.dark_total, num_exp=self.exp_total)

        # scan.info_raw = info_raw  # uncomment to avoid display
        scan.info = info
        self.scan = scan

        self.metadata = info

        gc.collect()

        if self.scan_stopped:
            self.scan_stopped = False
            self._thread = Thread(target=self.zmq_receive)
            self._thread.daemon = True
            self._thread.start()

    def on_new_path(self, nnpath):

        npath = str(nnpath)

        if self.fname_h5 == '':

            self.fname_h5 = os.path.split(npath)[0] + "/" + "raw_data.h5"

        scan = self.scan

        trunk, scan.stack_idx = os.path.split(npath)
        if str(scan.name) == '':
            scan.name = os.path.split(trunk)[1]
            self.print_status('Name: %s' % scan.name)
        self.save_dir = npath
        self.fname = npath + "test.hdf"

    def on_start_capture(self, msg):
        scan = self.scan
        if self.scan_is_dark:
            scan.fbuffer = scan.dark.frames
            scan.dark.dir = self.save_dir
            self.scan_is_dark = False

            dataset_name = "dark_frames"
            self.current_total = self.dark_total
            self.current_dataset = "dark_frames" 

        else:
            scan.dark.is_full = True
            scan.exp.dir = self.save_dir
            scan.fbuffer = scan.exp.frames

            dataset_name = "exp_frames"
            self.current_total = self.exp_total
            self.current_dataset = "exp_frames" 


            # this event signals the scan to be ready for being processed
            scan.save_meta()

            if self.mode == "socket":
                self.send_metadata(self.metadata)
            else:
                write_metadata(self.fname_h5, json.dumps(self.metadata))

            self.scan_is_dark = True

            scan.auto_save_data()  # moved up

    def response(self, cmd):
        return bytes('Command: ' + cmd + '\n\r', 'UTF-8')

    def print_status(self, status, color='yellow'):
        msg = term[color](status) + '\n'
        if self.statusterm is not None:
            self.statusterm.write(msg)
            self.statusterm.flush()
        else:
            print(msg)

    def print_cin_status(self, status):
        self.print_status(status, 'red')
        # self.statusterm.write(term['red'](status)+'\n')
        # self.statusterm.flush()

    def print_fg_status(self, status):
        self.print_status(status, 'green')
        # self.statusterm.write(term['green'](status)+'\n')
        # self.statusterm.flush()

    def on_finished_scan(self):
        self.scan_stopped = True
        self.scan.exp.full = True
        self.print_cin_status("Scan completed.")
        try:
            self.print_status("%d frames in buffer" % len(self.scan.fbuffer), 'blue')
        except:
            pass

    def get_scan_status(self):
        scan = self.scan
        dct = dict([(k, v) for k, v in self.scan.__dict__.items() if type(v) is str])
        # self.dnum = len(scan.dark_frames)
        # self.enum = scan.exp.num_processed
        # dmax = str(scan.info.get('dark_num_total',0)) if scan.info else 'unknown'
        # emax = str(scan.info.get('exp_num_total',0)) if scan.info else 'unknown'
        emax = str(scan.exp.num)
        dmax = str(scan.dark.num)
        dct['dark_num'] = str(scan.dark.num_processed) + '/' + dmax
        dct['exp_num'] = str(scan.exp.num_processed) + '/' + emax
        infostring = ["%20s : %s\n" % (k, dct[k]) for k in sorted(dct.keys())]
        return infostring

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, stat):
        self._status = stat
        self.print_cin_status(stat)


if __name__=='__main__':

    no_control = True

    if no_control:

        pars = {"shape":(1040,1152)}
        G = Grabber(mode = "socket")

        G.exp_total = 450
        G.dark_total = 50

        G.fname_h5 = "raw_data.h5"

        #This deletes and rewrites a previous file with the same name
        try:
            os.remove(G.fname_h5)
        except OSError:
            pass

        metadata = {
                    "energy": 800,
                    "exp_step_x": 0.03, 
                    "exp_step_y": 0.03,
                    "isDoubleExp": 1,
                    "double_exposure": True,
                    "exp_num_total": G.exp_total//2,
                    "dark_num_total": G.dark_total,
                    "exp_num_x": 15,
                    "exp_num_y": 15,
                    "dwell1": 100, 
                    "dwell2": 500
                    }

        metadata['translations'] = [(y * metadata["exp_step_y"], x * metadata["exp_step_x"]) for y in range(metadata["exp_num_y"]) for x in range(metadata["exp_num_x"])]        

        if G.mode == "socket":
            G.send_metadata(metadata)
        else:
            write_metadata(G.fname_h5, json.dumps(metadata))

        G.scan_stopped = False
        G._thread = Thread(target=G.zmq_receive(no_control=True))
        G._thread.daemon = False
        G._thread.start()

    else:
        G = Grabber()
        G.prepare()
        # G.IA.activate()
        # G.FG.start()
        while True:
            # process pending events
            G.QA.processEvents()
            time.sleep(0.1)
            #for line in G.get_scan_status():
            #    print(line)

            #if win is not None:
            #    win.clear()
            #    for line in G.get_scan_status():
            #        win.addstr(line)
            #    win.refresh()

            # handle client requests
            # G.IA.process_requests()


# parser = DEFAULT.add2argparser()
DEFAULT.parse_args()

