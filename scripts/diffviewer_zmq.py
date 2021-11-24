# -*- coding: utf-8 -*-
"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import zmq
import numpy as np
import time
import json
import msgpack
import msgpack_numpy
from PIL import Image

# magma colormap
from pyqtgraph import ColorMap

magma_cmap = ColorMap(np.linspace(0, 1, 11), np.array([(0.001462, 0.000466, 0.013866, 1.0),
                                                       (0.078814999999999996, 0.054184000000000003, 0.21166699999999999,
                                                        1.0),
                                                       (0.23207700000000001, 0.059888999999999998, 0.437695, 1.0),
                                                       (0.39038400000000001, 0.100379, 0.50186399999999998, 1.0),
                                                       (0.55028699999999997, 0.161158, 0.50571900000000003, 1.0),
                                                       (0.716387, 0.21498200000000001, 0.47528999999999999, 1.0),
                                                       (0.86879300000000004, 0.28772799999999998, 0.40930299999999997,
                                                        1.0),
                                                       (0.96767099999999995, 0.43970300000000001, 0.35981000000000002,
                                                        1.0),
                                                       (0.99473800000000001, 0.62434999999999996, 0.42739700000000003,
                                                        1.0),
                                                       (0.99568000000000001, 0.81270600000000004, 0.57264499999999996,
                                                        1.0),
                                                       (0.98705299999999996, 0.99143800000000004, 0.74950399999999995,
                                                        1.0)]))

from cosmic.camera.fccd import FCCD


class Framegrabber_viewer(QtCore.QThread):
    framedata = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, rows=520, roi=480, cols=1152, address="127.0.0.1:49206", mode = None, save = 0):
        self.address = address
        QtCore.QThread.__init__(self)
        self.number = ''
        self.frame = None
        self.CCD = FCCD(nrows=roi + 10)
        self.rows = rows
        self.roi = roi
        self.cols = cols
        self.save = save

    def __del__(self):
        self.wait()

    def run(self):
        addr = 'tcp://%s' % self.address
        context = zmq.Context()
        frame_socket = context.socket(zmq.SUB)
        frame_socket.setsockopt(zmq.SUBSCRIBE, b'')
        frame_socket.set_hwm(2000)
        frame_socket.connect(addr)
        row_bytes = self.cols * 4

        first_msg = True

        all_data = []
        index = 0

        while True:
            self.number, buf = frame_socket.recv_multipart()  # blocking

            # npbuf = np.frombuffer(buf[2304 * (975-self.roi -10) * 2: 2304 * 975 *2],'<u2')
            # pedestal = np.frombuffer(buf[2304 * 100 * 2: 2304 * 300 *2],'<u2')
            npbuf = np.frombuffer(buf[row_bytes * 5: row_bytes * (self.roi + 15)], '<u2')
            pedestal = np.frombuffer(buf[row_bytes * 5: row_bytes * 55], '<u2')
                
            self.frame = npbuf.reshape((npbuf.size // self.CCD._nbmux, self.CCD._nbmux)).astype(np.float)

            #Here we generate a descrambled preview of the frames
            if mode == 0:
                pedestal = pedestal.reshape((pedestal.size // self.CCD._nbmux, self.CCD._nbmux)).astype(np.float)
                bg = pedestal.mean(0).reshape((1, self.CCD._nbmux))

                assembled = self.CCD.assemble_nomask(self.frame)
                self.frame = assembled  # [(self.rows-self.roi):(self.rows+self.roi),:]

            if first_msg:
                #self.view.setRange(QtCore.QRectF(0, 0, *(self.frame.shape)))
                first_msg = False

            self.framedata.emit(self.frame)
            if self.save != 0:
                print("saving image")
                a = np.log(self.frame + 0.1)
                b = (a  / np.max(a) ) * 255
                im1 = Image.fromarray(np.array(b).astype(np.float32))
                #im1 = Image.fromarray(np.array(self.frame.T).astype(np.float32))
                im1.convert("L").save("images/mode_{}_{}.tiff".format(mode, index))
            index += 1
        frame_socket.disconnect(addr)


def receive_metadata(network_metadata):

    print("Waiting for metadata...")
    metadata = json.loads(network_metadata["input_socket"].recv_string())  # blocking
    print("Received metadata ")
    print(metadata)

    return metadata


def subscribe_to_socket(network_metadata):

    addr = 'tcp://%s' % network_metadata["input_address"]

    socket = network_metadata["context"].socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.set_hwm(2000)
    socket.connect(addr)

    return socket

class Viewer(QtCore.QThread):
    framedata = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, view, address="127.0.0.1:49206", mode = None, save = 0):
        self.address = address
        self.view = view
        self.save = save
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):

        network_metadata = {}
        network_metadata["context"] = zmq.Context()
        network_metadata["input_address"] = self.address
        network_metadata["input_socket"] = subscribe_to_socket(network_metadata)

        first_msg = True

        all_data = []
        index = 0

        while True:
            msg = network_metadata["input_socket"].recv()

            #This bypasses the metadata
            try:
                a = msgpack.unpackb(msg, object_hook= msgpack_numpy.decode, use_list=False,  max_bin_len=50000000, raw=False)
                if mode == 4 or mode == 5 or mode == 6:
                    (number, frame, probe) = a
                else:
                    (number, frame) = a
            except:
                continue
            #print(frame.shape)
            #print(type(frame))
            output = frame

            if mode == 4:
                #Here number is the frame width, and we crop using that
                output = np.abs(frame[number//2:-number//2,number//2:-number//2])
            if mode == 5:
                output = np.angle(frame[number//2:-number//2,number//2:-number//2])
            if mode == 6:
                output = np.abs(probe)

            output = np.rot90(output)

            if first_msg:
                self.view.setRange(QtCore.QRectF(0, 0, *(output.shape)))
                first_msg = False
            self.framedata.emit(output)
            if self.save != 0:
                print("saving image")
                a = np.log(output + 0.1)
                b = (a  / np.max(a) ) * 255
                im1 = Image.fromarray(np.array(b).astype(np.float32))
                im1.convert("L").save("images/mode_{}_{}.tiff".format(mode, index))
            index += 1
        frame_socket.disconnect(addr)


@QtCore.pyqtSlot(np.ndarray)
def updateData(data):
    global img
    img.setImage(data)

mode_title = list(range(0,7))
mode_title[0] = "Descrambled preview frames from framegrabber"
mode_title[1] = "Scrambled frames"
mode_title[2] = "Descrambled frames"
mode_title[3] = "Filtered frames"
mode_title[4] = "Reconstructed image (abs)"
mode_title[5] = "Reconstructed image (angle)"
mode_title[6] = "Reconstructed probe"

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    #mode 1 "scrambled"
    #mode 2 "descrambled"
    #mode 3 "filtered"
    #mode 4 "reconstructed"

    mode = 1

    if len(sys.argv) > 1:
        mode = int(sys.argv[1])
        address = sys.argv[2]
        save = int(sys.argv[3])
    else:
        address = "127.0.0.1:49206"
        save = 0

    print("Mode {}".format(mode))
    print(address)
    print(save)

    #-------------------
    app = QtGui.QApplication([])

    ## Create window with GraphicsView widget
    win = pg.GraphicsLayoutWidget()
    win.show()  ## show widget alone in its own window
    win.setWindowTitle(mode_title[mode])
    view = win.addViewBox()

    ## lock the aspect ratio so pixels are always square
    view.setAspectLocked(True)

    ## Create image item
    img = pg.ImageItem(border='w')
    view.addItem(img)

    ## Set initial view bounds
    view.setRange(QtCore.QRectF(0, 0, 1152, 1152))

    # Contrast/color control
    hist = pg.HistogramLUTItem()
    hist.setImageItem(img)
    #hist.gradient.setColorMap(magma_cmap)
    win.addItem(hist)
    #-------------------

    # P = ZMQPubThread()

    S = None

    if mode == 0 or mode == 1:
        S = Framegrabber_viewer(address=address, mode = mode, save = save)
    else:
        S = Viewer(view, address=address, mode = mode, save = save)

    S.framedata.connect(updateData)
    # P.start()
    S.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


