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


class ZMQSubThread(QtCore.QThread):
    framedata = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, rows=520, roi=480, cols=1152, pub_address="127.0.0.1:49206"):
        self.pub_address = pub_address
        QtCore.QThread.__init__(self)
        self.number = ''
        self.frame = None
        self.CCD = FCCD(nrows=roi + 10)
        self.rows = rows
        self.roi = roi
        self.cols = cols

    def __del__(self):
        self.wait()

    def run(self):
        addr = 'tcp://%s' % self.pub_address
        context = zmq.Context()
        frame_socket = context.socket(zmq.SUB)
        frame_socket.setsockopt(zmq.SUBSCRIBE, b'')
        frame_socket.set_hwm(2000)
        frame_socket.connect(addr)
        row_bytes = self.cols * 4
        while True:
            self.number, buf = frame_socket.recv_multipart()  # blocking
            # npbuf = np.frombuffer(buf[2304 * (975-self.roi -10) * 2: 2304 * 975 *2],'<u2')
            # pedestal = np.frombuffer(buf[2304 * 100 * 2: 2304 * 300 *2],'<u2')
            # print npbuf.size / 2304

            npbuf = np.frombuffer(buf[row_bytes * 5: row_bytes * (self.roi + 15)], '<u2')
            pedestal = np.frombuffer(buf[row_bytes * 5: row_bytes * 55], '<u2')
            # print npbuf.size / 2304
            npbuf = npbuf.reshape((npbuf.size // self.CCD._nbmux, self.CCD._nbmux)).astype(np.float)
            pedestal = pedestal.reshape((pedestal.size // self.CCD._nbmux, self.CCD._nbmux)).astype(np.float)
            bg = pedestal.mean(0).reshape((1, self.CCD._nbmux))
            #assembled = self.CCD.assemble_nomask(npbuf - bg)
            assembled = self.CCD.assemble_nomask(npbuf)
            # print assembled.shape
            self.frame = assembled  # [(self.rows-self.roi):(self.rows+self.roi),:]
            # scols = cols / 12
            # npbuf.reshape((2000,cols))
            # frame = np.reshape(np.transpose(np.reshape(npbuf,(12, rows, scols), order='F'), [1,0,2]), (rows, cols), order='F')
            self.framedata.emit(self.frame.T)
        frame_socket.disconnect(addr)


class ZMQPubThread(QtCore.QThread):

    def __init__(self, pub_address="127.0.0.1:49206"):
        self.pub_address = pub_address
        QtCore.QThread.__init__(self)
        self.number = 0
        self.frame = None

    def __del__(self):
        self.wait()

    def run(self):
        addr = 'tcp://%s' % self.pub_address
        context = zmq.Context()
        frame_socket = context.socket(zmq.PUB)
        frame_socket.set_hwm(2000)
        frame_socket.bind(addr)
        while True:
            self.number += 1
            frame = np.random.randint(10000, 30000, (1040, 1152), 'uint16')
            frame_socket.send_multipart([str(self.number), frame])  # blocking
            time.sleep(0.05)


app = QtGui.QApplication([])

## Create window with GraphicsView widget
win = pg.GraphicsLayoutWidget()
win.show()  ## show widget alone in its own window
win.setWindowTitle('pyqtgraph example: ImageItem')
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
hist.gradient.setColorMap(magma_cmap)
win.addItem(hist)


@QtCore.pyqtSlot(np.ndarray)
def updateData(data):
    global img
    img.setImage(data)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys

    # P = ZMQPubThread()
    S = ZMQSubThread()
    S.framedata.connect(updateData)
    # P.start()
    S.start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
