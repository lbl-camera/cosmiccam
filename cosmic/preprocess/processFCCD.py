import numpy as np
# from util import shift
from scipy.interpolate import interp1d
import scipy.ndimage as ndimage

import sys

"""
This should be part of a class
"""
from cosmic.ext.ptypy.utils.validator import ArgParseParameter
from io import StringIO

DEFAULT = ArgParseParameter(name='FCCD_Processor')

DEFAULT.load_conf_parser(StringIO(
    """
    [num_data_cols] 
    default = 10
    help = Number of data columns per super column
    
    [num_over_cols] 
    default = 2
    help = Number of overscan columns per super column
    
    [num_scols]
    default = 192
    help = Number of supercolumns
    
    [num_rows]
    default = 480
    help = Number of pixel rows per segment
    
    [scol_precount]
    default = 1
    help = Pixels prior to data or supercol readout
    
    [threshold_high]
    default = 50000
    help = Saturation threshold in ADU
    
    [threshold_low]
    default = 1000
    help = Empty pixel readout
    """
))


class ProcessFCCDraw(object):
    # All paramaters at class level. 
    # There will be only one instance so that is ok.

    # Parameters
    # rows = 480
    # n = 2*rows ##here the vertical size is equal to the final size because overscan modulates horizontal
    ndd = 10  ##number of data columns for super column
    nos = 2  ##number of overscan columns per super column
    # ns = 192 ##number of super columns

    kwargs_spike_removal = dict(
        thr=400,  ##intensity threshold for the FFT filter
        thr_frequency=500,  ##range of low frequency pixels to ignore
        thr_background=4.0,
        original=True
    )

    kwargs_overscan = dict(
        flat_pixels=100,  ## ignored pixel
    )
    DEFAULT = DEFAULT

    def __init__(self, rows=480, scols=192, precount=0, verbose=True):  # pars, **kwargs):
        # self.p = self.DEFAULT.make
        # self.p.update(pars)
        # self.p.upadte(kwargs)
        self.rows = rows
        self.ns = scols

        w = self.ndd + self.nos
        i = precount
        dummy = np.zeros((2 * self.rows, self.ns * w // 2))
        osmask = self.stretch(dummy)
        osmask[:, (0 + i) % w::w] = 1
        osmask[:, (w - 1 + i) % w::w] = 1
        dmask = 1 - osmask
        self.indexd = np.where(dmask)
        self.indexo = np.where(osmask)
        self.get_data = lambda x: np.reshape(x[self.indexd], (self.rows, self.ndd * self.ns))
        self.get_overscan = lambda x: np.reshape(x[self.indexo], (self.rows, self.nos * self.ns))
        self.fitmatrix = None

    def process_stack(self, data, verbose=True, **kwargs):

        res = []
        for ii, d in enumerate(data):

            res.append(self.process_one_frame(d, **kwargs))

            if verbose:
                sys.stdout.write('Processed %d percent \r' % (100 * ii / len(data)))
                sys.stdout.flush()

        return np.asarray(res)

    def process_one_frame(self, d, do_overscan=False, do_spike_removal=False):

        din = self.stretch(d)
        dd = self.get_data(din)

        if do_overscan:
            os = self.get_overscan(din)
            ##in this context, OSMASK masks pixels in overscan which are 0, from dropped packets
            osmask = os > 0.  # ignores the dropped packets, I hope
            os = os * osmask

            kwargs = self.kwargs_overscan.copy()

            # bkg_os, self.fitmatrix = fitOverscan(os, fitmatrix = self.fitmatrix, **kwargs)

            # dd -= bkg_os

            fitOverscanNew(dd, os)

        if do_spike_removal:
            kwargs = self.kwargs_spike_removal.copy()

            bkg_ripple = spike_detection(dd, **kwargs)

            dd -= bkg_ripple

        return self.stack(dd)

    @staticmethod
    def stretch(A):
        flip = lambda x: np.rot90(x, 2)
        ny = A.shape[0]
        return np.hstack((A[ny // 2:ny, :], flip(A[0:ny // 2, :])))

    @staticmethod
    def stack(A):
        flip = lambda x: np.rot90(x, 2)
        n = A.shape[1] // 2
        return np.vstack((flip(A[:, n:]), A[:, 0:n]))


def fitOverscanNew(dd, os, flat_pixel=100, ns=192, ndd=10, nos=2):
    for i in range(ns):
        j = i * ndd
        k = i * nos
        osd = os[:, k: k + 2].min(axis=1)
        temp = np.transpose(dd[:, j:j + 10])
        # goodIDX = np.where(osd > 0)[0]
        # allIDX = np.linspace(0, len(temp2) - 1, len(temp2)).astype('int')
        # f = interpolate.interp1d(goodIDX,osd[goodIDX], bounds_error = False, fill_value = osd.mean())
        temp2 = ndimage.filters.gaussian_filter(osd, sigma=10)
        rms = temp2[flat_pixel:flat_pixel + 50].std()
        mn = temp2[flat_pixel:flat_pixel + 50].mean()
        flat = temp2[0:flat_pixel]
        flat[abs(flat - mn) > 5. * rms] = mn
        temp2[0:flat_pixel] = flat
        temp -= temp2
        dd[:, j:j + 10] = np.transpose(temp)

    return dd


def fitOverscan(a, fitmatrix=None, flat_pixels=100):
    ll = a.shape[0]
    # avoids double computation
    if fitmatrix is None:
        xxb = np.matrix(np.linspace(0, 1, ll)).T
        w = np.ones((ll))
        w[0:flat_pixels] = 0
        w = np.matrix(np.diag(w))
        Q = np.matrix(
            np.hstack((np.power(xxb, 8), np.power(xxb, 6), np.power(xxb, 4), np.power(xxb, 2), (xxb * 0. + 1.))))
        fitmatrix = Q * (Q.T * w * Q).I * (w * Q).T

    bnfw = np.rot90(fitmatrix * np.matrix(a), 2)

    jkjk = np.array(bnfw)  # / np.rot90(mask,2)
    jkjk1 = np.zeros((2 * ll, 192))
    jkjk1[::2, 0:192] = jkjk[:, ::2]
    jkjk1[1::2, 0:192] = jkjk[:, 1::2]

    tt = np.reshape(np.linspace(1, 12 * ll, 12 * ll), (ll, 12))
    ttd = tt[:, 9::-1]
    tts = tt[:, 12:9:-1]

    f = interp1d(tts.flatten(), jkjk1, 'linear', axis=0, bounds_error=False, fill_value=jkjk1.mean())
    bkg = np.rot90(np.reshape(f(ttd), (ll, 192 * 10)), 2)
    bkg = f(ttd)
    bkg1 = np.zeros((ll, 1920))

    for i in range(10):
        bkg1[:, i::10] = bkg[:, i, :]

    return np.rot90(bkg1, 2), fitmatrix


def spike_detection(A, thr=400, thr_frequency=500, thr_background=4.0, original=True):
    """
    Suppresses high-frequency ripple noise across adc readout blocks (supecolumns). 
    """
    bthr = thr_background
    rows = A.shape[0]
    cols = A.shape[1]
    ndd = 10
    ns = cols // ndd

    if original:
        # original
        jk = np.reshape(np.transpose(np.reshape(A, (rows, ndd, ns)), (0, 1, 2)), (rows * ndd, ns))
        jkf = (np.fft.fftn((jk * (np.abs(jk) < bthr))))  # ; % only when it is background...
        msk = abs(jkf) > thr;
        msk[0:thr_frequency, :] = 0  # ; % Fourier mask, keep low frequencies.
        bkg = np.reshape(np.transpose(np.reshape(np.fft.ifftn(jkf * msk), (ndd, rows, ns)), (0, 1, 2)),
                         (rows, ndd * ns))

    else:
        # simplified
        jk = np.reshape(A, (rows * ndd, ns))
        bkg = jk * (np.abs(jk) < bthr)

        # readouts independent (transform only along supercolumn)
        bkg_fourier = np.fft.fft2(bkg, axis=0)

        # threshold dependent on colum width
        msk = abs(jkf) > thr;

        # keep low frequencies (why?)
        msk[0:thr_frequency, :] = 0

        bkg = np.reshape(np.fft.ifft2(bkg_fourier * mask, axis=0), (rows, ndd * ns))

    return bkg.real
