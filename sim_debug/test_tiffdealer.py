import numpy as np
from cosmic.preprocess import processFCCD
from cosmic.preprocess import process_5321_class as pro
import time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import cosmic.ext as ce
rows = 480
aalrows = 2*rows ##here the vertical size is equal to the final size because overscan modulates horizontal
ndd = 10 ##number of data columns for super column
nos = 2 ##number of overscan columns per super column
ns = 192 #
sh_overscan = (rows,ns*nos)
sh_frame = (rows,ns*(nos+ndd))
sh_data = (rows,ns*ndd)

osmask = np.zeros(sh_frame)
osmask[:,0::12] = 1
osmask[:,11::12] = 1
dmask = 1 - osmask
indexd = np.where(dmask)
indexo = np.where(osmask)

base = '/global/groups/cosmic/code/test_data_sets/180228048/'
bg_dir = base + '001/'
exp_dir = base + '002/'

log = pro.Log()

Dealer = pro.TiffDealer(
    exp_dir = exp_dir,
    bg_dir = bg_dir,
    dwell = None, #[1.0,8.0], 
    repetitions = 8,
    dtype= np.float,
    dark_total=None,
    exp_total=None,
    threshold = 2000, 
    rows = 480,
    scols = 192,
    precount = 0,
    do_overscan = True,
    do_spike_removal = False
    )
    
log.emit = "Averaging background frames..."
    
ndarks = Dealer.set_dark()

log.emit = "Done. Averaged %d frames." % ndarks

start = 0

frames_available  = Dealer.frames_available(start)

log.emit = "%d frames available." %  frames_available

node_indices = list(range(start,start+min(frames_available,1)))

data = Dealer.get_clean_data(node_indices)[0]

plt.figure()
plt.imshow(np.log10(data[200:-200,200:-200]+1))
plt.imshow(data* (data < 10) * (data > -10))
plt.colorbar()

plt.figure()
plt.imshow(data* (data > 2.8) * (data < 10))
plt.colorbar()

binned = ce.utils.rebin_2d(data,3)[0]
mn1, std1 =  data[150:300,150:300].mean(),data[150:300,150:300].std() 
mn2, std2 =  binned[50:100,50:100].mean(),binned[50:100,50:100].std() 
print(mn1, std1, mn2, std2)
#darks = Dealer.darks
d2 = (data - mn1) / 3.6
d2 = np.round(d2) 
d2 *= d2 > 0.
d2 = ce.utils.rebin_2d(d2,3)[0]
b2 = (binned - mn2) / 3.6
b2 *= b2 > 0.
plt.figure()
plt.imshow(b2 * (b2 < 6.) )
plt.colorbar()
plt.figure()
plt.imshow(d2 * (d2 < 6.) )
plt.colorbar()
plt.figure()
plt.imshow(d2 - b2 )
plt.colorbar()

plt.show()
