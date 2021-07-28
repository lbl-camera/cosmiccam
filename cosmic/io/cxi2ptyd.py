

import numpy as np

__all__ = ['cxi2ptyd'] 

def cxi2ptyd(cxi_in_path, ptyd_out_path = None):
    """
    Converts .cxi file at `cxi_in_path` to ptyd compatible file.
    """
    from cosmic.ext import io
    d = io.h5read(cxi_in_path)['entry_1']
    
    """
    if xytfile is None:
        positions = d['data_1']['translation'][:,::-1][:,1:]
    else:
        positions = load_pos_from_xyt(xytfile)
    """

    # Ptypys h5read ignores attributes, so we do not know if 'axes' of
    # the data are specified in this way. It works for ALS 5321 data
    # but this operation may differ for other data
    positions = d['data_1']['translation'][:,::-1][:,1:]
    data = d['data_1']['data']

    a = len(positions) 
    b = len(data)
    if a < b:
        print("Mismatch, %d positions vs %d data frames. Dropping data frames .." % (a,b))
        data=data[:a]
    elif a > b:
        print("Mismatch, %d positions vs %d data frames. Dropping positions .." % (a,b))
        positions=positions[:b]
    ptyd = {}
    #print len(positions), len(data)
    ptyd['chunks'] = {
        '0': {
        'data' : data,
        'positions' : positions,
        }
    }   
    ptyd['meta'] = dict(
        # Assuming  square data arrays
        shape = data.shape[-1],
        # Joule to kiloelectronvolt
        energy = 1./(1.6022 * 1e-16) * d['instrument_1']['source_1']['energy'] ,
        distance = float(d['instrument_1']['detector_1']['distance']),
        # assuming rectangular pixels,
        psize =float( d['instrument_1']['detector_1']['y_pixel_size']),
        propagation = 'farfield',
    )
    
    if ptyd_out_path is None:
        ptyd_out_path = cxi_in_path.replace('cxi','ptyd')
        
    io.h5write(ptyd_out_path,ptyd)
    
