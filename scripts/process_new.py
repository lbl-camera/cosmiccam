import os
import cosmic.ext as ce
import sys
import time

parallel = ce.utils.parallel

from cosmic.preprocess import process_5321_class as pro

DEFAULT = pro.DEFAULT

parallel = ce.utils.parallel

args = sys.argv[1:]
if len(args) > 0:
    scan_def = args[0]
else:
    scan_def = None

if len(args) > 1:
    param_json = args[1]
else:
    param_json = '/cosmic-dtn/groups/cosmic/Data/preprocess_cfg/default.json'
    # param_json  = '/cosmic-dtn/groups/cosmic/Data/preprocess_cfg/default_streaming2.json'

import json

f = open(param_json, 'r')
pars = json.load(f)
f.close()

P = pro.Processor(pars)
P.PROCESSED_FRAMES_PER_NODE = 10

if scan_def is not None:
    while not os.path.exists(scan_def):
        if parallel.master:
            print('Waiting for parameter file %s' % scan_def)
        time.sleep(.5)
parallel.barrier()

P.load_scan_definition(scan_def)

if parallel.master:
    print(ce.utils.verbose.report(P.param))

P.calculate_geometry()
P.calculate_translation()
# P.prepare()
# P.prepare(dark_dir_ram='/dev/shm/dark',exp_dir_ram='/dev/shm/exp')
P.prepare()
P.process(0)
P.write_cxi()
