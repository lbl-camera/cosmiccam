
import cosmic.ext as ce
import sys

parallel = ce.utils.parallel

from cosmic.preprocess.defaults import DEFAULT



if parallel.master:
    print "Welcome to readALS MPI!"
    if len(sys.argv) == 1:
        #print "Parameter file argument needed."
        #sys.exit()
        paramFile = '/global/groups/cosmic/code/test_data_sets/161102035_small/001/param.txt'
    elif len(sys.argv) > 1: 
        paramFile = sys.argv[1]
    try: 
        f = open(paramFile,'r')
    except IOError:
        print "IOError: could not open parameter file: %s" %(paramFile)
        sys.exit()
    else:
        param = DEFAULT.copy()
        for item in f:
            pStringList = [item.split('=')[0],item.split('=')[1].rstrip('\n')]
            if len(pStringList) != 2:
                pass
            else:
                param[pStringList[0]] = pStringList[1]

    param = parallel.bcast(param)
else:
    param = parallel.bcast(None)
    
from cosmic.preprocess import process_5321_new as pro

parallel = ce.utils.parallel

pro.process(param)
