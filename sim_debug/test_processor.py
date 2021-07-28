import os
import cosmic.ext as ce
import sys
import datetime

prefix = '/cosmic-dtn/groups/cosmic/Data/'
parallel = ce.utils.parallel
from cosmic.preprocess import process_5321_class as pro

if len(sys.argv) > 1:
    identifier = sys.argv[1]
    num = identifier[-3:]
    date = datetime.datetime.strptime(identifier[:6],'%y%m%d')
    path1 = prefix + date.strftime('%y%m%d/%y%m%d') + num + '/scan_info.txt'
    path2 = prefix + date.strftime('%Y/%m/%y%m%d/%y%m%d') + num + '/scan_info.txt'    
    if os.path.exists(path1):
        scan_info = path1
    elif os.path.exists(path2):
        scan_info = path2

if len(sys.argv) > 2:
    pro.DEFAULT = ArgParseParameter(name='Preprocessor')
    if os.path.exists(sys.argv[2]):
        with open(sys.argv[2]) as f:
            pro.DEFAULT.load_conf_parser(f)
            f.close()
            
P = pro.Processor()
P.load_scan_definition(scan_info)
print(ce.utils.verbose.report(P.param))
P.calculate_geometry()
P.calculate_translation()
P.prepare()
P.process()
P.write_cxi('test.cxi')
