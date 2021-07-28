

import cosmic.ext as ce
import sys

parallel = ce.utils.parallel

from cosmic.preprocess import process_5321_class as pro

DEFAULT = pro.DEFAULT


DEFAULT.parse_args()

import json
f = open('preprocess.json','w')
json.dump(DEFAULT.make_default(depth=10),f, indent =2 )
f.close()

