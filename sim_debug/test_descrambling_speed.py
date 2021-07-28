
from cosmic.ext.profilehooks import profile
from cosmic.camera import fccd
import time
import numpy as np

c=fccd.FCCD()
sh = (1920,960)
A,frame = np.indices(sh)

print('{' + ', '.join([str(q) for q in c._q]) + '}')
print((c._nbcol * c._nrows, c._nb*c._nmux))
def _convert(frame):
    frame[frame>63000]=63000
    return frame.astype(np.uint16).byteswap().tostring()
        
clk = c._clockXrow(c._rowXccd(A+frame))
print(np.allclose(clk,c._rawXclock(c._clockXraw(clk))))

@profile
def scramble(frame):
    return _convert(c._rawXclock(c._clockXrow(c._rowXccd(frame))))
    
@profile
def descramble(nframe):
    dframe = np.frombuffer(nframe, '<u2')
    return c.assemble2(c.descramble(dframe))
    
    
t = time.time()
N =20
for i in range(N): 
    # create a frame in byte stream. Cut off a bit at the end and attach ending message
    nframe = scramble(frame)

sf = N / (time.time()-t)
print("Scramble frequncy is %.2f" % sf) 



t = time.time()
for i in range(N):
    dframe = descramble(nframe)
    
sf = N / (time.time()-t)
print("Descramble frequncy is %.2f" % sf) 
