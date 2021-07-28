import sys
import zmq
from cosmic.camera import fccd

port = "5556"
if len(sys.argv) > 1:
    port =  sys.argv[1]
    int(port)
    
if len(sys.argv) > 2:
    port1 =  sys.argv[2]
    int(port1)

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

socket.connect("tcp://10.0.0.16:49206")
    
# Subscribe to zipcode, default is NYC, 10001
topicfilter = "rawframe"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

# Process 5 updates
total_value = 0
for update_nbr in range (1):
    topic, fnum, data = socket.recv_multipart()
    print(topic, int(fnum), len(data), type(data))

      
from matplotlib import pyplot as plt
import numpy as np
#plt.ion()
fig=plt.figure()
ax = fig.add_subplot(111)
nrows = 960
frame = np.frombuffer(data, '<u2')
nrows = frame.size/192/12
descrambled=frame.reshape((nrows*12,192))
A=descrambled.T.reshape(192,nrows,12).swapaxes(0,1).reshape(nrows,12*192)
ax.imshow(np.vstack([A[:,6*192:],A[:,:6*192][::-1,::-1]]))
plt.show()
