import socket
import struct
import time

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

hexfile = '/global/groups/cosmic/code/test_data_sets/cin_burst.hex'
#hexfile = '/home/nsptycho/cin_burst.txt'
class TestFrameUDP:

    def __init__(self, udp_address=('127.0.0.1',49205), hex_path = hexfile):
        self.end_msg ='\xde\xad\xf0\x0d'
        self.udp_address =udp_address
        with open(hex_path,'r') as f:
            off= f.tell()
            for fn in range(4):
                self.headers = []
                self.packets = []
                finished = False
                i= 0
                while not finished:
                    
                    packet = f.read(8192)
                    #print len(packet)
                    header = packet[:8]
                    self.headers.append(header)
                    packet = packet[8:]
                    
                    if self.end_msg in packet:
                        finished = True
                        dead = packet.index(self.end_msg)
                        packet = packet[:dead]
                        f.seek(f.tell()-8192+dead+13)
                        
                    h2 = self.make_header(self.udp_address[1], len(packet)+8, 0, i%256)
                    print(i, len(packet), self.hx(header),self.hx(h2))
                    self.packets.append(packet)
                    i+=1
                    
            f.close()
        self.sock = None
        self.delay = 100.

    def hx(self,st):
        return ':'.join(["%02x" % ord(c) for c in st])
        
    def make_header(self, port, length, frame_number, packet_number):
        """Returns header."""
        #header = struct.pack('!BBHHH', packet_number, 0, port, length, frame_number)
        header = struct.pack('!BBHHH', packet_number, 240, 61938, 62452, frame_number)
        return header
        
    def listen_for_greeting(self):
        """Open a socket for sending UDP packets to the framegrabber."""
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.udp_address)
        self.status = "Socket open on (%s) on %d " % self.udp_address
    
        self.status =  "Awaiting Handshake"
        data, addr = self.sock.recvfrom(255)
        
        self.status =  "Received %s from (%s) on %d,  " % (str(data),addr[0],addr[1])
        self.fg_addr = addr

        self.sock.connect(addr)

    def udpsend(self,packet):
        time.sleep(self.delay/1.e7)
        self.sock.send(packet)
        
    def send_frame_in_udp_packets(self, frame_number):
        """Chop frame into small UDP packets and send them out through connected socket."""
        print('sending %d packets' % len(self.packets))
        ip, port = self.udp_address
        try:
            for i,packet in enumerate(self.packets):
                # either original or selmade header
                h = self.make_header(port, len(packet)+8, frame_number, i%256)
                #h = self.headers[i]
                print(i, self.hx(h))
                bytes_sent = self.udpsend(h + packet) #, (self.ip, self.port))

            bytes_sent = self.udpsend(self.end_msg)
            

            packet = self.make_header(port, 48, frame_number, 0) + 32*"\x00"+self.end_msg
            bytes_sent = self.udpsend(packet)  #, (self.ip, self.port))
            bytes_sent = self.udpsend(packet)  #, (self.ip, self.port))
        except socket.error:
            self.status = "Connection error. Restarting ..."
            self.sock.close()
            self.listen_for_greeting()
            
    def status_emit(self,status,color=None):
        print(status)
        
    @property
    def status(self):
        return self._status
        
    @status.setter
    def status(self,put):
        self._status= put
        self.status_emit(put)


if __name__=='__main__':
    
    TF = TestFrameUDP()
    TF.listen_for_greeting()
    fn = 0
    while True:
        TF.send_frame_in_udp_packets(fn)
        fn+=1
        time.sleep(0.5)
