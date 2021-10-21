#!/bin/bash
python diffviewer_zmq.py 0 "127.0.0.1:49206" &
python diffviewer_zmq.py 1 "127.0.0.1:49206" &
python diffviewer_zmq.py 2 "127.0.0.1:50007" &
python diffviewer_zmq.py 3 "127.0.0.1:50008" &
python diffviewer_zmq.py 4 "127.0.0.1:50011" &
python diffviewer_zmq.py 5 "127.0.0.1:50011" &


