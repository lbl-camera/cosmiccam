#!/usr/bin/env python

import sys, getopt, os

default_output_address = "127.0.0.1:50007"
default_control_address = "131.243.73.179:8880" 

help =   "\nUsage: cosmiccam.py [options] 'input_address:port'\n\n\
\t -m M -> Output mode. Supports M = 'disk','socket' and 'disksocket'. 'disk' (default) saves the final results into disk, \n\
\t\t\t'socket' streams the data into a pub zmq socket, and 'disksocket' does the same but also stores the final results into disk at the end.\n\
\t -o ADDRESS -> Set ADDRESS as 'IP:PORT' corresponding to the address to publish all output data.\n\
\t\t\tDefaults to {}\n\
\t -c ADDRESS -> Set ADDRESS as 'IP:PORT' corresponding to the address of the control software.\n\
\t\t\tDefaults to {}\n\
\t -n -> Runs the code bypassing communication with the control software using predefined metadata.\n\
\n\n".format(default_output_address, default_control_address)

def parse_arguments(args, options = None):

    print("\nParsing parameters...\n")

    if options is None:
        options = {
                   "output_mode":"disk",
                   "output_address": default_output_address,
                   "control_address": default_control_address,
                   "no_control": False}

    try:
        opts, args_left = getopt.getopt(args,"m:o:c:n", \
                              ["output_mode=", "output_address=", "control_address=", "no_control"])

    except getopt.GetoptError:
        print(help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(help)
            sys.exit()
        if opt in ("-m", "--output_mode"):
            options["output_mode"] = str(arg)
        if opt in ("-o", "--output_address"):
            options["output_address"] = str(arg)
        if opt in ("-c", "--control_address"):
            options["control_address"] = str(arg)
        if opt in ("-n", "--no_control"):
            options["no_control"] = True


    if len(args_left) != 1:

        print(help)
        sys.exit(2)

    else:
        options["input_address"] = args_left[0]
    
    return options


    
