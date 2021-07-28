

    ######################################################################################################
    ###Load the data from the directory of tiff frames
    if param['speFile'] == '':
	pass
	###the frames are now loaded by the MPI processors so this line is not needed
        #dataStack, loadedFrames = loadDirFrames(scanDir, filePrefix, expectedFrames, cropSize, multi, fCCD, frame_offset)
    else:
        dataStack = np.asarray(readSpe(scanDir + param['speFile'])['data'])
        loadedFrames = len(dataStack)
        
        
    else: fsh = dataStack[0].shape; fNorm, indexd, indexo, os_bkg = None, None, None, None
    dy, dx = float(sh_pad[0]) / float(sh_sample[0]),float(sh_pad[1]) / float(sh_sample[1])
    dyn, dxn = int(float(fsh[0]) / dy), int(float(fsh[1]) / dx)
