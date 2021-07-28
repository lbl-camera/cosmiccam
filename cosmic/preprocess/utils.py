
def add_stxm_to_cxi(cxi_file, new_file=None):
    
    
    ihistMask = dataStack.sum(axis = 0)
    ihistMask = ihistMask > 0.1 * ihistMask.max()
    ihist = np.array([(dataStack[i] * ihistMask).sum() for i in range(nPoints)])
    indx = np.arange(nPoints)
    stxmImage = np.reshape(ihist,(ypts,xpts))[::-1,:]
    stxmImageInterp = interpolate_STXM(stxmImage, ssy, ssx)
    pass
    
def remove_outliers_from_cxi(cxi_file, new_file=None):
    
    
    """
    if removeOutliers:
        #gy, gx = gradient(stxmImage)
        gy = stxmImage - ndimage.filters.gaussian_filter(stxmImage, sigma = 0.25)
        gy = gy[::-1, :].flatten()  ##puts it in the same ordering as ccddata, starting lower left
        delta = 8. * gy.std()
        badIndices = np.where(gy < (gy.mean() - delta))[0]  ##the min Y gradient is one row below the bad pixel
        ihistMask = 1 - ihistMask
        ihist = np.array([(dataStack[i] * ihistMask).sum() for i in range(nPoints)])
        noiseIndices = np.where(ihist > (ihist.mean() + 2. * ihist.std()))
        badIndices = np.unique(np.append(badIndices,noiseIndices))
        stxmImage = stxmImage[::-1, :].flatten()
        k = 0
        if len(badIndices) > 0:
            for item in badIndices:
                stxmImage[item] = (stxmImage[item + 1] + stxmImage[item - 1]) / 2.
                if indx[item] > 0:
                    indx[item] = 0
                    indx[item + 1:nPoints] = indx[item + 1:nPoints] - 1
                else:
                    indx[item] = 0
                x = np.delete(x, item - k)
                y = np.delete(y, item - k)
                dataStack = np.delete(dataStack, item - k, axis=0)
                k += 1
        stxmImage = np.reshape(stxmImage, (ypts, xpts))[::-1, :]
        if verbose:
            print "Removed %i bad frames." % (len(badIndices))
    """
    
    pass
    
def interpolate_STXM(img, ssy, ssx):
    """
    Attempts to intepolate the pixel image `img` on a grid that is
    ssy (ssx) finer in vertical (horizontal) direction
    """
    ypts,xpts = img.shape[-2:]
    yr,xr = ssy * ypts, ssx * xpts
    y,x = np.meshgrid(np.linspace(0,yr,ypts),np.linspace(0,xr,xpts))
    y,x = y.transpose(), x.transpose()
    yp,xp = np.meshgrid(np.linspace(0,yr, ypts * (ssy / ypixnm)),np.linspace(0,xr, xpts * (ssx / xpixnm)))
    yp,xp = yp.transpose(), xp.transpose()
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[0,1] - x0
    dy = y[1,0] - y0
    ivals = (xp - x0)/dx
    jvals = (yp - y0)/dy
    coords = np.array([ivals, jvals])
    stxmImageInterp = ndimage.map_coordinates(stxmImage.transpose(), coords)
    stxmImageInterp = stxmImageInterp * (stxmImageInterp > 0.)
    
    return stxmImageInterp
