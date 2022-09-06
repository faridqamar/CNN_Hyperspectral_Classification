#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

def read_header(hdrfile, verbose=True):
    """
    Read a Middleton header file.
    """

    # -- alert
    if verbose:
        print("reading and parsing {0}...".format(hdrfile))

    # -- open the file and read in the records
    recs = [rec for rec in open(hdrfile)]

    # -- parse for samples, lines, bands, and the start of the wavelengths
    for irec, rec in enumerate(recs):
        if 'samples' in rec:
            samples = int(rec.split("=")[1])
        elif 'lines' in rec:
            lines = int(rec.split("=")[1])
        elif 'bands' in rec:
            bands = int(rec.split("=")[1])
        elif "Wavelength" in rec:
            w0ind = irec+1

    # -- parse for the wavelengths
    waves = np.array([float(rec.split(",")[0]) for rec in 
                      recs[w0ind:w0ind+bands]])

    # -- return a dictionary
    return {"nrow":samples, "ncol":lines, "nwav":bands, "waves":waves}


def read_raw(rawfile, shape, dtype=np.uint16):
    """
    Read a Middleton raw file.
    """

    # -- alert
    print("reading {0}...".format(rawfile))

    # return np.fromfile(open(rawfile),dtype) \
    #     .reshape(shape[2],shape[0],shape[1])[:,:,::-1] \
    #     .transpose(1,2,0)
    return np.memmap(open(rawfile),dtype,mode="r") \
        .reshape(shape[2],shape[0],shape[1])[:,:,::-1] \
        .transpose(1,2,0)


def read_hyper(fpath, fname=None, full=True):
    """
    Read a full hyperspectral scan.
    """

    # -- set up the file names
    if fname is not None:
        fpath = os.path.join(fpath,fname)

    # -- read the header
    hdr = read_header(fpath.replace("raw","hdr"))
    sh  = (hdr["nwav"],hdr["nrow"],hdr["ncol"])

    # -- if desired, only output data cube
    if not full:
        return read_raw(fpath,sh)

    # -- output full structure
    class output():
        def __init__(self,fpath):
            self.filname = fpath
            self.data    = read_raw(fpath,sh)
            self.waves   = hdr["waves"]
            self.nwav    = sh[0]
            self.nrow    = sh[1]
            self.ncol    = sh[2]

    return output(fpath)


def read_clean(fpath, fname=None, shape=None):
    """
    Read a cleaned hyperspectral scan.
    """

    # -- set up the file names
    if fname is not None:
        fpath = os.path.join(fpath,fname)

    # -- alert
    print("reading {0}...".format(fpath))

    # -- read and return
    if shape is None:
        return np.fromfile(fpath,dtype=float)
    else:
        return np.fromfile(fpath,dtype=float).reshape(shape)



def hyper_viz(cube, img, asp=0.45):
    """
    Visualize a hyperspectral data cube.
    """


    def update_spec(event):
        if event.inaxes == axim:
            rind = int(event.ydata)
            cind = int(event.xdata)

            tspec = cube.data[:,rind,cind]
            linsp.set_data(cube.waves,cube.data[:,rind,cind])
            axsp.set_ylim(tspec.min(),tspec.max()*1.1)
            axsp.set_title("({0},{1})".format(rind,cind))

            fig.canvas.draw()


    # -- set up the plot
    fig, ax    = plt.subplots(2,1,figsize=(10,10))
    axsp, axim = ax

    # -- show the image
    axim.axis("off")
    im = axim.imshow(img,"gist_gray",interpolation="nearest",aspect=asp)

    # -- show the spectrum
    axsp.set_xlim(cube.waves[0],cube.waves[-1])
    linsp, = axsp.plot(cube.waves,cube.data[:,0,0])

    fig.canvas.draw()
    fig.canvas.mpl_connect("motion_notify_event",update_spec)

    plt.show()

    return


def make_rgb8(data,waves,lam=[610.,540.,475.],scl=2.5):
    ind = [np.argmin(np.abs(waves-clr)) for clr in lam]
    rgb = data[ind]
    wgt = rgb.mean(0).mean(0)
    scl = scl*wgt[0]/wgt * 2.**8/2.**12

    return (rgb*scl).clip(0,255).astype(np.uint8).transpose(1,2,0)

