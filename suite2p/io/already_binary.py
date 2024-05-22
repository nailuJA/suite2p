# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 15:08:20 2023

@author: Eddie
"""

import math
import gc
import numpy as np
from .utils import init_ops, find_files_open_binaries
from .binary import BinaryRWFile
from typing import Tuple#, Union, Optional


def open_bin(biny: int, binx: int, file: str) -> Tuple[BinaryRWFile, int]:
    """ Returns image and its length from bin file """

    binf = BinaryRWFile(biny, binx, file)
    Lbinf = 1 if len(binf.shape) < 3 else binf.shape[0]  # single page tiffs
    return binf, Lbinf

def already_binary(ops):
    """  finds binaries created by prairie link and makes them ready to be
         analyzed in the suite2p pipeline

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops : dictionary of first plane
            'Ly', 'Lx', ops['reg_file'] or ops['raw_file'] is created binary

    """
    ops1 = init_ops(ops)
    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']
    biny = ops1[0]['biny'] #These two need to be defined when starting the process 
    binx = ops1[0]['binx'] #currently with Jupyter script
    
    ops1, binlist, reg_file, reg_file_chan2 = find_files_open_binaries(ops1, False)
    ops = ops1[0]
    
    batch_size = ops['batch_size']
    batch_size = nplanes*nchannels*math.ceil(batch_size/(nplanes*nchannels))
    
    # loop over all tiffs
    which_folder = -1
    ntotal=0
    
    #TODO check tiff.py starting line 130 for inspiration
    
    for ik, file in enumerate(binlist):
        
        
        binf, Lbinf = open_bin(biny, binx, file)
        ix = 0
        
        while 1:
            if ix >= Lbinf:
                break
            nfr = max(Lbinf - ix, batch_size)
            # tiff reading

            im = binf.data
            iplane = 0
            which_folder = 0


            if im.shape[0] > nfr:
                im = im[:nfr, :, :]
            nframes = im.shape[0]
            for j in range(0,nplanes):
                if ik==0 and ix==0:
                    ops1[j]['nframes'] = 0
                    ops1[j]['frames_per_file'] = np.zeros((len(binlist),), dtype=int)
                    ops1[j]['meanImg'] = np.zeros((im.shape[1], im.shape[2]), np.float32)
                    if nchannels>1:
                        ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1], im.shape[2]), np.float32)
                i0 = nchannels * ((iplane+j)%nplanes)
                if nchannels>1:
                    nfunc = ops['functional_chan']-1
                else:
                    nfunc = 0
                im2write = im[int(i0)+nfunc:nframes:nplanes*nchannels]

                reg_file[j].write(bytearray(im2write))
                ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                ops1[j]['nframes'] += im2write.shape[0]
                ops1[j]['frames_per_file'][ik] += im2write.shape[0]
                ops1[j]['frames_per_folder'][which_folder] += im2write.shape[0]
                #print(ops1[j]['frames_per_folder'][which_folder])
                if nchannels>1:
                    im2write = im[int(i0)+1-nfunc:nframes:nplanes*nchannels]
                    reg_file_chan2[j].write(bytearray(im2write))
                    ops1[j]['meanImg_chan2'] += im2write.mean(axis=0)


            iplane = (iplane-nframes/nchannels)%nplanes
            ix+=nframes
            ntotal+=nframes
            #if ntotal%(batch_size*4)==0:
            #    print('%d frames of binary, time %0.2f sec.'%(ntotal,time.time()-t0))
        gc.collect()
    # write ops files
    do_registration = ops['do_registration']
    for ops in ops1:
        ops['Ly'],ops['Lx'] = ops['meanImg'].shape
        ops['yrange'] = np.array([0,ops['Ly']])
        ops['xrange'] = np.array([0,ops['Lx']])
        ops['meanImg'] /= ops['nframes']
        if nchannels>1:
            ops['meanImg_chan2'] /= ops['nframes']
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(0,nplanes):
        reg_file[j].close()
        if nchannels>1:
            reg_file_chan2[j].close()
    return ops1[0]
        