# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:44:27 2016

@author: yungkuo
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

def movingaverage(interval, window_size, mode='valid'):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, mode)

def get_dF(timetrace, frame_start=2):
    assert frame_start%2 == 0
    res = np.diff(timetrace[frame_start:])
    res[::2] *= -1
    return res

def get_dFF(timetrace, frame_start=2, running_avg_F=False, stitchends=True, window_size=4):
    assert frame_start%2 == 0
    assert window_size%2 == 0
    dF = np.diff(timetrace[frame_start:])
    dF[::2] *= -1
    F = np.array(timetrace[frame_start+1::2].repeat(2)[:-1])
    dFF = dF/F
    if running_avg_F:
        if not stitchends:
            F = movingaverage(F, window_size)
            dFF = dF[window_size//2:-window_size//2+1]/F
        else:
            F = timetrace[frame_start+1::2].repeat(2)
            F = np.append(np.append(F[-window_size//2:],F), F[:window_size//2])
            F = movingaverage(F, window_size)[:len(dF)]
            dFF = dF/F
    return dFF, F

def get_clipped_dff(tt, frame_start):
    signal, f = get_dFF(tt, frame_start)#, running_avg_F=True, stitchends=True, window_size=window)
    _min = np.median(signal)-5
    _max = np.median(signal)+5
    signal[np.logical_or(signal>_max, signal<_min)] = 0
    return np.array(signal), f

def get_clipped_dl(tt, frame_start):
    _min = np.median(tt[frame_start:])-30
    _max = np.median(tt[frame_start:])+30
    for a, l in enumerate(tt[frame_start:]):
        if np.logical_or(l>_max, l<_min):
            tt[a+frame_start] = tt[a+frame_start-1]
    signal = get_dF(tt, frame_start)
    return signal

def get_info_in_burst(b, x_signal, ttf, ttl, frame_start=2):
    dff, f = get_dFF(ttf, frame_start)#, running_avg_F=True, stitchends=True, window_size=window)
    dff_min = np.median(dff)-5
    dff_max = np.median(dff)+5
    
    l_min = np.median(ttl[frame_start:])-30
    l_max = np.median(ttl[frame_start:])+30
    for a, l in enumerate(ttl[frame_start:]):
        if np.logical_or(l>l_max, l<l_min):
            ttl[a+frame_start] = ttl[a+frame_start-1]
    
    seg = np.arange(x_signal[b['start']], x_signal[b['stop']]+1)
    seg1 = np.append(seg, seg[-1]+1)
    dFF, dL, = [], []
    #Fon, Foff, Lon, Loff = [], [], [], []
    for i in seg:
        if i%2 == 0:
            Fon1 = ttf[i]; Foff1 = ttf[i+1]
            Lon1 = ttl[i]; Loff1 = ttl[i+1]
        elif i%2 == 1:
            Fon1 = ttf[i+1]; Foff1 = ttf[i]
            Lon1 = ttl[i+1]; Loff1 = ttl[i]
        dff1 = (Fon1 - Foff1)/Foff1
        dl1 = (Lon1 - Loff1)
        if not np.logical_and(dff1<dff_max, dff1>dff_min):
            dff1 = 0
        dFF = np.append(dFF, dff1)
        dL = np.append(dL, dl1)
        #Fon = np.append(Fon, Fon1); Foff = np.append(Foff, Foff1)
        #Lon = np.append(Lon, Lon1); Loff = np.append(Loff, Loff1)
    cov = np.cov(dFF, dL)[0][1]
    cor = np.corrcoef(dFF, dL)[0][1]
    
    F = np.array(ttf[seg1])
    L = np.array(ttl[seg1])
    Fon = np.array([ttf[i] for i in seg1 if i%2==0])
    Lon = np.array([ttl[i] for i in seg1 if i%2==0])
    Foff = np.array([ttf[i] for i in seg1 if i%2==1])
    Loff = np.array([ttl[i] for i in seg1 if i%2==1])
    
    #shuffle trace
    idx = np.random.permutation(seg1)
    sF = np.array(ttf[idx])
    sL = np.array(ttl[idx])
    sdFF, sdL, = [], []
    for j, i in enumerate(seg):
        if i%2 == 0:
            Fon1 = sF[j]; Foff1 = sF[j+1]
            Lon1 = sL[j]; Loff1 = sL[j+1]
        elif i%2 == 1:
            Fon1 = sF[j+1]; Foff1 = sF[j]
            Lon1 = sL[j+1]; Loff1 = sL[j]
        dff1 = (Fon1 - Foff1)/Foff1
        dl1 = (Lon1 - Loff1)
        if not np.logical_and(dff1<dff_max, dff1>dff_min):
            dff1 = 0
        sdFF = np.append(sdFF, dff1)
        sdL = np.append(sdL, dl1)
    scov = np.cov(sdFF, sdL)[0][1]
    scor = np.corrcoef(sdFF, sdL)[0][1]
    
    sFon = np.array([sF[j] for j,i in enumerate(seg1) if i%2==0])
    sLon = np.array([sL[j] for j,i in enumerate(seg1) if i%2==0])
    sFoff = np.array([sF[j] for j,i in enumerate(seg1) if i%2==1])
    sLoff = np.array([sL[j] for j,i in enumerate(seg1) if i%2==1])
    
    assert np.all(np.sort(sF)==np.sort(F))
    assert np.all(np.sort(sL)==np.sort(L))
    assert np.all(np.sort(F) == np.sort(np.append(sFon,sFoff)))
    assert np.all(np.sort(L) == np.sort(np.append(sLon,sLoff)))
    info = {'dFF': dFF, 'Fon': Fon, 'Foff': Foff, 'F': F, 'sdFF': sdFF, 'sF': sF, 'sFon': sFon, 'sFoff': sFoff,
            'dL': dL,   'Lon': Lon, 'Loff': Loff, 'L': L, 'sdL':  sdL,  'sL': sL, 'sLon': sLon, 'sLoff': sLoff,
            'cov': cov, 'cor': cor, 'scov': scov, 'scor': scor}
    return info


import seaborn as sns
sns.set_style('ticks')
def plot_2d_hist(x, y, DF):
    x = DF[x]
    y = DF[y]
    index = np.logical_and(2>x, x>-2)&np.logical_and(20>y, y>-20)
    graph = sns.jointplot(x[index], y[index], kind="hex", gridsize=35, marginal_kws=dict(bins=35), 
                  cmap='gist_heat_r', color='r').set_axis_labels('$\Delta F/F$', '$\Delta \lambda$ (nm)')
    graph.ax_joint.axhline(y=0, c='k', ls='--', alpha=0.5)
    graph.ax_joint.axvline(x=0, c='k', ls='--', alpha=0.5)
    graph.ax_marg_y.axhline(y=0, c='k', ls='--', alpha=0.5)
    graph.ax_marg_x.axvline(x=0, c='k', ls='--', alpha=0.5)
    index0 = np.logical_and(2>x, x>0) & np.logical_and(20>y, y>-20) 
    graph.ax_marg_y.hist(y[index0], bins=35, histtype='step', color='b', orientation='horizontal', label='$\Delta F/F > 0$')
    index1 = np.logical_and(0>x, x>-2) & np.logical_and(20>y, y>-20)
    graph.ax_marg_y.hist(y[index1], bins=35, histtype='step', color='g', orientation='horizontal', label='$\Delta F/F < 0$')
    graph.ax_marg_y.legend()
    index0 = np.logical_and(2>x, x>-2) & np.logical_and(20>y, y>0) 
    graph.ax_marg_x.hist(x[index0], bins=35, histtype='step', color='b', label='$\Delta \lambda > 0$')
    index1 = np.logical_and(2>x, x>-2) & np.logical_and(0>y, y>-20)
    graph.ax_marg_x.hist(x[index1], bins=35, histtype='step', color='g', label='$\Delta \lambda < 0$')
    graph.ax_marg_x.legend()
    print('No. of cycles in the plot: %d' % np.sum(index))
    return 