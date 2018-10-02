#
# Copyright 2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines functions to detect bursts of signal.
"""

from __future__ import division
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def burstsearch_py(signal, m, threshold, t_threshold, debug=False):
    """Sliding window burst search. Pure python version.
    Returns:
        Record array of burst data, one element per burst.
        Each element is a composite data type containing
        burst start, burst stop and burst score (the
        integrated signal over the burst duration).
    """
    bursts = []
    in_burst = False
    score = signal[:m].mean()
    deltasignal = (signal[m:] - signal[:-m])/m
    if debug:
        score_list = [score]
    for i, delta in enumerate(deltasignal):
        if np.abs(score)**2 > threshold:
            if not in_burst:
                # index of first cycle in burst
                start = i
                in_burst = True
        elif in_burst:
            # index of last cycle in burst
            stop = i - 1
            totalscore = signal[start:stop + 1].sum()
            in_burst = False
            if stop-start+1 > t_threshold:
                bursts.append((start, stop, totalscore))
        score += delta
        if debug:
            score_list.append(score)

    # Create numpy recarray
    dt = np.dtype([('start','int32'), ('stop','int32'), ('score', 'float64')])
    bursts = np.array(bursts, dtype=dt).view(np.recarray)
    if debug:
        return bursts, np.array(score_list)
    else:
        return bursts

    
def plot_tt_with_burst(tt, bursts, score, x_score, signal, x_signal, threshold, frame_start, kind='$\Delta F/F$', bins=20):
    fig, ax = plt.subplots(1,5, figsize=(20,4))
    ax0 = plt.subplot2grid((1,5), (0,0), colspan=4)
    ax1 = plt.subplot2grid((1,5), (0,4), colspan=1)
    #ax2 = plt.subplot2grid((1,5), (1,2), colspan=2)
    
    #Plot traces
    ax0.plot(x_signal, signal**2, label=kind+' $score^2$', alpha=0.5)
    ax0.plot(x_score, score**2*8, label='moving averaged $score^2*8$', ls='--', c='k')
    ax0.axhline(y=threshold*8, color='r', ls='--', label='threshold*8')
    for b in bursts:
        ax0.axvspan(x_score[b['start']], x_score[b['stop']], alpha=0.15)
    ax0.legend(fontsize=14, bbox_to_anchor=(0.8, 1.1))
    ax0.set_ylabel('$score^2$', fontsize=16)
    ax0.set_xlabel('Frame', fontsize=16)
    ax0.set_title(kind+' burst search' , fontsize=20)
    for tick in ax0.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax0.yaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
        
    ax0t = ax0.twinx()
    ax0t.plot(tt, alpha=0.5, c='g')
    ax0t.grid(False)
    ax0t.plot(np.arange(len(tt))[frame_start::2], tt[frame_start::2], 'ro', ms=3)
    ax0t.plot(np.arange(len(tt))[frame_start+1::2], tt[frame_start+1::2], 'bo', ms=3)
    if kind=='$\Delta F/F$':
        ax0t.set_ylabel('Fluorescence intensity (a.u.)', color='g', fontsize=16)
    else:
        ax0t.set_ylabel('Wavelength (nm)', color='g', fontsize=16)
    ax0t.axhline(y=0, ls='--', c='k')
    plt.yticks(color='g',fontsize=16)
    ax0t.set_ylim(tt[frame_start:].min()-(tt[frame_start:].max()-tt[frame_start:].min()), tt[frame_start:].max())

    #Plot score histrgram
    c,b,p = ax1.hist(score, bins=bins, color='b', alpha=0.5);
    if kind=='$\Delta F/F$':
        ax1.axvline(x=1, ls='--', c='k')
        ax1.axvline(x=-1, ls='--', c='k')
        ax1.axvline(x=np.mean(score), ls='-', c='c')
        ax1.axvline(x=np.mean(score)+np.std(score, ddof=1), ls=':', c='c')
        ax1.axvline(x=np.mean(score)-np.std(score, ddof=1), ls=':', c='c')
        ax1.set_title(kind + 'score')
    else:
        ax1.set_title('$\Delta \lambda$ score')
       
    #Plot tt histogram
    #c,b,p = ax2.hist(np.array(tt[frame_start:]), bins=30, alpha=0.5, color='g');
    #ax2.hist(np.array(tt[frame_start+1::2]), bins=b, alpha=0.5, color='b');
    #ax2.hist(np.array(tt[frame_start::2]), bins=b, alpha=0.5, color='r');
    #if kind=='$\Delta F/F$':
    #    ax2.set_title('F tt')
    #else:
    #    ax2.set_title('$\lambda$ tt')
    fig.tight_layout()
    return fig, [ax0, ax0t, ax1]


def get_No_NRs(DF_bursts):
    a = np.where(DF_bursts['NR No.']==0)[0]
    idx = [a[i] for i in range(len(a)) if a[i]-a[i-1] != 1]
    NoNR = 0
    if len(idx) > 1:
        for i in range(len(idx)-1):
            NoNR1 = np.unique(DF_bursts['NR No.'][idx[i]:idx[i+1]])
            NoNR = NoNR + len(NoNR1)
        NoNR1 = np.unique(DF_bursts['NR No.'][idx[i+1]:])
        NoNR = NoNR + len(NoNR1)
    elif len(idx) == 1:
        NoNR = len(np.unique(DF_bursts['NR No.']))
    return NoNR


import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
cmap1 = matplotlib.cm.get_cmap('spring')

def plot_bursts_hist(DF_bursts, bins=np.arange(-3,5,0.1), hue='ncycle', weight=None, normed=True, _x=1, kind='$\Delta F/F$', cmap=cmap):
    fig, ax = plt.subplots(figsize=(8,8))
    x = bins.repeat(2)[1:-1]
    labels = ' '
    for i, name in enumerate(np.unique(DF_bursts[hue])):
        labels = np.append(labels, name)
        select = DF_bursts[hue]==name
        data = DF_bursts[select]['score/width']
        weights = np.abs(DF_bursts[select][weight], dtype='float')
        # get number of NRs
        NoNR = get_No_NRs(DF_bursts[select])
        #histograms
        counts, b = np.histogram(data, bins=bins, normed=normed)
        y = counts.repeat(2)
        ax.plot(x, (y*_x+i), c=cmap(i/len(np.unique(DF_bursts[hue]))), ls='--', alpha=0.4)

        counts, b = np.histogram(data, bins=bins, weights=weights, normed=normed)
        y = counts.repeat(2)
        ax.plot(x, (y*_x+i), c=cmap(i/len(np.unique(DF_bursts[hue]))), 
                label='n=%d (%d)' % (np.sum(DF_bursts[hue]==name), NoNR))
    if kind == '$\Delta F/F$':
        ax.axvline(x=-1, ls='--', c='gray')
        ax.axvline(x=1, ls='--', c='gray')
        ax.set_xlabel('$\Delta F/F$', fontsize=20)
    else:
        ax.set_xlabel('$\Delta \lambda$', fontsize=20)
    ax.legend(bbox_to_anchor=(1.4,1), fontsize=14)
    ax.set_ylabel(hue, fontsize=20)
    if hue == 'NR':
        ax.set_ylabel('Type of nanosensors', fontsize=20)
    
    l = []
    for name in labels: 
        if name == '1296':
            label = '12nm \n type-II \n ZnSe/CdS NR'
        elif name == '1296BZn':
            label = '(v) 12nm \n type-II \n ZnSe/CdS NR'
        elif name == '40nmNR':
            label = '(iv) 40nm \n quasi-type-I \n CdSe/CdS NR'
        elif name == '40nmNRTe':
            label = '(iii) 40nm \n quasi-type-I \n Te-doped CdSe/CdS NR'
        elif name == 'kQD':
            label = '(ii) 12nm \n quasi-type-I \n CdS/CdSe/CdS QD'
        elif name == 'rQD':
            label = '(i) 6nm \n type-I \n CdSe/ZnS QD'
        elif name == 'Mn':
            label = ' \n type-I \n Mn-doped QD'
        elif name == '9BZn':
            label = '12nm \n type-II \n ZnSe/CdS NR \n (batch #2)'
        else:
            label = name
        l = np.append(l, label)
    ax.set_yticklabels(l, fontsize=14);
    plt.setp(ax.get_xticklabels(), fontsize=14)
    #ax[i].grid()
    return fig, ax