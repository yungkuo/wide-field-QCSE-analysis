# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:44:27 2016

@author: yungkuo
"""

import numpy as np
from lmfit.models import PolynomialModel
import scipy.ndimage as ndi
import pandas as pd
import ROI
import matplotlib.pyplot as plt
from IPython.display import display, FileLink

def plot_QCSE_report(movie, pt, scan, w, T, count, threshold, frame_start=2):
    tt = ROI.get_timetrace(movie, pt, scan)
    #threshold = ROI.get_blinking_threshold(movie[frame_start:,:,:], pt, scan, frame_start, 15, 1)
    Von_spec = ROI.get_Von_spec(movie, pt, scan, threshold)
    Voff_spec = ROI.get_Voff_spec(movie, pt, scan, threshold)

    mod = PolynomialModel(7)
    pars1 = mod.guess(Von_spec, x=w)
    pars2 = mod.guess(Voff_spec, x=w)
    result1 = mod.fit(Von_spec, pars1, x=w)
    result2 = mod.fit(Voff_spec, pars2, x=w)
    fitpeak1 = w[len(w)//4+int(np.where(result1.best_fit[(len(w)//4):-(len(w)//4)] == np.max(result1.best_fit[(len(w)//4):-(len(w)//4)]))[-1])]
    fitpeak2 = w[len(w)//4+int(np.where(result2.best_fit[(len(w)//4):-(len(w)//4)] == np.max(result2.best_fit[(len(w)//4):-(len(w)//4)]))[-1])]
    deltaL = fitpeak1 -fitpeak2
            
    P = ROI.get_peak_ccms(movie, pt, scan, w)
    P = np.clip(P, 460, 800)
    Pon = np.array([P[i] for i in range(frame_start, movie.shape[0]) if i%2==0 and tt[i] > threshold])
    Poff = np.array([P[i] for i in range(frame_start, movie.shape[0]) if i%2==1 and tt[i] > threshold])
    Ton = [T[i] for i in range(frame_start, movie.shape[0]) if tt[i] > threshold and i%2 == 0]
    Toff = [T[i] for i in range(frame_start, movie.shape[0]) if tt[i] > threshold and i%2 == 1]


    fig, ax = plt.subplots(3,5, figsize=(18,5))
    ax[0,0] = plt.subplot2grid((3,5), (0,0), colspan=4, rowspan=1)
    ax[1,0] = plt.subplot2grid((3,5), (1,0), colspan=4, rowspan=1)
    ax[2,0] = plt.subplot2grid((3,5), (2,0), colspan=4, rowspan=1, sharex=ax[1,0])
    ax[0,4] = plt.subplot2grid((3,5), (0,4), colspan=1, rowspan=1)
    ax[1,4] = plt.subplot2grid((3,5), (1,4), colspan=1, rowspan=1)
    ax[2,4] = plt.subplot2grid((3,5), (2,4), colspan=1, rowspan=1, sharex=ax[0,1], sharey=ax[2,0])

    extent = [0,movie.shape[0],0,scan[1]*2+1]
    stackmovie = ROI.get_stack_image(movie, pt, scan)
    ax[0,0].set_title('QD{}'.format(count))
    ax[0,0].set_ylabel('Pixel')
    ax[0,0].set_xlabel('Frame')
    ax[0,0].imshow(stackmovie, cmap='afmhot', vmin=np.min(stackmovie), vmax=np.max(stackmovie), extent=extent, aspect ='auto', interpolation='None')

    ax[1,0].plot(T, tt[:], 'g')
    ax[1,0].plot(T[frame_start::2], tt[frame_start::2], 'b.')
    ax[1,0].set_xlim([0,T.max()])
    ax[1,0].axhline(y = threshold, c='r')
    ax[1,0].set_ylim(np.min(tt[frame_start:]), np.max(tt[frame_start:]))
    ax[1,0].set_ylabel('Intensity')
    ax[1,0].set_xlabel('Time (s)')

    ax[2,0].plot(Toff, Poff, 'b.')
    ax[2,0].plot(Ton, Pon, 'r.')
    ax[2,0].plot(T, P, c='0.8')
    ax[2,0].set_ylim(Poff.mean()-25, Poff.mean()+25)
    ax[2,0].set_ylabel('Peak COM (nm)')
    ax[2,0].set_xlabel('Time (s)')

    ax[0,4].plot(w, Von_spec, 'r.')#, label='Von data')
    ax[0,4].plot(w, Voff_spec, 'b.')#, label='Voff data')
    ax[0,4].plot(w, result1.best_fit, '-', label='Von (%.2f nm)'% fitpeak1, color='r')
    ax[0,4].plot(w, result2.best_fit, '-', label='Voff (%.2f nm)'% fitpeak2, color='b')
    ax[0,4].axvline(x=fitpeak1, alpha=0.3, linestyle='--', color='r')
    ax[0,4].axvline(x=fitpeak2, alpha=0.3, linestyle='--', color='b')
    ax[0,4].annotate('$\Delta$$\lambda$ = %.2f nm'% deltaL, xy=(1,1), xytext=(1.2,1), xycoords='axes fraction', fontsize=10)
    ax[0,4].legend(bbox_to_anchor=(1.9, 1), frameon=False, fontsize=10)
    ax[0,4].set_xlabel('Wavelength (nm)')
    ax[0,4].set_ylabel('Intensity')
    plt.setp(ax[0,4].get_xticklabels(), fontsize=10, rotation=25)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    counts, bins, patches = ax[1,4].hist(tt[frame_start::2], bins=25, histtype='bar', orientation='horizontal', label='Intensity', color='b')
    ax[1,4].set_ylim(np.min(tt[frame_start:]), np.max(tt[frame_start:]))
    ax[1,4].axhline(y = threshold, c='r')
    ax[1,4].set_ylabel('Intensity')
    ax[1,4].set_xlabel('Counts')

    BINS = np.linspace(Poff.mean()-25, Poff.mean()+25, 25)
    counts, bins, patches = ax[2,4].hist(Pon, bins=BINS, histtype='bar', orientation='horizontal', alpha=0.5, label='Von ({}nm)'.format(round(Pon.mean(),2)), color='r')
    counts, bins, patches = ax[2,4].hist(Poff, bins=BINS, histtype='bar', orientation='horizontal', alpha=0.5, label='Voff ({}nm)'.format(round(Poff.mean(),2)), color='b')
    ax[2,4].legend(bbox_to_anchor=(1.2, 1), frameon=False, fontsize=10)
    ax[2,4].axhline(y=Pon.mean(), color='r')
    ax[2,4].axhline(y=Poff.mean(), color='b')
    ax[2,4].annotate('$\Delta$$\lambda$ = {} nm'.format(round(Pon.mean()-Poff.mean(),2)), xy=(1,1), xytext=(1.2,1), xycoords='axes fraction', fontsize=10)
    ax[2,4].set_ylabel('Peak COM (nm)')
    ax[2,4].set_xlabel('Counts')
    
    dL1 = pd.DataFrame({'dL':deltaL,
                        'dL_COM': [Pon.mean()-Poff.mean()],
                        'Lon': fitpeak1,
                        'Loff': fitpeak2,
                        'Lon_COM': Pon.mean(),
                        'Loff_COM': Poff.mean(),
                        'dF/F': [np.sum(Von_spec-Voff_spec)/np.sum(Voff_spec)],
                        'dE (meV)': (1240/fitpeak1-1240/fitpeak2)*1000,
                        'dE_COM (meV)': (1240/Pon.mean()-1240/Poff.mean())*1000},
                      index=[count])
    return fig, dL1


def plot_zoomed_region(movie, pt, scan, w, frame_period, frame_start=2):
    data = movie.shape
    x_frame = np.arange(0, data[0], 1)

    tt = ROI.get_timetrace(movie, pt, scan)
    threshold = ROI.get_blinking_threshold(movie[frame_start:,:,:], pt, scan, frame_start, 5, 0.8)
    P = ROI.get_peak_ccms(movie, pt, scan, w)
    P = np.clip(P, 460, 800)
    Pon = np.array([P[i] for i in range(frame_start, data[0]) if i%2==0 and tt[i] > threshold])
    Poff = np.array([P[i] for i in range(frame_start, data[0]) if i%2==1 and tt[i] > threshold])
    x_frame_on = [x_frame[i] for i in range(frame_start, data[0]) if tt[i] > threshold and i%2 == 0]
    x_frame_off = [x_frame[i] for i in range(frame_start, data[0]) if tt[i] > threshold and i%2 == 1]
    stackmovie = ROI.get_stack_image(movie[frame_period[0]:frame_period[1],:,:], pt, scan)

    extent = [frame_period[0],frame_period[1],0,scan[1]*2]
    fig, ax = plt.subplots(2,1, figsize=(18,3))
    ax[0].imshow(stackmovie, cmap='afmhot', vmin=np.min(stackmovie), vmax=np.max(stackmovie), 
              extent=extent, aspect ='auto', interpolation='None')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Pixel')
    ax[1].plot(x_frame_off, Poff, 'b.')
    ax[1].plot(x_frame_on, Pon, 'r.')
    ax[1].plot(x_frame, P, c='0.7')
    ax[1].set_xlim(frame_period)
    period_slice = slice(frame_period[0]//2,frame_period[1]//2)
    ax[1].set_ylim(np.min(np.append(Poff[period_slice], Pon[period_slice])), np.max(np.append(Poff[period_slice], Pon[period_slice])))
    ax[1].grid(True)
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Peak COM (nm)')
    
    return
    
def plot_big_figure(movie, pt, scan, w, name, frame_start=2):
    data = movie.shape
    x_frame = np.arange(0, data[0], 1)

    tt = ROI.get_timetrace(movie, pt, scan)
    threshold = ROI.get_blinking_threshold(movie[frame_start:,:,:], pt, scan, frame_start, 5, 0.8)
    P = ROI.get_peak_ccms(movie, pt, scan, w)
    P = np.clip(P, 460, 800)
    Pon = np.array([P[i] for i in range(frame_start, data[0]) if i%2==0 and tt[i] > threshold])
    Poff = np.array([P[i] for i in range(frame_start, data[0]) if i%2==1 and tt[i] > threshold])
    x_frame_on = [x_frame[i] for i in range(frame_start, data[0]) if tt[i] > threshold and i%2 == 0]
    x_frame_off = [x_frame[i] for i in range(frame_start, data[0]) if tt[i] > threshold and i%2 == 1]
    stackmovie = ROI.get_stack_image(movie, pt, scan)

    fig, ax = plt.subplots(3,1, figsize=(100,3))
    ax[0].imshow(stackmovie, cmap='afmhot', vmin=np.min(stackmovie), vmax=np.max(stackmovie), 
                 aspect ='auto', interpolation='None')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Pixel')
    ax[1].plot(x_frame, tt, 'g')
    
    ax[2].plot(x_frame_off, Poff, 'b.')
    ax[2].plot(x_frame_on, Pon, 'r.')
    ax[2].plot(x_frame, P, c='0.7')
    ax[2].set_ylim(np.min(np.append(Poff, Pon)), np.max(np.append(Poff, Pon)))
    ax[2].grid(True)
    ax[2].set_xlabel('Frame')
    ax[2].set_ylabel('Peak COM (nm)')
   
    name = name+'.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)
    #display(FileLink(name))
    
    return display(FileLink(name))