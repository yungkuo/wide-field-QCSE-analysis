# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:44:27 2016

@author: yungkuo
"""

import numpy as np
import os
import scipy.optimize as opt
from lmfit.models import PolynomialModel, QuadraticModel
import scipy.ndimage as ndi
import pandas as pd
from scipy.interpolate import interp1d
from scipy import arange, array, exp


def listdir(path, startswith, endswith):
    for f in os.listdir(path):
        if f.startswith(startswith):
            if f.endswith(endswith):
                yield f

def get_roi_square(point, pad):
    """Return a square selection of pixels around `point`.
    """
    col, row = point
    col = int(col)
    row = int(row)
    mask = (slice(row-pad[0], row+pad[0]+1), slice(col-pad[1], col+pad[1]+1))
    return mask

def get_roi_square_3d(point, pad):
    """Return a square selection of pixels around `point`.
    """
    col, row = point
    col = int(col)
    row = int(row)
    mask = (slice(None, None), slice(row-pad[0], row+pad[0]+1), slice(col-pad[1], col+pad[1]+1))
    return mask

def get_local_max(image, point, pad):
    roi = get_roi_square(point, pad)
    col, row = point.astype('int')
    while image[row, col] != image[roi].max():
        row = row-pad[0]+np.where(image[roi]==image[roi].max())[0]
        col = col-pad[1]+np.where(image[roi]==image[roi].max())[1]
        point = [col,row]
        roi = get_roi_square(point, pad)
    return point


def get_background(movie, pts, scan):
    bg_indx= np.ones(movie.shape[1])
    for i in range(len(pts[:,1])):
        bg_indx[int(pts[i,1])-scan[0] : int(pts[i,1])+scan[0]] = 0
    bg_mask = np.tile(bg_indx,(movie.shape[2],1)).T
    bg = movie*bg_mask
    return bg

def get_stack_image(movie, pt, pad):
    roi = get_roi_square_3d(pt, pad)
    boxmovie = movie[roi]
    stack_image = np.reshape(boxmovie, (movie.shape[0]*(2*pad[0]+1),2*pad[1]+1)).T
    return stack_image



def get_timetrace(movie, pt, pad):
    roi = get_roi_square_3d(pt, pad)
    tt = movie[roi].mean(1).mean(1)
    return tt

def get_Von_spec(movie, pt, pad, threshold):
    roi = get_roi_square_3d(pt, pad)
    tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    movie = movie[::2]
    movie = np.array([movie[i] for i in range(len(movie)) if tt[::2][i] > threshold])
    spec = movie.mean(1).mean(0)
    return spec

def get_Voff_spec(movie, pt, pad, threshold):
    roi = get_roi_square_3d(pt, pad)
    tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    movie = movie[1::2]
    movie = np.array([movie[i] for i in range(len(movie)) if tt[1::2][i] > threshold])
    spec = movie.mean(1).mean(0)
    return spec

def get_peak_ccms(movie, pt, pad, x):
    roi = get_roi_square_3d(pt, pad)
    #tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    spec_series = movie.mean(1)
    peaks = np.sum(spec_series*x, axis=1)/np.sum(spec_series, axis=1)
    return peaks
    
def finddot(image, scan, nstd):
    pts = []
    for i in range(scan[0]*3, image.shape[0]-scan[0]*3, 1):
        for j in range(scan[1]*2, image.shape[1]-scan[1]*2, 1):
            roi = get_roi_square([j,i], scan)
            if image[i,j] == np.max(image[roi]):
                roi_big = get_roi_square([j,i], [scan[0]*3,scan[1]*2])
                if np.mean(image[roi]) > np.mean(image[roi_big])+nstd*np.std(image[roi_big], ddof=1):
                    pt = [j,i]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)//2,2])


def findparticle(image, boundary, scan, nstd):
    pts = []
    for i in np.arange(scan[0],image.shape[0]-scan[0],1, dtype='int'):
        for j in np.arange(boundary[0],boundary[1],1, dtype='int'):
            if image[i,j] == np.max(image[(i-scan[0]):(i+scan[0]),(j-scan[1]):(j+scan[1])]):
                #if (np.sum(image[(i-1):(i+2),(j-1):(j+2)])-image[i,j])/8 > (np.sum(image[(i-2):(i+3),(j-2):(j+3)])-np.sum(image[(i-1):(i+2),(j-1):(j+2)]))/16:
                if np.mean(image[(i-scan[0]):(i+scan[0]),(j-scan[1]):(j+scan[1])]) > np.mean(image[(i-scan[0]):(i+scan[0]),boundary[0]:boundary[1]])+nstd*np.std(image[(i-scan[0]):(i+scan[0]),boundary[0]:boundary[1]], ddof=1):
                    pt = [j,i]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)/2,2])


def mask3d(refimg, pts, scan):
    local = refimg[:,pts[1]-scan[0]:pts[1]+scan[0],pts[0]-scan[1]:pts[0]+scan[1]]
    return local

def shift_match(point_prism, point_dot):
    def func(p):
        return np.sum((point_dot[:,1]-(point_prism[:,1]*p[0]-p[1]))**2)
    res = opt.minimize(func, x0=np.array([1.,0.]), method='SLSQP')
    if res.success:
        magn = res.x[0]
        shift = res.x[1]
    return magn, shift

def get_blinking_threshold(movie, pt, pad, frame_start, iterations_to_find_threshold, nstd):
    tt = get_timetrace(movie, pt, pad)
    threshold = np.mean(tt[frame_start:])
    if iterations_to_find_threshold != 0:
        off_mean = np.mean([tt[i] for i in range(frame_start, len(tt)) if tt[i] < threshold])
        off_std = np.std([tt[i] for i in range(frame_start, len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
        threshold1 = off_mean + off_std*nstd
        for i in range(iterations_to_find_threshold):
            if threshold1 != threshold:
                threshold = threshold1
                off_mean = np.mean([tt[i] for i in range(frame_start, len(tt)) if tt[i] < threshold])
                off_std = np.std([tt[i] for i in range(frame_start, len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
                threshold1 = off_mean + off_std*nstd
    return threshold

def get_blinking_threshold_tt(tt, iterations_to_find_threshold, nstd):
    threshold = np.mean(tt)
    if iterations_to_find_threshold != 0:
        off_mean = np.mean([tt[i] for i in range(len(tt)) if tt[i] < threshold])
        off_std = np.std([tt[i] for i in range(len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
        threshold1 = off_mean + off_std*nstd
        for i in range(iterations_to_find_threshold):
            if threshold1 != threshold:
                threshold = threshold1
                off_mean = np.mean([tt[i] for i in range(len(tt)) if tt[i] < threshold])
                off_std = np.std([tt[i] for i in range(len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
                threshold1 = off_mean + off_std*nstd
    return threshold


def get_Gauss_filtered_image(image, sigma):
    image_LPed = ndi.gaussian_filter(image, sigma)
    image_HPed = image-ndi.gaussian_filter(image, sigma)
    return image_LPed, image_HPed



def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)
    def ufunclike(xs):
        return array(map(pointwise, array(xs)))
    return ufunclike

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



from sys import platform
if platform.startswith('win'):
    prePath = 'H:/My Drive/'
elif platform == 'darwin':
    prePath = '/Volumes/GoogleDrive/My Drive/'
    
pink = pd.read_pickle(prePath+'Code/QCSE analysis script/calibration/pink.pkl')
yellow = pd.read_pickle(prePath+'Code/QCSE analysis script/calibration/yellow.pkl')
blue = pd.read_pickle(prePath+'Code/QCSE analysis script/calibration/blue.pkl')
red = pd.read_pickle(prePath+'Code/QCSE analysis script/calibration/red.pkl')

def get_wavelength(x, dx, scan, x_or_y):
    mod = QuadraticModel()
    wavelengths = [515, 560, 645, 750]
    if x_or_y == 'x':
        dx_560 = x*pink.loc[(slice(None),['x']),['slope']]['slope']+pink.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_515 = x*yellow.loc[(slice(None),['x']),['slope']]['slope']+yellow.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_645 = x*blue.loc[(slice(None),['x']),['slope']]['slope']+blue.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_750 = x*red.loc[(slice(None),['x']),['slope']]['slope']+red.loc[(slice(None),['x']),['intercept']]['intercept']
    if x_or_y == 'y':
        dx_560 = x*pink.loc[(slice(None),['y']),['slope']]['slope']+pink.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_515 = x*yellow.loc[(slice(None),['y']),['slope']]['slope']+yellow.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_645 = x*blue.loc[(slice(None),['y']),['slope']]['slope']+blue.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_750 = x*red.loc[(slice(None),['y']),['slope']]['slope']+red.loc[(slice(None),['y']),['intercept']]['intercept']
    pars = mod.guess((dx_515.mean(),dx_560.mean(),dx_645.mean(),dx_750.mean()), x=wavelengths)
    result_2  = mod.fit((dx_515.mean(),dx_560.mean(),dx_645.mean(),dx_750.mean()), pars, x=wavelengths)
    
    a = result_2.best_values['a']
    b = result_2.best_values['b']
    c = result_2.best_values['c']
    #print(a,b,c)
    
    if x_or_y == 'x':
        dx = np.arange(dx-scan[1], dx+scan[1]+1, 1)
        wavelength = (-b+np.sqrt(b**2-4*a*(c-dx)))/(2*a)
    if x_or_y == 'y':
        dx = np.arange(dx-scan[0], dx+scan[0]+1, 1)
        wavelength = (-b-np.sqrt(b**2-4*a*(c-dx)))/(2*a)
    return wavelength


def get_wavelength_extrapolate_fill_nans(x, dx, scan, x_or_y):
    mod = QuadraticModel()
    wavelengths = [515, 560, 645, 750]
    if x_or_y == 'x':
        dx_560 = x*pink.loc[(slice(None),['x']),['slope']]['slope']+pink.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_515 = x*yellow.loc[(slice(None),['x']),['slope']]['slope']+yellow.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_645 = x*blue.loc[(slice(None),['x']),['slope']]['slope']+blue.loc[(slice(None),['x']),['intercept']]['intercept']
        dx_750 = x*red.loc[(slice(None),['x']),['slope']]['slope']+red.loc[(slice(None),['x']),['intercept']]['intercept']
    if x_or_y == 'y':
        dx_560 = x*pink.loc[(slice(None),['y']),['slope']]['slope']+pink.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_515 = x*yellow.loc[(slice(None),['y']),['slope']]['slope']+yellow.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_645 = x*blue.loc[(slice(None),['y']),['slope']]['slope']+blue.loc[(slice(None),['y']),['intercept']]['intercept']
        dx_750 = x*red.loc[(slice(None),['y']),['slope']]['slope']+red.loc[(slice(None),['y']),['intercept']]['intercept']
    pars = mod.guess((dx_515.mean(),dx_560.mean(),dx_645.mean(),dx_750.mean()), x=wavelengths)
    result_2  = mod.fit((dx_515.mean(),dx_560.mean(),dx_645.mean(),dx_750.mean()), pars, x=wavelengths)
    
    a = result_2.best_values['a']
    b = result_2.best_values['b']
    c = result_2.best_values['c']
    
    if x_or_y == 'x':
        dx = np.arange(dx-scan[1], dx+scan[1]+1, 1)
        wavelength = (-b+np.sqrt(b**2-4*a*(c-dx)))/(2*a)
    if x_or_y == 'y':
        dx = np.arange(dx-scan[0], dx+scan[0]+1, 1)
        wavelength = (-b-np.sqrt(b**2-4*a*(c-dx)))/(2*a)
    
    nans, x = nan_helper(wavelength)
    f_i = interp1d(np.arange(len(wavelength[~nans])), wavelength[~nans])
    f_x = extrap1d(f_i)
    wavelength[nans] = f_x(x(nans))

    return wavelength
