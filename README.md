# Wide-field-QCSE-analysis
This repository contains all the notebooks used to analyze the spectra of single nanoparticles in response to voltage modulations. The acquisitions were done using a home-built wide-field spectrally-resolved microscope. <br/>
<br/>
The repository contains several ipython notebook (.ipynb) files: <br/>
<br/>
1. "wide-field QCSE spectra analysis.ipynb" shows the processes, including background subtraction, single particle detection, wavelength conversion (provided with calibration data extracted from https://github.com/yungkuo/wavelength-calibration), and extraction of the single particle intensity and wavelength time traces. It also caluclates averaged spectra and plot analysis results in a report figure.<br/>
2. "wide-field QCSE spectra-analysis- extract F and L trace-090817.ipynb" have the same work flow as 1. but saves time traces without plotting analysis report figures.<br/>
3. "Master notebook.ipynb" and "Master notebook-extract F and L trace.ipynb" were functions originally developed by Antonio Ingargiola. They allow repeating excution of notebook 1. and 2. to analyze different sets of data.<br/>
4. "Plot single particle statistics results.ipynb" compiles resutls extratced from notebook 1. and 2. and plots histograms.<br/>
5. "burst search from time trace.ipynb" demonstrates the burst search algorithm applied to data extratced from notebook 2.. It uses subfunctions contained in "burstsearch.py", which was modified from the original developemnt by Antonio Ingagiola. 

### Read the notebooks:
[wide-field QCSE spectra analysis.ipynb] (http://nbviewer.jupyter.org/github/yungkuo/wide-field-QCSE-analysis/blob/master/Wide-field%20QCSE%20spectra%20analysis.ipynb)
[Plot single particle statistics results.ipynb](http://nbviewer.jupyter.org/github/yungkuo/wide-field-QCSE-analysis/blob/master/Plot%20single%20particle%20statistics%20results.ipynb)
[burst search from time trace.ipynb] (http://nbviewer.jupyter.org/github/yungkuo/wide-field-QCSE-analysis/blob/master/burst%20search%20from%20time%20trace.ipynb)

### Data:
The raw data (.tif files) of all the QCSE measurements are deposited on Figshare: XX.

### Liscence
All the text and documentation are released under the CC BY 4.0 license. All code is under the MIT license (see LICENSE.txt).
<br/>
Copyright 2017 The Regents of the University of California, Yung Kuo, Antonino Ingargiola
