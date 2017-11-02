# Wide-field-QCSE-analysis
This repository contains all the notebooks used to analyze the spectra of single nanoparticles in response to voltage modulations. The acquisitions were done using a home-built wide-field spectrally-resolved microscope. <br/>
<br/>
The repository contains several ipython notebook (.ipynb) files: <br/>
<br/>
1. "wide-field QCSE spectra analysis.ipynb" shows the processes, including background subtraction, single particle detection, wavelength conversion (provided with calibration data extracted from ), and extraction of the single particle intensity and wavelength time traces. It also caluclates averaged spectra and plot analysis results in a report figure.<br/>
2. "wide-field QCSE spectra-analysis- extract F and L trace-090817.ipynb" have the same work flow as 1. but saves time traces without plotting analysis report figures.<br/>
3. "Master notebook.ipynb" and "Master notebook-extract F and L trace.ipynb" were functions originally developed by Antonio Ingargiola. They allow repeating excution of notebook 1. and 2. to analyze different sets of data.<br/>
4. "Plot single particle statistics results.ipynb" compiles resutls extratced from notebook 1. and 2. and plots histograms.<br/>
5. "burst search from time trace.ipynb" demonstrates the burst search algorithm applied to data extratced from notebook 2.. It uses subfunctions contained in "burstsearch.py", which was modified from the original developemnt by Antonio Ingagiola. 



