(c) Vyacheslav (Slava) Leshchneko 
leschenkoslava@gmail.com

%main features
The program is for pulse reconstruction from SHG-FROG using PCGPA algorithm.
Latest ideas from Trebino's team are implemented in the code including better first spectrum guess and 
multi-grid algorithm (option to choose) [Opt. Express 26, 2643-2649 (2018), Opt. Express 27, 2112-2124 (2019)]

The program is written in Python. The interface in Qt. (I have only tested it on Windows with Python3)
Therefore, check that you have the PyQt5 and pyqtgraph packages. In principle, try to run, and errors will indicate which package is missing if any.

An automated binned is included, which allows almost on fly reconstruction of raw experimental data. 
(according to multiple real life tests, it seems to be working well with at least roughly compressed pulses.)

% some instructions
A number of input file formats can be used (though most of them are specific to our lab data acquisition software)
The standard (from free Trebino's soft) *.frg file format is also included.
a generic FROG raw data file format that is accepted by the program should have .txt extension, and the following structure
1st line: delays
2d line: wavelengths in nm
following lines: data, so that each line is a spectrum for different delays (mast not be though, since there is a transpose option in the interface)
(you can also add your file format to load_frog function in PCGPA.py)

fundamental spectrum can also have multiple formats (but they are also specific to our lab data acquisition software)
a generic spectrum data file format that is accepted by the program should have .txt or .dat extension, 
and the two-column structure with the first column being wavelength in nm and the second column being intensity

