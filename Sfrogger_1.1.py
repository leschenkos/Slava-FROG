"""FROG reconstraction with Qt GUI
v1.1 @ Vyacheslav Leshchenko 2019

to do:
axis scale of frog
ask about rewriting by saving
negative values in a loaded spectrum
"""
import time     
import sys

#from the file directory find the first folder containing the name python and append it
import os
Path=os.path.abspath(__file__)
SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)
from PyQt5 import QtWidgets
import FROG_Qt
import numpy as np
Pi=np.pi
from multiprocessing import Pool
import pyqtgraph as pg
from PyQt5 import QtGui
import classes.error_class as ER
from classes.Pulse_class import width, remove_phase_jumps
from color_maps.color_maps import ImageColorMap
from scipy import interpolate
import PCGPA
from constants import c
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
from load_files.load_folder import imp_spec
from multiprocessing import Pool, cpu_count


class FROG_class(QtWidgets.QMainWindow, FROG_Qt.Ui_MainWindow):
    def __init__(self):
        self.app=QtWidgets.QApplication(sys.argv)
        super().__init__()
        self.setupUi(self)
        
        self.startBt.clicked.connect(self.fstart)
        self.continueBt.clicked.connect(self.fcontinue)
        self.stopBt.clicked.connect(self.fstop)
        self.browser_fun.clicked.connect(self.load_fund)
        self.browser_FROG.clicked.connect(self.open_SHG)
        self.checkmaxwavelength.clicked.connect(self.maxwavelength_click)
        self.maxwavelength.valueChanged.connect(self.set_wavelength)
        self.checkminwavelength.clicked.connect(self.minwavelength_click)
        self.minwavelength.valueChanged.connect(self.set_wavelength)
        self.checktranspose.clicked.connect(self.transpose_frog)
        self.checkdelaycorrection.clicked.connect(self.delaycorrection_click)
        self.delay_correction.valueChanged.connect(self.set_delaycorrection)
        self.check_bkgsubtract.clicked.connect(self.bkgsubtract_click)
        self.bkg.valueChanged.connect(self.set_bkg)
        self.multi_grid.clicked.connect(self.multi_grid_click)
        self.check_logscale.clicked.connect(self.logscale_click)
        self.size.valueChanged.connect(self.set_size)
        self.check_symmetry.clicked.connect(self.symmetry_click)
                        

        self.actionsave_FROG.triggered.connect(self.savefrog)
        self.actionsave_experimental_FROG.triggered.connect(self.savefrog_in)
        self.actionsave_time.triggered.connect(self.savetime)
        self.actionsave_spectrum.triggered.connect(self.savespectrum)
        self.actionsave_all.triggered.connect(self.saveall)


    Args={'frog_file' : '' ,'fundspec_file' : '','fundspec_xtype' : 'nm',
          'parallel_poolobject' : None, 'MaxWavelength' : None, 'MinWavelength' : None, 'fix_fund_spec' : False,
          'use_fund_spec' : False,
          'ret_population' : 20,'G_goal' : 10**-3,'ret_iterations' : 50, 'hi' : 0.5, 'Gausfiltercof' : 0.7, 
          'parallel_proc' : 4, 'delay correction' : False, 'delay corection coefficient' : 1.,
          'multi_grid' : False, 'parallel' : False, 'substract_bkg' : False, 'background' : 0.,
          'symmetrization' : False, 'logscale' : False, 'max size' : 0}
    
    Results={'pulse' : None, 'gate' : None, 'G' : None, 'frog_load' : [],'frog_in' : None, 'frog_out' : None, 
             'frog_in_0' : None,'W' : None,'W_load' : None, 'Wf' : None, 'Sfund' : None, 'T' : None, 'T_load' : None, 
             'Sfund_load' : None, 'Step' : 0, 'Time' : 0, 'pulse_w' : None, 'frog_in_processed?' : False,
             'G_best' : None, 'pulse_best' : None, 'pulse_w_best' : None, 'frog_out_best' : None}
    
    def open_SHG(self):
        self.error_message.setPlaceholderText('')#clear errors
        file=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if not file == '':
            self.Args['frog_file']=file
            self.Results['frog_in_processed?']=False
            self.file_SHG.clear()
            self.file_SHG.insert(self.Args['frog_file'])
            try:
                self.loadFROG()
                self.set_wavelength()
                self.set_bkg()
                self.showFROGex()
            except ER.SL_exception as error:
                self.showerror(error)
                self.file_SHG.clear()
                self.Args['frog_file'] = ''
    
    def set_wavelength(self):
        if self.checkmaxwavelength.isChecked() or self.checkminwavelength.isChecked():
            self.Args['MaxWavelength']=self.maxwavelength.value()
            self.Args['MinWavelength']=self.minwavelength.value()
            if not self.Args['frog_file'] == '':
                if self.checkmaxwavelength.isChecked():
                    Wmin=2*Pi*c/self.Args['MaxWavelength']*10**9*10**-15
                else:
                    Wmin=self.Results['W_load'][0]
                if self.checkminwavelength.isChecked():
                    Wmax=2*Pi*c/self.Args['MinWavelength']*10**9*10**-15
                else:
                    Wmax=self.Results['W_load'][-1]
                    
                ind=np.logical_and(self.Results['W_load'] > Wmin , self.Results['W_load'] < Wmax)
                self.Results['W']=np.copy(self.Results['W_load'][ind])
                self.Results['frog_in_0']=self.Results['frog_load'][:,ind]
                if self.Args['substract_bkg']:
                    self.remove_bkg()
                else:
                    self.Results['frog_in']=self.Results['frog_in_0']
                self.showFROGex()
        else:
            if not self.Args['frog_file'] == '':
                self.Results['W']=np.copy(self.Results['W_load'])
                self.Results['frog_in_0']=self.Results['frog_load']
                if self.Args['substract_bkg']:
                    self.remove_bkg()
                else:
                    self.Results['frog_in']=self.Results['frog_in_0']
                self.showFROGex()
        self.Results['frog_in_processed?']=False        
    
    def maxwavelength_click(self):
        if self.checkmaxwavelength.isChecked():
            self.maxwavelength.setEnabled(True)            
        else:
            self.maxwavelength.setEnabled(False)
        self.set_wavelength()  
        
    def minwavelength_click(self):
        if self.checkminwavelength.isChecked():
            self.minwavelength.setEnabled(True)            
        else:
            self.minwavelength.setEnabled(False)
        self.set_wavelength()  

    def set_delaycorrection(self):
        if self.checkdelaycorrection.isChecked():
            self.Args['delay correction']=True
            self.Args['delay corection coefficient']=self.delay_correction.value()
            cof=self.Args['delay corection coefficient']
            if not self.Args['frog_file'] == '':
                self.Results['T']=np.copy(self.Results['T_load'])*cof
                self.showFROGex()
        else:
            if not self.Args['frog_file'] == '':
                self.Results['T']=np.copy(self.Results['T_load'])
                self.showFROGex()

        self.Results['frog_in_processed?']=False        
       
    def delaycorrection_click(self):
        if self.checkdelaycorrection.isChecked():
            self.delay_correction.setEnabled(True)
        else:
            self.delay_correction.setEnabled(False)
        self.set_delaycorrection()
        self.Results['frog_in_processed?']=False
                
    def bkgsubtract_click(self):
        if self.check_bkgsubtract.isChecked():
            self.bkg.setEnabled(True)
            self.Args['substract_bkg']=True
        else:
            self.bkg.setEnabled(False)
            self.Args['substract_bkg']=False
        self.set_bkg()
    
    def set_bkg(self):
        if self.check_bkgsubtract.isChecked():
            self.Args['background']=self.bkg.value()
            self.remove_bkg()
        else:
            self.Results['frog_in']=self.Results['frog_in_0']
        self.showFROGex()
    
    def remove_bkg(self):
        M=self.Results['frog_in_0'].max()
        self.Results['frog_in']=self.Results['frog_in_0']-self.Args['background']*M
        ind=self.Results['frog_in']<0
        self.Results['frog_in'][ind]=0
        M=self.Results['frog_in'].max()
        self.Results['frog_in']=self.Results['frog_in']/M
    
    def multi_grid_click(self):
        if self.multi_grid.isChecked():
            self.check_parallel.setEnabled(True)
            self.Args['multi_grid']=True
        else:
            self.check_parallel.setEnabled(False)
            self.Args['multi_grid']=False
            
    def symmetry_click(self):
        self.Results['frog_in_processed?']=False
            
    def logscale_click(self):
        if self.check_logscale.isChecked():
            self.Args['logscale']=True
        else:
            self.Args['logscale']=False
        if (not self.Args['frog_file'] == '') and (len(self.Results['frog_in'])>1):
            self.showFROGex()
        if self.Results['Step'] > 0:
            self.showFROGsim()
            
    def set_size(self):
        self.Results['frog_in_processed?']=False
    
    def loadFROG(self):
        """loads a FROG trace"""
        #try:
        (self.Results['T_load'],self.Results['W_load'],self.Results['frog_load'])=PCGPA.load_frog(self.Args['frog_file'])
        self.Results['T']=self.Results['T_load']
        ind=self.Results['frog_load']<0
        self.Results['frog_load'][ind]=0
        M=self.Results['frog_load'].max()
        self.Results['frog_load']=self.Results['frog_load']/M
        if self.checktranspose.isChecked():
            self.Results['frog_in']=np.transpose(self.Results['frog_load'])
        else:
            self.Results['frog_in']=self.Results['frog_load']
            self.Results['frog_in_0']=self.Results['frog_in']
        
        if self.checkmaxwavelength.isChecked():
            ind=self.Results['W_load'] > 2*Pi*c/self.Args['MaxWavelength']*10**9*10**-15
            self.Results['W']=np.copy(self.Results['W_load'][ind])
            self.Results['frog_in']=self.Results['frog_in'][:,ind]
        else:
            self.Results['W']=np.copy(self.Results['W_load'])
                
#        except ER.SL_exception as error:
#            self.showerror(error)
#            self.file_SHG.clear()
#            self.Args['frog_file'] = ''
    
    def load_fund(self):
        """loads the fundumental spectrum"""
        self.error_message.setPlaceholderText('')#clear errors
        self.Args['fundspec_file']=QtWidgets.QFileDialog.getOpenFileName(self)[0]
        if not self.Args['fundspec_file'] == '':
            self.file_fun.clear()
            self.file_fun.insert(self.Args['fundspec_file'])
            try :
                Sp=imp_spec(self.Args['fundspec_file'])
                self.Results['Sfund_load']=Sp
            except ER.SL_exception as error:
                self.showerror(error)
            # W=Sp[:,0]
            # Field_W=Sp[:,1]
            # plt.plot(W,Field_W)
            # plt.show()
    
    def transpose_frog(self):
        if not self.Args['frog_file'] == '':
            if len(self.Results['frog_load'])==0:
                self.showerror(ER.SL_exception('no FROG loaded yet'))
            else:
                self.Results['frog_in']=np.transpose(self.Results['frog_in'])
            self.showFROGex()
    
    def showFROGex(self):
        X=self.Results['T']
        Y=self.Results['W']/2/Pi
        FROG=self.Results['frog_in']
        Ntick=7 #number of ticks
        
        #log scaling if selected
        if self.Args['logscale']:
            floor=10**-3
            FROG=np.log10(FROG+floor)
        
        View=self.FROG_exp
        S=View.size()
        scene = QtGui.QGraphicsScene(self)
        L=pg.GraphicsLayoutWidget()
        
        #main image
        view = pg.ViewBox()
        img = pg.ImageItem(FROG)
        img.setImage(lut=ImageColorMap('Wh_rainbow',512))
        view.addItem(img)
        
        #Y axis
        ax1=pg.AxisItem('left')
        ax1.linkToView(view)
        ax1.setLabel('frequency','PHz')
        Ny=[round((len(Y)-1)*i/(Ntick-1)) for i in range(Ntick)]
        Wy=np.copy(Y[Ny]).round(decimals=2)
        ticks1=[list(zip(Ny,Wy))]
        ax1.setTicks(ticks1)
        #X axis
        ax2=pg.AxisItem('bottom')
        ax2.linkToView(view)
        ax2.setLabel('delay','fs')
        Nx=[round((len(X)-1)*i/(Ntick-1)) for i in range(Ntick)]
        Tx=np.copy(X[Nx].round())
        ticks2=[list(zip(Nx,Tx))]
        ax2.setTicks(ticks2)
        
        L.addItem(ax1)
        L.addItem(view)
        L.nextRow()
        L.addItem(ax2, col=1)
        L.resize(S*0.99)
        scene.addWidget(L)
        View.setScene(scene)
        
    def showFROGsim(self):
        X=self.Results['T']
        Y=self.Results['W']/2/Pi
        FROG=self.Results['frog_out_best']
        Ntick=7 #number of ticks
        
        View=self.FROG_sim
        S=View.size()
        scene = QtGui.QGraphicsScene(self)
        L=pg.GraphicsLayoutWidget()
        
        #log scaling if selected
        if self.Args['logscale']:
            floor=10**-3
            FROG=np.log10(FROG+floor)
        
        #main image
        view = pg.ViewBox()
        img = pg.ImageItem(FROG)
        img.setImage(lut=ImageColorMap('Wh_rainbow',512))
        view.addItem(img)
        
        #Y axis
        ax1=pg.AxisItem('left')
        ax1.linkToView(view)
        ax1.setLabel('frequency','PHz')
        Ny=[round((len(Y)-1)*i/(Ntick-1)) for i in range(Ntick)]
        Wy=np.copy(Y[Ny]).round(decimals=2)
        ticks1=[list(zip(Ny,Wy))]
        ax1.setTicks(ticks1)
        #X axis
        ax2=pg.AxisItem('bottom')
        ax2.linkToView(view)
        ax2.setLabel('delay','fs')
        Nx=[round((len(X)-1)*i/(Ntick-1)) for i in range(Ntick)]
        Tx=np.copy(X[Nx].round())
        ticks2=[list(zip(Nx,Tx))]
        ax2.setTicks(ticks2)
        L.addItem(ax1)
        L.addItem(view)
        L.nextRow()
        L.addItem(ax2, col=1)
        L.resize(S*0.99)
        scene.addWidget(L)
        View.setScene(scene)
    
    def showerror(self,error):
        self.error_message.setPlaceholderText(error.Message)
        #print('open error', error.Message)     
     
    def readparam(self):
        """read parameters for the reconstruction"""
        self.Args['G_goal']=self.goal.value()
        self.Args['ret_iterations']=self.iternumber.value()
        self.Args['fix_fund_spec']=self.fix_fund_spec.isChecked()
        self.Args['use_fund_spec']=self.use_fund_spec.isChecked()
#        self.Args['delay correction']=self.checkdelaycorrection.isChecked()
#        if self.Args['delay correction']:
#            self.Args['delay corection coefficient']=self.delay_correction.value()
                            
        self.progressBar.setMaximum(self.Args['ret_iterations'])
        self.progressBar.setMinimum(0)
        
        self.Args['max size']=int(self.size.value())
        self.Args['parallel']=self.check_parallel.isChecked()
        self.Args['symmetrization']=self.check_symmetry.isChecked() 
                   
        
        if self.Args['frog_file'] == '':
            raise ER.SL_exception('no FROG file is loaded')
     
    def fstop(self):
        self.Stop=False
        #print('stop')
        self.error_message.setPlaceholderText('')
        
    def fstart(self):
        """initiate main retreive function"""
        self.error_message.setPlaceholderText('')#clear errors
        try:
            self.readparam()
        except ER.SL_exception as error:
            self.showerror(error)
        else:
        
            #preprocess
            if not self.Args['frog_file'][-3:]=='frg' and not self.Args['frog_file'][-3:]=='frog' and not self.Results['frog_in_processed?']:
                """resize frog"""
                self.set_delaycorrection()
                self.set_wavelength()
                (self.Results['T'],self.Results['W'],self.Results['frog_in_0'])=PCGPA.resize_frog(
                        self.Results['T'],self.Results['W'],self.Results['frog_in_0'],PCGPA.TBP_frog(
                                self.Results['T'],self.Results['W'],self.Results['frog_in_0'])*1.3*2,
                                self.Args['max size'])
                #*1.3 might be a bit too small for pulses with very large chirp, so be careful (and modify if necessary)
                self.remove_bkg()
                self.showFROGex()
            
            self.Results['Wf']=self.Results['W']-self.Results['W'][int(len(self.Results['W'])/2+1)]/2 #fundamental frequencies
            
            self.Results['frog_in_processed?']=True
            
            if self.Args['symmetrization']:
                #simmetrization of the exp frog trace
                frog=self.Results['frog_in']
                frogsim=(frog[1 : int(len(frog)/2)]+np.flip(frog[int(len(frog)/2)+1:],axis=0))/2
                self.Results['frog_in']=np.concatenate(([frog[0]],frogsim,
                            [frog[int(len(frog)/2)]],np.flip(frogsim,axis=0)),axis=0)
                self.Results['frog_in']=self.Results['frog_in']/np.max(self.Results['frog_in']) #normalization
                self.showFROGex()
            
            #initiate counters
            self.Results['Step']=0
            self.Stop=True
            self.Results['G']=1
            self.Results['G_best']=1
        
            if self.Args['use_fund_spec']:
                #use the loaded spectrum
                Sp=self.Results['Sfund_load']
                W=Sp[:,0]*10**-15
                Spec=Sp[:,1]
                #cut out the part of the spectrum with the data
                M=Spec.max()
                #find spectrum boarders on the 3**2 level
                N2=0
                for i in range(len(Spec)):
                    if Spec[i] > M/3: N2=i
                N1=len(Spec)-1
                for i in range(len(Spec)-1,-1,-1):
                    if Spec[i] > M/3: N1=i
                N1=int(np.max([0,N1-(N2-N1)/2]))
                N2=int(np.min([len(Spec),N2+(N2-N1)/2]))
                Spec1=Spec[N1:N2]
                W1=W[N1:N2]
                #interpolate the spectrum
                Ew=interpolate.PchipInterpolator(W1,Spec1)
                Wout=self.Results['Wf']
                Sout=np.zeros(Wout.shape)*1.
                indS=np.logical_and(Wout < np.ones(Wout.shape)*W1[-1],Wout > np.ones(Wout.shape)*W1[0])
                Sout[indS]=Ew(Wout[indS])                

                #zero negative values (in case they appear after interpolation)
                ind0=Sout < np.zeros(Sout.shape)
                Sout[ind0]=0
                self.Results['Sfund']=Sout**2
            else:
                #get fundamental spectrum from the FROG
                self.Results['Sfund']=PCGPA.spectrum_fromSHG(
                            self.Results['T'],self.Results['W'],self.Results['frog_in'])
#                plt.plot(self.Results['Sfund'])
#                plt.show()
            
            
            if self.Args['multi_grid']:
                #preprocess with the multi-grid approach
                self.preprocess()
            else:
                #start with random phase
                phase=np.random.random(len(self.Results['W']))*2*Pi*1
                Sf=self.Results['Sfund']
                pulse_w=np.sqrt(Sf)*np.exp(1j*phase) #in spectral domain
                pulse_t=ifftshift(ifft(ifftshift(pulse_w))) #convet to to time domain
                gate_t=np.copy(pulse_t)
                self.Results['pulse']=pulse_t
                #self.Results['pulse_w']=pulse_w
                self.Results['gate']=gate_t

            self.reconstruction()
                    
            self.showresults()
            self.progressBar.setValue(self.Args['ret_iterations'])
            
    def preprocess(self):
        """preprocess with multi-grid algorithm"""
        Max_population=16;
        MNStep=5;
        if self.Args['parallel']:
            Ncpu=cpu_count()
            if Ncpu > 1:
                Npr=Ncpu-1 #number of parallel processes in the Pool
            else:
                Npr=1
            p=Pool(Npr)
#            print(1)
            self.Results['pulse']=PCGPA.parallel_IG(p,self.Results['T'],self.Results['W'],
                        self.Results['frog_in'], self.Results['Sfund'],
                        keep_fundspec=self.Args['fix_fund_spec'],max_population=Max_population,NStep=MNStep,
                        parallel=True)
            self.Results['gate']=np.copy(self.Results['pulse'])
            p.close()
            p.join()
        else:
            self.Results['pulse']=PCGPA.parallel_IG(None,self.Results['T'],self.Results['W'],
                        self.Results['frog_in'], self.Results['Sfund'],
                        keep_fundspec=self.Args['fix_fund_spec'],max_population=Max_population,NStep=MNStep,
                        parallel=False)
    
    def fcontinue(self):
        """contimue the main retreive function (after stopping it)"""
        self.Stop=True
        self.readparam()
        self.reconstruction()
                    
        self.showresults()
        self.progressBar.setValue(self.Args['ret_iterations'])
        
    def reconstruction(self):
        """main reconstruction function"""
        #initiate time counters
        timestart=time.time()
        timershow=0 #for measuring the time interval from the last results indication
        Showstep=0.5 #time interval between showing results
        timercheck=0 #for measuring the time interval from the last inputs check
        Checkstep=0.25 #time interval between checking inputs (including stop)
        
        while self.Stop and self.Results['Step']<self.Args['ret_iterations'] and self.Results['G']>self.Args['G_goal']:
            (self.Results['pulse'],self.Results['gate'],
             self.Results['G'],self.Results['frog_out'])=PCGPA.PCGPA_step(
                     self.Results['pulse'],self.Results['gate'],self.Results['frog_in'])
            
            if self.Args['fix_fund_spec']:
                """fixing the fundumental spectrum"""
                pulse_w=np.sqrt(self.Results['Sfund'])*np.exp(1j*np.angle(fftshift(fft(fftshift(self.Results['pulse'])))))
                pulse_t=ifftshift(ifft(ifftshift(pulse_w)))
                gate_t=np.copy(pulse_t)
                self.Results['pulse']=pulse_t
                #self.Results['pulse_w']=pulse_w
                self.Results['gate']=gate_t
                #recalculate frog_sim and G for the corrected pulse
                (self.Results['G'],self.Results['frog_out'])=PCGPA.PCGPA_G(
                     self.Results['pulse'],self.Results['gate'],self.Results['frog_in'])
            
            if self.Results['Step']==0:
                self.Results['G_best']=self.Results['G']
                self.Results['pulse_best']=self.Results['pulse']
                self.Results['gate_best']=self.Results['gate']
                self.Results['frog_out_best']=self.Results['frog_out']
                self.Results['pulse_w_best']=fftshift(fft(fftshift(self.Results['pulse_best'])))
            
            #save the best reconstruction
            if self.Results['G'] < self.Results['G_best'] and self.Results['Step'] > 0:
                (self.Results['pulse'], self.Results['gate'])= PCGPA.shift2zerodelay(self.Results['pulse'],self.Results['gate'])
                self.Results['G_best']=self.Results['G']
                self.Results['pulse_best']=self.Results['pulse']
                self.Results['gate_best']=self.Results['gate']
                self.Results['frog_out_best']=self.Results['frog_out']
                #shift to 0 delay
                                                          
            self.Results['Time']=time.time()-timestart
            if (self.Results['Time']-timershow) > Showstep:
                timershow=self.Results['Time']
                
            #                    plt.plot(self.Results['pulse_best'])
            #                    plt.show()
                self.Results['pulse_w_best']=fftshift(fft(fftshift(self.Results['pulse_best'])))
                self.showresults()
            if (self.Results['Time']-timercheck)>Checkstep:
                timercheck=self.Results['Time']
                self.app.processEvents()
                
            if self.Results['Step']==0:
                self.showresults()
            
            #print(self.Results['Step'])
            self.Results['Step']+=1
    
    def showresults(self):
        self.showFROGsim()
        self.show_spectrum()
        self.show_time()
        self.show_progress()
        self.show_width()
        
    def show_width(self):
        self.duration.display(width(self.Results['T'],np.abs(self.Results['pulse_best'])**2))
        self.Tstep.display(self.Results['T'][1]-self.Results['T'][0])
        self.Swidth.display(width(2*Pi*c/self.Results['Wf']/10**15*10**9,np.abs(self.Results['pulse_w_best'])**2))
        NW0=int(len(self.Results['Wf'])/2)
        self.Sstep.display(np.abs(2*Pi*c/self.Results['Wf'][NW0]/10**15*10**9
                                  -2*Pi*c/self.Results['Wf'][NW0+1]/10**15*10**9))
        pulse_TL=np.abs(ifftshift(ifft(ifftshift(np.abs(self.Results['pulse_w_best'])))))**2
        self.duration_tl.display(width(self.Results['T'],pulse_TL))
    
    def show_spectrum(self):
        View=self.SpectrumView
        scene = QtGui.QGraphicsScene(self)
        pw=pg.PlotWidget(labels={'left': 'Intensity a.u.', 'bottom': 'frequency PHz'})
        pw.resize(View.size()*0.99)
        p1=pw.plotItem
        M1=(np.abs(self.Results['pulse_w_best'])**2).max()
        p1.plot(self.Results['Wf']/2/Pi,np.abs(self.Results['pulse_w_best'])**2/M1, 
                pen=pg.mkPen((0,0,255), width=2.5))
        p2 = pg.ViewBox()
        p1.showAxis('right')
        p1.scene().addItem(p2)
        p1.getAxis('right').linkToView(p2)
        p2.setXLink(p1)
        p1.getAxis('right').setLabel('phase rad', color='#00ff00')
        
        def updateViews():
            p2.setGeometry(p1.vb.sceneBoundingRect())
            p2.linkedViewChanged(p1.vb, p2.XAxis)
           
        updateViews()
        p1.vb.sigResized.connect(updateViews)
            
        p2.addItem(pg.PlotCurveItem(self.Results['Wf']/2/Pi,np.angle(self.Results['pulse_w_best']), 
                                    pen=pg.mkPen((0,255,0), width=2.5)))
        
        scene.addWidget(pw)
        View.setScene(scene)
        
    def show_time(self):
        View=self.TimeView
        scene = QtGui.QGraphicsScene(self)
        plot=pg.PlotWidget(labels={'left': 'Intensity a.u.', 'bottom': 'time fs'})
        M1=(np.abs(self.Results['pulse_best'])**2).max()
        plot.plot(self.Results['T'],np.abs(self.Results['pulse_best'])**2/M1, pen=pg.mkPen((0,0,255), width=2.5))
        plot.resize(View.size()*0.99)
        scene.addWidget(plot)
        View.setScene(scene)
    
    def show_progress(self):
        self.step.display(self.Results['Step'])
        self.error.display(self.Results['G_best'])
        self.timer.display(self.Results['Time'])
        self.progressBar.setValue(self.Results['Step'])
        
    
    def savefrog(self,file):
        if type(file) == bool:
            file=QtWidgets.QFileDialog.getSaveFileName(self)[0]
        
        file+='.frog'
            
        X=self.Results['T']
        Y=self.Results['W']/2/Pi
        FROG=self.Results['frog_out_best']
        np.savetxt(file,
                   np.concatenate(([X],[Y],FROG),axis=0),
                                   delimiter='\t',comments='')
        
    def savefrog_in(self,file):
        if type(file) == bool:
            file=QtWidgets.QFileDialog.getSaveFileName(self)[0]
        file+='_in.frog'
        X=self.Results['T']
        Y=self.Results['W']/2/Pi
        FROG=self.Results['frog_in']
        np.savetxt(file,
                   np.concatenate(([X],[Y],FROG),axis=0),
                                   delimiter='\t',comments='')
    
    def savetime(self,file):
        if type(file) == bool:
            file=QtWidgets.QFileDialog.getSaveFileName(self)[0]
        pulse_TL=np.abs(ifftshift(ifft(ifftshift(np.abs(self.Results['pulse_w_best'])))))
        MTL=(np.abs(pulse_TL)**2).max()
        file_I=file+'_intensity.time'
        file_Ph=file+'_phase.time'
        file_Itl=file+'_intensityTL.time'
        M1=(np.abs(self.Results['pulse_best'])**2).max()
        np.savetxt(file_I,
                   np.concatenate((self.Results['T'].reshape((-1,1)),
                                            np.abs(self.Results['pulse_best'].reshape((-1,1)))**2/M1),axis=1),
                                   header='time fs \t Intensity a.u.',delimiter='\t',comments='')
        np.savetxt(file_Ph,
                   np.concatenate((self.Results['T'].reshape((-1,1)),
                                            np.angle(self.Results['pulse_best'].reshape((-1,1)))),axis=1),
                                   header='time fs \t phase rad',delimiter='\t',comments='')
        # transform limited
        np.savetxt(file_Itl,
                  np.concatenate((self.Results['T'].reshape((-1,1)),
                                  np.abs(pulse_TL.reshape((-1,1)))**2/MTL),axis=1),
                                  header='time fs \t Intensity a.u.',delimiter='\t',comments='')
                  
    
    def savespectrum(self,file):
        if type(file) == bool:
            file=QtWidgets.QFileDialog.getSaveFileName(self)[0]
        file_I=file+'_intensity.spectrum'
        file_Ph=file+'_phase.spectrum'
        M1=(np.abs(self.Results['pulse_w_best'])**2).max()
        np.savetxt(file_I,
                   np.concatenate((self.Results['Wf'].reshape((-1,1))/2/Pi,
                                            np.abs(self.Results['pulse_w_best'].reshape((-1,1)))**2/M1,
                                            2*Pi*c/self.Results['Wf'].reshape((-1,1))/10**15*10**9),axis=1),
                                   header='frequency PHz \t Intensity a.u. \t wavelength nm',
                                   delimiter='\t',comments='')
        np.savetxt(file_Ph,
                   np.concatenate((self.Results['Wf'].reshape((-1,1))/2/Pi,
                                            remove_phase_jumps(np.angle(self.Results['pulse_w_best'])).reshape((-1,1)),
                                            2*Pi*c/self.Results['Wf'].reshape((-1,1))/10**15*10**9),axis=1),
                                   header='frequency PHz \t phase rad \t wavelength nm',
                                   delimiter='\t',comments='')
        
    def saveparam(self,file):
        if type(file) == bool:
            file=QtWidgets.QFileDialog.getSaveFileName(self)[0]
        file+='_results.txt'
        DT=width(self.Results['T'],np.abs(self.Results['pulse_best'])**2)
        dt=self.Results['T'][1]-self.Results['T'][0]
        DW=width(2*Pi*c/self.Results['Wf']/10**15*10**9,np.abs(self.Results['pulse_w_best'])**2)
        NW0=int(len(self.Results['Wf'])/2)
        dw=np.abs(2*Pi*c/self.Results['Wf'][NW0]/10**15*10**9
                                  -2*Pi*c/self.Results['Wf'][NW0+1]/10**15*10**9)
        pulse_TL=np.abs(ifftshift(ifft(ifftshift(np.abs(self.Results['pulse_w_best'])))))**2
        DTl=width(self.Results['T'],pulse_TL)
        np.savetxt(file,np.array([[self.Results['G_best'],DT,DTl,dt,DW,dw]]),
                   header='G error \t pulse duration fs \t TL pulse duration fs \t dt fs \t Spectrum width nm \t spectrum step nm',
                   delimiter='\t',comments='')
        
    def saveall(self):
        file1=QtWidgets.QFileDialog.getSaveFileName(self)
        self.savetime(file=file1[0])
        self.savefrog(file=file1[0])
        self.savefrog_in(file=file1[0])
        self.savespectrum(file=file1[0])
        self.saveparam(file=file1[0])
        #add asking about rewriting
        
    def closeEvent(self, event):
        self.fstop()
        event.accept()
    


def main():
    pg.setConfigOption('background', 'w')
    window = FROG_class()
    window.show()
    window.app.exec_()

if __name__ == '__main__':
    main()
