"""class pulse

to do
faster version of the fourier the custom transform through matrixes

"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)
SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)

from load_files.load_folder import imp_spec, imp_phase
import numpy as np
from myconstants import c, pulse_dt2dw, lam2w
Pi=np.pi
from scipy import interpolate, integrate
from scipy.integrate import odeint
from scipy.fftpack import fft, ifft, fftshift, ifftshift
#import matplotlib.pyplot as plt
import classes.error_class as ER
from dispersion_data.dispersion import n, dn_delay, n_range, crossing_interpol
import matplotlib.pyplot as plt
from tkinter import filedialog
from color_maps.color_maps import plt_cmap
from absorption_data.absorption import transmission
from scipy import ndimage

class pulse:
    
    def __init__(self,W=[],spectrum_I=[],specrum_phase=[]):
        """defines the pulse using spectral intensity and phase
        W=2*Pi*c/lambda
        """
        if len(W) > 1:
            self.W=W
            self.Field_W=np.sqrt(spectrum_I)*np.exp(1j*specrum_phase)
            self.W0=np.sum(W*spectrum_I)/np.sum(spectrum_I)
    
        self.Field_W=np.array([],dtype=np.complex64)
        self.W=[]
        self.W0=0 #central frequency (defined as a center of mass)
        self.Field_T=np.array([],dtype=np.complex64)
        self.T=[]
        self.T0=0 #time offset
        self.GD=[]
        self.GDD=[]
        
    def init_pulse(self,Ttfl,lam0,shape='Gaussan',Nw=100,Nt=50):
        """init pulse; Tftl is the transform limited FWHM pulse duration in fs;
        lam 0 is the central wavelength in um
        N is the number of pixels"""
        Te2=Ttfl*10**-15*(2/np.log(2))**0.5 #duration e^2
        Dw=pulse_dt2dw(Ttfl*10**-15,lam0*1000)*2*Pi
        Dwe2=Dw*(2/np.log(2))**0.5
        w0=lam2w(lam0)
        dw0=Dwe2/Nw #requested spectral resolution
        DT=2*Pi/dw0 #coresponding time window
        dt0=Te2/Nt
        
        Nbin0=2*Pi/dt0/dw0
        if np.log2(Nbin0)%1:
            Nbin=int(2**(np.floor(np.log2(Nbin0))+1)) #increase the number of points to the nearest 2**? value
        else:
            Nbin=Nbin0
            
        print('Nbin ',Nbin)
        dt=2*Pi/dw0/(Nbin-1)
        
        W=np.array([w0+dw0*(i-Nbin/2) for i in range(Nbin)])
        T=np.array([0+dt*(i-Nbin/2) for i in range(Nbin)])
        
        def I_gaus(w,w0,dw):
            	return np.exp(-4*np.log(2)*(w-w0)**2/dw**2)
            
        Iw=I_gaus(W,w0,Dw)
        Ew=Iw**0.5
        
        self.T=T
        self.W=W
        self.Field_W=Ew
        self.Wrange=np.array((w0-Dwe2,w0+Dwe2)) #99% of energy
    
    def init_spectrum(self,w0,dW,W,Stype='Gauss'):
        """initiares spectrum of type Stype
        w0 central frequency, dW width, W frequency vector"""
        if Stype == 'Gauss':
            self.W=W
            self.W0=w0
            self.Field_W=np.exp(-2*np.log(2)*(W-w0)**2/dW**2)
            self.GD=np.zeros(len(self.W))*1.
            self.GDD=np.zeros(len(self.W))*1.
        elif Stype == 'FlatTop':
            self.W=W
            self.W0=w0
            self.Field_W=np.logical_and(W<w0+dW/2,W>w0-dW/2)*1.
            self.GD=np.zeros(len(self.W))*1.
            self.GDD=np.zeros(len(self.W))*1.
        else:
            print('unknown spectrum type')
    
    def loadspectralintensity(self,file=None,xcal='nm',axis=(0,1),
                              correctInt=False,IntCorFile=None,lam_max=None,lam_range=[],
                              lam_bkg=[],GF_sigma=0):
        """load spectrum from file
        xcal - x calibration nm, or THz
        correctInt if apply intensity responce correction
        IntCorFile is the file with correction function
        lam_max is the max wavelentgh to take into account in nm
        lam_range is the range to limit the import data in um
        GF_sigma is the sigma parameter for Gaussian smoothing
        lam_bkg is the spectral range to take as a background in nm
        """
        if file == None:
            file=filedialog.askopenfilename()
        if file == '':
            print('wrong file path')
            #!!!add raise error
        else:
            Sp=imp_spec(file,xcal=xcal,axis=axis,lam_bkg=lam_bkg,GF_sigma=GF_sigma)
            if not lam_max==None:
                ind=Sp[:,0]>2*Pi*c/lam_max/10**-9
                Sp=Sp[ind]
            if len(lam_range)>0:
                ind=np.logical_and(Sp[:,0] >= 2*Pi*c/lam_range[1]/10**-6,
                                   Sp[:,0] <= 2*Pi*c/lam_range[0]/10**-6)
                Sp=Sp[ind]
            self.I=Sp[:,1].copy()**2
            self.W=Sp[:,0]
            
            self.GD=np.zeros(len(self.W))*1.
            self.GDD=np.zeros(len(self.W))*1.
            
            if correctInt:
                if IntCorFile== None:
                    IntCorFile=filedialog.askopenfilename()
                if IntCorFile == '':
                    print('wrong file path')
                    #!!!add raise error
                else:
                    calib=np.loadtxt(IntCorFile)
                    if len(Sp)==len(calib):
                        Cal=calib[-1::-1,1]**0.5
                        S=Sp[:,1]
                        indc=(1/Cal)>1.6
                        # print(1/Cal)
                        inds=(S/S.max())<(1/100.)
                        ind=np.logical_or(indc,inds)
                        S[ind]=0
                        self.Field_W=S/Cal
                    else:
                        print('different length of spectrum and calibration files')
                        self.Field_W=Sp[:,1]
            else:
                self.Field_W=Sp[:,1]
            
            self.W0=self.Wcentral(Cutlevel=0.04)
            
    def load_and_average_spectra(self,files):
        S=[]
        for f in files:
            self.loadspectralintensity(f)
            S.append(self.I.copy())
        S1=np.sum(S,axis=0)**0.5
        self.Field_W=S1
                
    def loadspectralbkg(self,file=None,xcal='nm',axis=(0,1),
                              correctInt=False,IntCorFile=None,lam_max=None,lam_range=[],
                              lam_bkg=[],GF_sigma=0,RenormCoef=1,LamShift=0):
        if file == None:
            file=filedialog.askopenfilename()
        if file == '':
            print('wrong file path')
            #!!!add raise error
        else:
            Sp=imp_spec(file,xcal=xcal,axis=axis,lam_bkg=lam_bkg,GF_sigma=GF_sigma)
            if not lam_max==None:
                ind=Sp[:,0]>2*Pi*c/lam_max/10**-9
                Sp=Sp[ind]
            if len(lam_range)>0:
                ind=np.logical_and(Sp[:,0] >= 2*Pi*c/lam_range[1]/10**-6,
                                   Sp[:,0] <= 2*Pi*c/lam_range[0]/10**-6)
                Sp=Sp[ind]
                W=Sp[:,0]
            
            if correctInt:
                if IntCorFile== None:
                    IntCorFile=filedialog.askopenfilename()
                if IntCorFile == '':
                    print('wrong file path')
                    #!!!add raise error
                else:
                    calib=np.loadtxt(IntCorFile)
                    if len(Sp)==len(calib):
                        Cal=calib[-1::-1,1]**0.5
                        S=Sp[:,1]
                        indc=(1/Cal)>1.6
                        # print(1/Cal)
                        inds=(S/S.max())<(1/100.)
                        ind=np.logical_or(indc,inds)
                        S[ind]=0
                        Field_W=S/Cal
                    else:
                        print('different length of spectrum and calibration files')
                        Field_W=Sp[:,1]
            else:
                Field_W=Sp[:,1]
            
            self.Field_W-=Field_W
            ind0=self.Field_W < 0
            self.Field_W[ind0]=0
            
            #spectrum correction
            if RenormCoef != 1:
                self.Field_W=self.Field_W**RenormCoef
            if LamShift != 0:
                Wshift=2*Pi*c/(2*Pi*c/self.W0 + LamShift*10**-6) - self.W0
                self.W+=Wshift
            
            #new central wavelength
            self.W0=self.Wcentral(Cutlevel=0.04)
        
        
    def Wcentral(self,W=None,S=None,Type='center of mass',Cutlevel=0):
        """retuns the central frequency of a spectrum
        Tyoe: center of mass; peak
        Cutlevel is the level below which the data is ignored (for ecample to avoid noise contribution)"""
        if W == None:
            #use data stored in the class
            W=self.W
            S=np.abs(self.Field_W)**2
        if Type == 'center of mass':
            S0=S/S.max()
            ind=S0>Cutlevel
            W0=np.sum(W[ind]*S0[ind])/np.sum(S0[ind])
            return W0
        elif Type == 'peak':
            N0=S.argmax()
            return W[N0]
        
    def loadspectralphase(self,file=None,xcal='nm',axis=(0,1),DataType='phase',Nmirrors=1,
                          lam_max=None):
        """DataType can be: 'phase', 'GD', 'GDD'  (GD and GDD are assumed to be in fs)
        Nmirrors is the number of mirrors (otherwise a multiplication factor to the phase to be added)"""
        
        if file == None:
            file=filedialog.askopenfilename()
        if file == '':
            print('wrong file path')
            #!!!add raise error
        else:
            Sp=imp_phase(file,xcal=xcal,axis=axis)
            if not lam_max==None:
                ind=Sp[:,0]>2*Pi*c/lam_max
                Sp=Sp[ind]
            if len(self.Field_W)==0:
                self.Field_W=np.ones(len(Sp))*1j
            if DataType == 'GDD':
                plt.plot(Sp[:,0],Sp[:,1])
                plt.ylim([-100,100])
                plt.show()
                self.add_GDD(Sp[:,1],Sp[:,0],Nmirrors=Nmirrors)
                # W=Sp[:,0]
                # N0=np.sum(W<self.W0) #index of the W0 - central frequency
                # # print(N0)
                # # GDD=interpolate.PchipInterpolator(Sp[:,0],Sp[:,1]*10**-30) #!!!assuming that it is in fs**2
                # # GD=odeint(GDD, 0, W/10**15)
                # # plt.plot(W,GD)
                # # plt.ylim([-1,1])
                # # plt.show()
                # GD=FlistInteg(W,Sp[:,1]*10**-30)#!!!assuming that it is in fs**2
                # GD-=GD[N0]
                # plt.plot(W,GD*10**15)
                # plt.ylim([-2,2])
                # plt.show()
                # Ph=FlistInteg(W,GD)
                # Ph-=Ph[N0]
                # # print(Ph.shape,Ph)
                # IntPh=interpolate.PchipInterpolator(W,Ph)
                # Phase=IntPh(self.W)*Nmirrors
                # self.Field_W=self.Field_W*np.exp(1j*Phase)
                # plt.plot(W,Ph)
                # plt.ylim([-1,1])
                # plt.show()
            elif DataType == 'GD':
                plt.plot(Sp[:,0],Sp[:,1])
                plt.ylim([-100,100])
                plt.show()
                self.add_GD(Sp[:,1],Sp[:,0],Nmirrors=Nmirrors)
            else:
                self.add_phase(Sp[:,1],Sp[:,0],Nmirrors=Nmirrors,evenlyspaced=False)
                # IntPh=interpolate.PchipInterpolator(Sp[:,0],Sp[:,1])
                # Phase=IntPh(self.W)*Nmirrors
                # self.Field_W=self.Field_W*np.exp(1j*Phase)
                
    def convert2idler(self,lam_pump=2.4):
        """convert spectrum to idler, assuming that the spectrum is signal. 
        pump is assumed to be monochromatic with lam_pump wavelength in um
        the spectral phase is assumed to be flat
        """
        Wp=2*Pi*c/lam_pump/10**-6
        Xs=self.W
        Ys=np.abs(self.Field_W)**2
        Xi=Wp-Xs
        Yi=Ys*Xi
        Yi/=Yi.max()
        
        self.W=Xi[::-1]
        self.Field_W=Yi[::-1]**0.5
        self.W0=self.Wcentral(Cutlevel=0.04)
                
    def add_GD(self,GD,W,isfs=True,Nmirrors=1,OfRangeZero=True):
        """add GD to phase, GD and GDD"""
        if isfs:
            GD*=10**-15
        GD*=Nmirrors
        N0=np.sum(W<self.W0) #index of the W0 - central frequency
        GD-=GD[N0]
        IntGD=interpolate.PchipInterpolator(W,GD)
        GD1=IntGD(self.W)
        GDD=FlistDeff(W,GD,equallyspaced=False)
        plt.plot(W,GDD*10**30)
        plt.ylim(-100,100)
        plt.show()
        IntGDD=interpolate.PchipInterpolator(W,GDD)
        GDD1=IntGDD(self.W)
        Ph=FlistInteg(W,GD,equallyspaced=False)
        Ph-=Ph[N0]
        IntPh=interpolate.PchipInterpolator(W,Ph)
        Phase=IntPh(self.W)
        if OfRangeZero:
            ind=np.logical_not(np.logical_and(self.W>=W[0],self.W<=W[-1]))
            # print(W)
            # print(ind)
            GDD1[ind]=0
            GD1[ind]=0
            Phase[ind]=0
        self.Field_W=self.Field_W*np.exp(1j*Phase)
        self.GD+=GD1
        self.GDD+=GDD1
        
    def add_GDD(self,GDD,W,isfs=True,Nmirrors=1,OfRangeZero=True):
        """add GDD to phase, GD and GDD
        GDD and W are vectors"""
        if isfs:
            GDD*=10**-30
        GDD*=Nmirrors
        IntGDD=interpolate.PchipInterpolator(W,GDD)
        N0=np.sum(W<self.W0) #index of the W0 - central frequency
        GD=FlistInteg(W,GDD,equallyspaced=False)
        GD-=GD[N0]
        # plt.plot(W,GD*10**15)
        # plt.ylim([-2*Nmirrors,2*Nmirrors])
        # plt.show()
        IntGD=interpolate.PchipInterpolator(W,GD)
        Ph=FlistInteg(W,GD,equallyspaced=False)
        Ph-=Ph[N0]
        IntPh=interpolate.PchipInterpolator(W,Ph)
        Phase=IntPh(self.W)
        GDD1=IntGDD(self.W)
        GD1=IntGD(self.W)
        if OfRangeZero:
            ind=np.logical_not(np.logical_and(self.W>=W[0],self.W<=W[-1]))
            # print(W)
            # print(ind)
            GDD1[ind]=0
            GD1[ind]=0
            Phase[ind]=0
        self.Field_W=self.Field_W*np.exp(1j*Phase)
        self.GD+=GD1
        self.GDD+=GDD1
        
    def add_phase(self,phase,W,Nmirrors=1,OfRangeZero=True,evenlyspaced=True):
        """adds phase and corresponding GD and GDD"""
        N0=np.sum(W<self.W0) #index of the W0 - central frequency
        phase-=phase[N0]
        plt.plot(W,phase)
        plt.show()
        phase*=Nmirrors
        IntPh=interpolate.PchipInterpolator(W,phase)
        Phase=IntPh(self.W)
        GD=FlistDeff(self.W,Phase,equallyspaced=evenlyspaced)
        GD-=GD[N0]
        plt.plot(W,GD*10**15)
        plt.show()
        IntGD=interpolate.PchipInterpolator(W,GD)
        GDD=FlistDeff(W,GD,equallyspaced=False)
        IntGDD=interpolate.PchipInterpolator(W,GDD)
        GDD1=IntGDD(self.W)
        GD1=IntGD(self.W)
        if OfRangeZero:
            ind=np.logical_not(np.logical_and(self.W>=W[0],self.W<=W[-1]))
            # print(W)
            # print(ind)
            GDD1[ind]=0
            GD1[ind]=0
            Phase[ind]=0
        self.Field_W=self.Field_W*np.exp(1j*Phase)
        self.GD+=GD1
        self.GDD+=GDD1
    
    def remove_lin_phase(self,RemovePhaseJumps=False):
        w0=self.W0
        N0=np.sum(self.W<self.W0) #index of the W0 - central frequency
        phaseIn=np.angle(self.Field_W)
        phaseIn=remove_phase_jumps(phaseIn)
        phaseIn-=phaseIn[N0]
        GD=FlistDeff(self.W,phaseIn,equallyspaced=True)
        phaseOut=phaseIn-GD[N0]*(self.W-w0)
        # print(GD[N0],self.W[N0],2*Pi*c/self.W[N0])
        
        # plt.figure(7)
        # plt.clf()
        # plt.plot(phaseIn)
        # plt.plot(phaseOut)
        
        # if RemovePhaseJumps:
        #     phaseOut=remove_phase_jumps(phaseOut)
        #     phaseOut-=phaseOut[N0]
        
        self.Field_W=np.abs(self.Field_W)*np.exp(1j*phaseOut)
        self.phaseW_cor=phaseOut
        return phaseOut

    def def_spectralphase(self,W,Ph):
        IntPh=interpolate.PchipInterpolator(W,Ph)
        Ph=IntPh(self.W)
        self.Field_W=np.real(self.Field_W)*np.exp(1j*Ph)
        
    def add_dispersion_orders(self,lam0=None,k2=0,k3=0,k4=0,AddGD=True):
        """add dispersion k2 in fs^2, k3 in fs^3, k4 in fs^4
        lam0 in um is the cental wavelength (None is using the previously defined default value)"""
        if lam0 == None:
            w0=self.W0
        else:
            w0=2*Pi*c/lam0/10**6
        lam0=2*Pi*c/w0
        phase=k2*10**-30*(self.W-w0)**2/2+k3*10**-45*(self.W-w0)**3/6+k4*10**-60*(self.W-w0)**4/24
        self.Field_W=self.Field_W*np.exp(1j*phase)
        if AddGD:
            GD1=FlistDeff(self.W,phase)
            GDD1=FlistDeff(self.W,GD1)
            if len(self.GD)>1:
                self.GD+=GD1
            if len(self.GDD)>1:
                self.GDD+=GDD1
        
    def add_materialdisp(self,material,Length,use_spec_lam_range=False,AddGD=True):
        """adds spectral phase corresponding the the material of length Lenghth in m"""
        Lam=2*Pi*c/self.W
        w0=self.W0
        lam0=2*Pi*c/w0
        if np.array(n(1*10**6,material)).shape == ():
            phase=np.array([2*Pi/lam*Length*n(lam*10**6,material) for lam in Lam])
            GD=dn_delay(lam0*10**9,material,1,Length*10**3)*10**-15
        else:
            #!!! add proper handling of birefringent materials
            phase=np.array([2*Pi/lam*Length*n(lam*10**6,material)[0] for lam in Lam])
            GD=dn_delay(lam0*10**9,material,1,Length*10**3)[0]*10**-15
        phase-=GD*(self.W-w0) #remove group delay to keep the pulse at 0 delay
        # phase-=2*Pi/lam*Length*n(lam0*10**6,material) #removeabsolute phase at lam0
        # print(GD)
        ind= np.isnan(phase)
        phase[ind]=0 #remove undefined values
        if use_spec_lam_range:
            Lam_range=n_range(material)/10**6 # the wavelength range specified for n to be accurate
            ind_lam=np.logical_or(Lam<Lam_range[0],Lam>Lam_range[1])
            # print(ind_lam,Lam*10**6)
            self.Field_W[ind_lam]=0
            phase[ind_lam]=0
        self.Field_W[ind]=0
        # self.add_phase(phase,self.W)
        self.Field_W=self.Field_W*np.exp(1j*phase)
        # print(Lam)
        if AddGD:
            GD1=np.vectorize(dn_delay)(Lam*10**9,material,1,Length*10**3)*10**-15
            GD1-=GD
            self.GD+=GD1
        # GDD1=np.vectorize(dn_delay)(Lam*10**9,material,2,Length*10**3)*10**-30
        # self.GD+=GDD1
        # plt.plot(self.W,phase)
        # plt.show()
        # plt.plot(self.W,np.abs(self.Field_W)**2)
        # plt.show()
        # print(np.abs(self.Field_W)**2)
    
    def add_absorption(self,material,thickness,smooth=False,Gsmooth=4,RenormCoef=1):
        """adjust spectrom intensity according to transmission
        thickness in mm"""
        lam=2*Pi*c/self.W*10**6
        Tr=np.array([transmission(l,material,thickness) for l in lam])
        if smooth:
            Tr=ndimage.gaussian_filter1d(Tr,Gsmooth)
        
        self.Field_W*=Tr**0.5
        self.Field_W=np.abs(self.Field_W)**(RenormCoef/2)*np.exp(1j*np.angle(self.Field_W))
        
    def zero_spectalphase(self):
        """zero the spectral phase"""
        self.Field_W=np.abs(self.Field_W)
        
    def def_spectrum(self,W,E):
        """loads the specified spectral field"""
        self.Field_W=E
        self.W=W
        self.T=[]
        self.Field_T=[]
        
        #def W0
        Int=np.abs(E)**2
        self.W0=np.sum(W*Int)/np.sum(Int)
        
    def settimewindow(self,twindow,tstep,correct2power2=True):
        """sets the desired time window"""
        T1=twindow[0]
        T2=twindow[1]
        Nt0=int(np.round((T2-T1)/tstep+1)) #specified number of points
        Nt=Nt0 #modified number of points which need to be a power of 2 for the best performance
        if correct2power2:
            if np.log2(Nt0)%1:
                Nt=int(2**(np.floor(np.log2(Nt0))+1)) #increase the number of points to the nearest 2**? value
        self.T=np.linspace(T1,T2,Nt) #time points vector
        self.T0=self.T[int(Nt/2)] #effective 0 time
        
    def spectrum2time(self,set_T=False,timewindow=[],tstep=0,correct2power2=True,
                      slow_custom=False):
        """calculating temporal profile from the given spectrum"""
        
        if slow_custom:
            """slow but simple and reliable version when just temporal structure
            is wanted (not for iterative algoritms)"""
            T=np.linspace(timewindow[0],timewindow[1],round((timewindow[1]-timewindow[0])/tstep)-1) #time vector
            self.T=T
            self.Field_T=fourier_fixedT(self.Field_W,T,self.W)
            
        else:
            if set_T:
                self.settimewindow(timewindow,tstep,correct2power2)
                dw=2*Pi/(self.T[-1]-self.T[0])
                Nw=len(self.T)
                W=np.linspace(-Nw/2,Nw/2-1,Nw)*dw+self.W0
                Ew=interpolate.PchipInterpolator(self.W,self.Field_W)
                Field_W=np.zeros(Nw)*1j
                indS=np.logical_and(np.ones(Nw)*W[0] <= W , W <= np.ones(Nw)*W[-1])
                #calculate the new interpolated E in frequency domain; with 0 values outside the interpolation range
                Field_W[indS]=Ew(W[indS])
                
            else:
                Nw0=len(self.W)
                if np.log2(Nw0)%1:
                    Nw=int(2**(np.floor(np.log2(Nw0))+1)) #increase the number of points to the nearest 2**? value
                else:
                    Nw=Nw0
                #interpolate spectrum to even ferquency spacing
                Ew=interpolate.PchipInterpolator(self.W,self.Field_W)
                W=np.linspace(self.W[0],self.W[-1],Nw)
                Field_W=np.zeros(Nw)*1j
                indS=np.logical_and(np.ones(Nw)*W[0] <= W , W <= np.ones(Nw)*W[-1])
                #calculate the new interpolated E in frequency domain; with 0 values outside the interpolation range
                Field_W[indS]=Ew(W[indS])
                #define corresponding time vector
                Nt=Nw
                dt=2*Pi/(W[-1]-W[0])
                T=np.linspace(-Nt/2,Nt/2-1,Nt)*dt
                self.T=T
                self.T0=0
                
    #        W0=self.W0
    #        T0=self.T0
            self.Field_T=ifftshift(ifft(Field_W))*np.exp(-1j*W[0]*self.T)
        
    def pulseduration(self,method='FWHM',transform_limited=False):
        if transform_limited:
            #for the transform limited option
            Nw0=len(self.W)
            if np.log2(Nw0)%1:
                Nw=int(2**(np.floor(np.log2(Nw0))+1)) #increase the number of points to the nearest 2**? value
            else:
                Nw=Nw0
            #interpolate spectrum to even ferquency spacing
            Ew=interpolate.PchipInterpolator(self.W,np.abs(self.Field_W)) #spectral phase is removed by np.abs
            W=np.linspace(self.W[0],self.W[-1],Nw)
            Field_W=np.zeros(Nw)*1j
            indS=np.logical_and(np.ones(Nw)*W[0] <= W , W <= np.ones(Nw)*W[-1])
            #calculate the new interpolated E in frequency domain; with 0 values outside the interpolation range
            Field_W[indS]=Ew(W[indS])
            #define corresponding time vector
            Nt=Nw
            dt=2*Pi/(W[-1]-W[0])
            T=np.linspace(-Nt/2,Nt/2-1,Nt)*dt
            Field_T=ifftshift(ifft(Field_W)*np.exp(1j*W[0]*T))
            It=np.abs(Field_T)**2
        else:
            It=np.abs(self.Field_T)**2
            T=self.T
            # plt.plot(T,It)
            # plt.show()
            
        return width(T,It,method)
    
    def t2I(self,t):
        """returns intensity (erlative to peak) at time t in s"""
        ind=self.T<=t
        t0=self.T[ind][-1]
        t1=self.T[np.sum(ind)]
        I=np.abs(self.Field_T)**2
        Im=I.max()
        I0=I[ind][-1]/Im
        I1=I[np.sum(ind)]/Im
        return I0+(I1-I0)/(t1-t0)*(t-t0)
        
    def peakintensity(self):
        It=np.abs(self.Field_T)**2
        return It.max()/np.sum(It)
    
    def TLpeakintensity(self):
        self.Field_T_TL=fourier_fixedT(np.abs(self.Field_W),self.T,self.W)
        It=np.abs(self.Field_T_TL)**2
        return It.max()/np.sum(It)
    
    def temporalSR(self):
        """temporal Strehl ratio"""
        I1=self.peakintensity()
        Itl=self.TLpeakintensity()
        return I1/Itl
    
    def spectrum_width(self,method='FWHM'):
        Iw=np.abs(self.Field_W)**2
        W=self.W
        return width(W,Iw,method)
    
    def peakpower(self,E):
        """peak power in W for a given energy E in J"""
        It=np.abs(self.Field_T)**2
        T=self.T
        dt=T[1]-T[0]
        Im=It.max()
        return E*Im/np.sum(It)/dt
    
    def wigner(self,T,W):
        """calculates wigner fanction
        T and W need to be equidistant data (with constant spacing)"""
        Af=interpolate.PchipInterpolator(self.T,self.Field_T)
        # Tau=T[]
        
    def export_spec_int(self,file=None,evenWspacing=False,lamRange=[]):
        """export spectral intenstiy
        lamRange in nm"""
        if file==None:
            file = filedialog.asksaveasfilename()
        if file != '':
            Y=np.abs(self.Field_W)**2
            if evenWspacing:
                X=self.W.copy()
                IntPh=interpolate.PchipInterpolator(X,Y)
                X=np.linspace(X.min(),X.max(),len(X))
                Y=IntPh(X)
            else:
                X=self.W.copy()
            
            if len(lamRange)>0:
                ind=np.logical_and(X>2*Pi*c/lamRange[1]/10**-9,X<2*Pi*c/lamRange[0]/10**-9)
                X=X[ind]
                Y=Y[ind]
            X=X/2/Pi
            Y/=Y.max()
            np.savetxt(file+'.dat', np.concatenate((np.array(X).reshape((-1,1)), 
                                                    np.array(Y).reshape((-1,1)),
                                                    np.array(c/X*10**9).reshape((-1,1))),axis=1),
                            header='frequency Hz \t Intensity (normalized)\t wavelength (nm)',delimiter='\t',comments='')
            
            # plt.clf()
            plt.plot(X,Y)
            
    def simFROG(self,size=512,slow_custom=True,TimeWindow=200,Type='SHG'):
        """compute SHG FROG with grid dimentions of size
        Timewindow in fs"""
        self.FROG_type=Type
        self.interpolateSpec(size)
        self.spectrum2time(set_T=True,timewindow=[-TimeWindow/2*10**-15,TimeWindow/2*10**-15],
                           tstep=TimeWindow*10**-15/size/4,correct2power2=True,
                      slow_custom=True)
        Tau=self.pulseduration()
        # print(Tau)
        DT=Tau*20
        dt=DT/size
        T=(np.arange(size)-size/2)*dt
        # print(T[0],T[-1])
        if slow_custom:
            Et=fourier_fixedT(self.Field_W_int,T,self.W_int-self.W0)
        FROG=np.zeros((size,size))*1.
        
        #rescale spectrum for FFT
        dw=2*Pi/(T[-1]-T[0])
        Nw=len(T)
        W=np.linspace(-Nw/2,Nw/2-1,Nw)*dw
        for i in range(len(FROG)):
            if Type=='SHG':
                Gt=np.roll(Et,i-int(size/2))
            elif Type=='TG':
                Gt=np.abs(np.roll(Et,i-int(size/2)))**2
            else:
                print('unknown FROG type')
            ind=np.logical_not(np.logical_or(np.arange(len(Et))<i+int(size/2),np.arange(len(Et))>i-int(size/2)))
            Gt[ind]=0
            # FROG[i]=np.abs(fft(fftshift(Et*Gt)))**2
            FROG[i]=np.abs(ifftshift(fft(Et*Gt)))**2
        if Type=='SHG':
            W2=np.linspace(-Nw/2,Nw/2-1,Nw)*dw+self.W0*2
        else:
            W2=np.linspace(-Nw/2,Nw/2-1,Nw)*dw+self.W0
        
        self.FROG=FROG
        self.FROG_T=T*10**15
        self.FROG_lam=2*Pi*c/(W2)*10**9
        self.FROG_W=W2
        
        #plot
        Y=W2*10**-15
        X=T*10**15
        # print(X)
        # print(Y)
        plt.figure(2)
        plt.clf()
        plt.pcolormesh(X,Y,FROG.transpose(),cmap=plt_cmap(),shading='nearest')
        plt.show()
        
        # plt.figure(3)
        # plt.clf()
        # plt.plot(FROG[256])
        
    def interpolateSpec(self,size=512):
        """"""
        Ew=interpolate.PchipInterpolator(self.W,self.Field_W)
        Wint=np.linspace(self.W[0],self.W[-1],size)
        Ewint=Ew(Wint)
        self.W_int=Wint
        self.Field_W_int=Ewint
        
    def exportFROG(self,file=None):
        """export FROG"""
        if file == None:
            file=filedialog.asksaveasfilename(title='file path to save FROG')
        if file == '':
            print('Error in Export FROG. wrong file path')
            #add raise error
        else: 
            np.savetxt(file+'_'+self.FROG_type+'.dat', 
                       np.concatenate(([self.FROG_T],[self.FROG_lam],self.FROG)),
                        delimiter='\t',comments='')
            
    def export_spec_TLtime(self,file=None):
        """export spectrum and TL time"""
        if file==None:
            file = filedialog.asksaveasfilename()
        if file != '':
            #spec
            X=2*Pi*c/self.W*10**6
            Y=np.abs(self.Field_W)**2
            Y/=Y.max()
            np.savetxt(file+'_spec.dat', np.concatenate((np.array(X).reshape((-1,1)), 
                                                    np.array(Y).reshape((-1,1))),axis=1),
                            header='wavelength um \t Intensity (normalized)',delimiter='\t',comments='')
            
            #TL time
            X=self.T
            Y=np.abs(self.Field_T)**2
            Y/=Y.max()
            np.savetxt(file+'_TLtime.dat', np.concatenate((np.array(X).reshape((-1,1)), 
                                                    np.array(Y).reshape((-1,1))),axis=1),
                            header='time fs \t Intensity (normalized)',delimiter='\t',comments='')
            
    def export_spec(self,file=None,smooth=False,G=2):
        if file==None:
            file = filedialog.asksaveasfilename()
        if file != '':
            #spec
            X=2*Pi*c/self.W*10**6
            Y=np.abs(self.Field_W)**2
            if smooth:
                Y=ndimage.gaussian_filter1d(Y,G)
            Y/=Y.max()
            np.savetxt(file+'_spec.dat', np.concatenate((np.array(X).reshape((-1,1)), 
                                                    np.array(Y).reshape((-1,1))),axis=1),
                            header='wavelength um \t Intensity (normalized)',delimiter='\t',comments='')
            
            
        
    def showSpInt(self,normalize=True):
        S=np.abs(self.Field_W)**2
        if normalize:
            Y=S/S.max()
        else:
            Y=S
        X=2*Pi*c/self.W*10**6
        plt.plot(X,Y)
        plt.ylim(bottom=0)
        # plt.xlim((2.8,4))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('wavelength ($\mu$m)',fontsize=18)
        plt.ylabel('Intensity (normalized)',fontsize=18)
        
def fourier_fixedW(Et,T,W,Window=None):
    """calculate fourier transform for Et[T] in the spectral range of W
    Indows: hamming; hanning"""
    # Ew=np.array([np.sum(Et*np.exp(1j*w*T)) for w in W])
    Win=np.ones(len(Et))
    if Window=='hamming':
        Win=np.hamming(len(Et))
    elif Window=='hanning':
        Win=np.hanning(len(Et))
    Ew=np.sum(Et*Win*np.exp(1j*W[:,None]*T),axis=1)
    return Ew

def fourier_fixedT(Ew,T,W):
    """calculate inverse fourier transform for Ew[W] in the temporal range of T"""
    # Et=np.array([np.sum(Ew*np.exp(-1j*W*t)) for t in T])
    Et=np.sum(Ew*np.exp(-1j*T[:,None]*W),axis=1)
    return Et

def remove_phase_jumps(phase,JumpThresold=Pi,FixedJump=True,JumpValue=2*Pi):
    """removes the 2*Pi jumps (or any other specified by jumpThresold)"""
    Nj=np.zeros(len(phase))
    nj=0
    for i in range(1,len(Nj)):
        if np.abs(phase[i]-phase[i-1]) > JumpThresold:
            nj=nj+np.sign(phase[i]-phase[i-1])
        Nj[i]=nj
    
    phaseout=phase-JumpValue*Nj
    phaseout=phaseout-phaseout[int(len(phaseout)/2)]
    return phaseout

def width(X,Y,method='FWHM'):
    """computed the width (for example pulse duration of spectrum width) of a data set.
    data are expected to be 1d np.array"""
    if len(Y)>0:
        M=Y.max()
        if method=='FWHM':
            """in case of mutiple peaks, the maximum width will be returned"""
            # NM=Y.argmax()
            # N1=NM
            # N2=NM
            # for i in range(NM-1,-1,-1):
            #     if Y[i] > M/2: N1=i
            # for i in range(NM+1,len(Y)):
            #     if Y[i] > M/2: N2=i
            level=0.5
            ind=Y > M*level
            indx=np.where(ind == True)[0]
            # print('indx ',indx,len(indx))
            if len(indx)>1:
                N1=indx[0]
                N2=indx[-1]
            else:
                N1=1
                N2=0
            # print(X[N1],X[N2],N1,N2)
            # print(Y[N1]/M,Y[N2]/M)
            # print(Y[N1-1]/M,Y[N2+1]/M)
            
        elif method == 'e^-2':
            level=np.exp(-2)
            ind=Y > level*M
            indx=np.where(ind == True)[0]
            if len(indx)>1:
                N1=indx[0]
                N2=indx[-1]
            else:
                N1=1
                N2=0
            # print(X[N1],X[N2])
        elif method == 'e^-1':
            level=np.exp(-1)
            ind=Y > level*M
            indx=np.where(ind == True)[0]
            if len(indx)>1:
                N1=indx[0]
                N2=indx[-1]
            else:
                N1=1
                N2=0
            # print(X[N1],X[N2])
            
        elif method == '4sigma':
            Xmean=np.sum(X*Y)/np.sum(Y)
            sigmaX=(np.sum((X-Xmean)**2*Y)/np.sum(Y))**0.5
            Width=4*sigmaX
        else:
            raise ER.SL_exception('unknown method')
        # print("XX: ", X[N2]-X[N1],N2,N1)
        if method == '4sigma':
            return Width
        else:
            if N1-1 > 0:
                if Y[N1-1] == M*level:
                    X1=X[N1-1]
                else:
                    y1=Y[N1-1]
                    y2=Y[N1]
                    x1=X[N1-1]
                    x2=X[N1]
                    # print(y1)
                    # print((M*level-y1)/(y2-y1))
                    if abs(y2-y1) > 10**-5:
                        X1=x1+(M*level-y1)/(y2-y1)*(x2-x1) #linear interpolation for accuracy improvement
                        # print(M*level,y1)
                    else:
                        X1=x1
            else:
                X1=X[0]
                
            
            if N2+1 < len(Y)-1:
                if Y[N2+1] == M*level:
                    X2=X[N2+1]
                else:
                    y1=Y[N2]
                    y2=Y[N2+1]
                    x1=X[N2]
                    x2=X[N2+1]
                    if abs(y2-y1) > 10**-5:
                        X2=x1+(M*level-y1)/(y2-y1)*(x2-x1) #linear interpolation for accuracy improvement
                        # print('2 ',M*level,y1)
                    else:
                        X2=x1
            else:
                X2=X[-1]
            # print(abs(y2-y1))
            Width=np.abs(X2-X1)
            # print("width: ",X2-X1)
            return Width
        
    else:
        raise ER.SL_exception('no data for width calculation')
        

        
def tau2dw(t,Stype='Gauss',level = 'FWHM'):
    """returns spectrum width corresponding to the pulse duration t for pulse type Stype"""
    if Stype == 'Gauss':
        if level == 'FWHM':
            return 4*np.log(2)/t
    elif Stype == 'FlatTop':
        if level == 'FWHM':
            return 1.3915573782515103/t*4
        
def SFG(P1,P2):
    """summ frequency generation between the two pulses P1 and P2 definded as pulse class"""
    ws=P1.W0+P2.W0
    print(P1.W0/10**12,P2.W0/10**12,ws/10**12)
    P1.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=0.1*10**-15,slow_custom=True)
    # plt.plot(P1.T,np.abs(P1.Field_T))
    # plt.show()
    P2.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=0.1*10**-15,slow_custom=True)
    # plt.plot(P2.T,np.abs(P2.Field_T))
    # plt.show()
    
    Et=P1.Field_T*P2.Field_T
    DW=max((np.abs(P1.W[-1]-P1.W[0]),np.abs(P2.W[-1]-P2.W[0])))
    W2=np.linspace(ws-DW/2,ws+DW/2,1000)
    Ew=fourier_fixedW(Et,P1.T,W2)
    # plt.plot(W2,np.abs(Ew)**2)
    # plt.show()
    P3=pulse()
    P3.def_spectrum(W2,Ew)
    print(width(W2,np.abs(Ew)**2)/10**12)
    return P3

def plot_duration(material,lam0,T0,thickness,Stype='Gauss',level='FWHM',
                  timewindow=[-500*10**-15,500*10**-15],tstep=1*10**-15,
                  title=None):
    """plot dependence of pulse duraion on the material thickness
    T0 transform limited duraiton in fs
    thickness is a vector of thicknesses to plot
    lam0 is the central wavelength in m
    """
    dw=tau2dw(T0,Stype)
    w0=2*Pi*c/lam0
    N=10**3
    W=np.linspace(w0-5*dw,w0+5*dw,N)
    P=pulse()
    P.init_spectrum(w0,dw,W,Stype)
    T=np.zeros(len(thickness))*1.
    for i in range(len(thickness)):
        th=thickness[i]
        P.zero_spectalphase()
        P.add_materialdisp(material, th)
        P.spectrum2time(timewindow=timewindow,tstep=tstep,slow_custom=True)
        T[i]=P.pulseduration()*10**15
        
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 1.5
    plt.rcParams['figure.figsize'] = (6, 4)
    
    plt.plot(thickness,T,linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('thickness (m)',fontsize=18)
    plt.ylabel('duration (fs)',fontsize=18)
    plt.xlim(left=thickness.min())
    if not title == None:
        plt.title(title,fontsize=18)

def FlistInteg(X,Y,equallyspaced=True):
    """numerical list integration
    to be more precise solution of a differential equation dZ/dx=Y (returns Z)
    data must be equally spaced"""
    Z=np.zeros(Y.shape)*1.
    dx=X[1]-X[0]
    # print(Y1-Y)
    if equallyspaced:
        for i in range(len(X)-1):
            Z[i+1]=integrate.simps(Y[:i+1],dx=dx)
            # Z[i+1]=np.sum(Y[:i+1])*dx
            # Z[i+1]=np.trapz(Y[:i+1],dx=dx)
    else:
        for i in range(len(X)-1):
            Z[i+1]=integrate.simps(Y[:i+1],X[:i+1])
    return Z

def FlistDeff(X,Y,equallyspaced=True):
    """numerical list differentiation
    data should be equally spaced"""
    if equallyspaced:
        dx=X[1]-X[0]
        return np.gradient(Y,dx)
    else:
        return np.gradient(Y,X)
    

#print(width(np.array([0,1,2,3,4,5,6]),np.array([0.1,0,0.2,1,0.9,0,0]),method='FWHM'))

#==== stretching of 800nm pulse
# lam=800*10**-9
# w0=2*Pi*c/lam
# T=50*10**-15
# dw=tau2dw(T)
# N=10**3
# W=np.linspace(w0-5*dw,w0+5*dw,N)
# P=pulse()
# P.init_spectrum(w0,dw,W)
# P.add_materialdisp('FS', 6*10**-3)
# P.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=1*10**-15,slow_custom=True)
# plt.plot(P.T,np.abs(P.Field_T)**2)
# print('pulse duration: ',P.pulseduration()*10**15,' fs')

#==== stretching of OPA pulse
# lam=2000*10**-9
# w0=2*Pi*c/lam
# T=300*10**-15
# stype='Gauss'
# dw=tau2dw(T,stype)
# N=10**3
# W=np.linspace(w0-5*dw,w0+5*dw,N)
# P=pulse()
# P.init_spectrum(w0,dw,W,stype)
# # P.add_materialdisp('GaSe', 1.5*10**-3)
# P.add_dispersion_orders(k2=400*10)
# # P.add_materialdisp('FS', 6*10**-3)
# P.spectrum2time(timewindow=[-5000*10**-15,5000*10**-15],tstep=10*10**-15,slow_custom=True)
# plt.plot(P.T,np.abs(P.Field_T)**2)
# print('pulse duration: ',P.pulseduration()*10**15,' fs')


# I=np.abs(P.Field_T)**2
# I/=I.max()
# plt.show()
# plt.plot(P.T*10**15,I)
# plt.show()

#==== stretching of 1030nm pulse
# lam=1030*10**-9
# w0=2*Pi*c/lam
# T=60*10**-15
# stype='Gauss'
# dw=tau2dw(T,stype)
# print(dw)
# N=10**3
# W=np.linspace(max(0,w0-5*dw),w0+5*dw,N)
# # print(max(0,w0-5*dw),w0+5*dw)
# P=pulse()
# P.init_spectrum(w0,dw,W,stype)
# # P.add_materialdisp('calcite', 25*10**-3)
# P.add_dispersion_orders(k2=1000)
# # P.add_materialdisp('air_simple', 0)
# #'air_simple'
# P.spectrum2time(timewindow=[-3000*10**-15,3000*10**-15],tstep=1*10**-15,slow_custom=True)
# plt.figure(1)
# plt.clf()
# plt.plot(P.T,np.abs(P.Field_T)**2)
# # plt.figure(3)
# # plt.clf()
# # E=np.real(P.Field_T)
# # E/=np.abs(E).max()
# # plt.plot(P.T*10**15,E)
# # I=np.abs(P.Field_T)**2
# # I/=I.max()
# # plt.plot(P.T*10**15,I)
# # plt.show()
# print('pulse duration: ',P.pulseduration()*10**15,' fs')

# plt.clf()
# thik=np.linspace(0,10,31)*10**-3
# plot_duration('CaF2',lam,T,thik,timewindow=[-300*10**-15,300*10**-15],tstep=0.5*10**-15)
# thik=np.linspace(0,10,31)
# plot_duration('air_simple',lam,T,thik,timewindow=[-300*10**-15,300*10**-15],tstep=0.5*10**-15)
# plt.title('FS, 1030 nm, 7 fs input',fontsize=18)

#==== stretching of 2400nm pulse
# lam=2400*10**-9
# w0=2*Pi*c/lam
# lmax=20*10**-6 #max wavelength in the spectrum
# wmin=2*Pi*c/lmax
# T=48*10**-15
# DT=400*10**-15
# dw=tau2dw(T)
# N=6*10**3
# W=np.linspace(max((w0-5*dw,wmin)),w0+5*dw,N)
# P=pulse()
# P.init_spectrum(w0,dw,W)

# P.add_materialdisp('FS', 3*10**-3)

# # P.add_materialdisp('CaF2', 12*10**-3)
# # P.add_materialdisp('YAG', 4*10**-3)
# # P.add_materialdisp('Si', 5*10**-3)
# # print(np.abs(P.Field_W)**2)
# P.spectrum2time(timewindow=[-DT/2,DT/2],tstep=1*10**-15,slow_custom=True)
# plt.plot(P.T,np.abs(P.Field_T)**2)
# print('pulse duration: ',P.pulseduration()*10**15,' fs')



#=========== central wavelength from experiment===

# P=pulse()
# P.loadspectralintensity(correctInt=False)
# #,lam_max=1600
# S=np.abs(P.Field_W)**2
# plt.clf()
# plt.plot(2*Pi*c/P.W*10**9,S/S.max())
# plt.show()
# print(2*Pi*c/P.W0*10**9)
# print(2*Pi*c/P.Wcentral(Cutlevel=0.1)*10**9)
# print(2*Pi*c/P.Wcentral(Cutlevel=0.3)*10**9)

# P.export_spec_int(lamRange=[1200,1500],evenWspacing=True)

# P.loadspectralintensity()
# S=np.abs(P.Field_W)**2
# plt.plot(2*Pi*c/P.W*10**9,S/S.max())
# plt.show()

#====for chirped mirrors
# lam=1030*10**-9
# w0=2*Pi*c/lam
# T=7*10**-15
# stype='FlatTop'
# dw=tau2dw(T,stype)
# N=10**3
# W=np.linspace(max(0,w0-5*dw),w0+5*dw,N)
# P=pulse()
# P.init_spectrum(w0,dw,W,stype)

# # P.loadspectralphase(r'D:\OSU\NEXUS\beamlines\quotes\optics\mirrors\fs\pc2103 5deg pair GDD.txt',
# #                     isGDD=True,Nmirrors=30)
# # P.loadspectralphase(r'D:\OSU\NEXUS\beamlines\quotes\optics\mirrors\fs\b15994_45_gdd.dat',
# #                     DataType='GDD',Nmirrors=20)
# P.loadspectralphase(r'D:\OSU\NEXUS\beamlines\quotes\optics\mirrors\fs\optoman_45_gd.dat',
#                     DataType='GD',Nmirrors=10)
# plt.plot(P.W,P.GD*10**15)
# plt.show()
# plt.plot(P.W,np.abs(P.Field_W))
# plt.show()
# P.add_materialdisp('FS',30*10**-3)
# P.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=0.1*10**-15,slow_custom=True)

# print("Strehl ratio ",P.temporalSR())

# print('pulse duration: ',P.pulseduration()*10**15,' fs')

# I=np.abs(P.Field_T)**2
# I/=I.max()
# plt.plot(P.T*10**15,I)
# plt.show()
# plt.plot(P.W,P.GD*10**15)
# plt.ylim(-50,50)
# plt.show()

#======sum frequency generaion====
# lam1=1030*10**-9
# w1=2*Pi*c/lam1
# w2=w1*3
# T1=30*10**-15
# dw1=tau2dw(T1)
# print('fund width THz', dw1/10**12)
# N=10**3
# W1=np.linspace(w1-5*dw1,w1+5*dw1,N)
# W2=np.linspace(w2-5*dw1,w2+5*dw1,N)

# P1=pulse()
# P1.init_spectrum(w1,dw1,W1)
# P2=pulse()
# P2.init_spectrum(w2,dw1,W2)

# plt.plot(P1.W,np.abs(P1.Field_W)**2)
# plt.show()

# P3=SFG(P1,P2)
# # P3.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=0.1*10**-15,slow_custom=True)
# # plt.plot(P3.T,np.abs(P3.Field_T))
# # plt.show()

# plt.plot(P3.W,np.abs(P3.Field_W)**2)
# plt.show()
    
# print('SFG width ', P3.spectrum_width()/10**12)

# print((P1.W[-1]-P1.W[0])/10**12,(P3.W[-1]-P3.W[0])/10**12)

#=======FROG generation====
# P=pulse()
# P.loadspectralintensity(correctInt=False)
# #,lam_max=1600
# S=np.abs(P.Field_W)**2
# plt.clf()
# plt.plot(2*Pi*c/P.W*10**9,S/S.max())
# plt.show()

# # P.add_materialdisp('FS',1*10**-3)
# P.simFROG()
# # P.simFROG(Type='TG')
# print(P.pulseduration())

# P.exportFROG()



#=========== idler from 10um OPA===
# file=r'D:\OneDrive\experiments\exp data\10um OPA\2021.10.15 10opa\opa 3 aggase 1.6  3.txt'

# file=r'D:\OneDrive\experiments\exp data\10um OPA\2022.08.20\sig 66 62 1140 2.txt'
# # file=r'D:\OneDrive\experiments\exp data\10um OPA\2022.08.20\sig 66 62 1140.txt'
# # file=r'D:\OneDrive\experiments\exp data\10um OPA\2022.08.18\66 62 940  2.txt'


# P=pulse()
# P.loadspectralintensity(file,lam_range=[2.5,3.9],lam_bkg=[[4300,4500]],GF_sigma=4)
# #,lam_max=1600
# S=np.abs(P.Field_W)**2
# plt.figure(1)
# plt.clf()
# plt.plot(2*Pi*c/P.W*10**6,S/S.max())
# plt.ylim(bottom=0)
# plt.show()

# # P.add_materialdisp('YAG', 4*10**-3)
# plt.figure(4)
# plt.clf()
# P.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=1*10**-15,slow_custom=True)
# plt.plot(P.T,np.abs(P.Field_T)**2)
# print('pulse duration: ',P.pulseduration()*10**15,' fs')

# # # print(2*Pi*c/P.W0*10**9)
# # # print(2*Pi*c/P.Wcentral(Cutlevel=0.1)*10**9)

# P.convert2idler()
# S=np.abs(P.Field_W)**2
# plt.figure(2)
# plt.clf()
# plt.plot(2*Pi*c/P.W*10**6,S/S.max())
# plt.ylim(bottom=0)
# plt.show()

# # P.export_spec_int()
# # 
# # P.add_materialdisp('GaSe', 2*10**-3)
# # P.add_materialdisp('Ge', 1*10**-3)
# plt.figure(3)
# plt.clf()
# P.spectrum2time(timewindow=[-200*10**-15,200*10**-15],tstep=1*10**-15,slow_custom=True)
# plt.plot(P.T,np.abs(P.Field_T)**2)
# print('pulse duration: ',P.pulseduration()*10**15,' fs')

# print('central wavelength: ',2*Pi*c/P.Wcentral(Cutlevel=0.06)*10**9)
