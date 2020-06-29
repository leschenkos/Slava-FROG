"""class pulse"""

from load_files.load_folder import imp_spec, imp_phase
import numpy as np
from constants import c
Pi=np.pi
from scipy import interpolate
from scipy.fftpack import fft, ifft, fftshift, ifftshift
#import matplotlib.pyplot as plt
import classes.error_class as ER

class pulse:
    
    def __init__(self,W=[],spectrum_I=[],specrum_phase=[]):
        """defines the pulse using spectral intensity and phase
        W=2*Pi*c/lambda
        """
        self.W=W
        self.Field_W=np.sqrt(spectrum_I)*np.exp(1j*specrum_phase)
        self.W0=np.sum(W*spectrum_I)/np.sum(spectrum_I)
    
    Field_W=np.array([],dtype=np.complex64)
    W=[]
    W0=0 #central frequency (defined as a center of mass)
    Field_T=np.array([],dtype=np.complex64)
    T=[]
    T0=0 #time offset
    
    def loadspectralintensity(self,FP,xcal='nm',axis=(0,1)):
        Sp=imp_spec(FP,xcal=xcal,axis=axis)
        self.W=Sp[:,0]
        self.Field_W=Sp[:,1]
        
    def loadspectralphase(self,FP,xcal='nm',axis=(0,1)):
        Sp=imp_phase(FP,xcal=xcal,axis=axis)
        if len(self.Field_W)==0:
            self.Field_W=np.ones(len(Sp))*1j
        IntPh=interpolate.PchipInterpolator(Sp[:,0],Sp[:,1])
        Ph=IntPh(self.W)
        self.Field_W=self.Field_W*np.exp(1j*Ph)
    
    def def_spectralphase(self,W,Ph):
        IntPh=interpolate.PchipInterpolator(W,Ph)
        Ph=IntPh(self.W)
        self.Field_W=np.real(self.Field_W)*np.exp(1j*Ph)
        
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
            
        return width(T,It,method)
        
    def peakintensity(self):
        It=np.abs(self.Field_T)**2
        return It.max()/np.sum(It)
    
    def spectrum_width(self,method='FWHM'):
        Iw=np.abs(self.Field_W)**2
        W=self.W
        return width(W,Iw,method)
        
def fourier_fixedW(Et,T,W):
    
    Ew=np.array([np.sum(Et*np.exp(1j*w*T)) for w in W])
    return Ew

def fourier_fixedT(Ew,T,W):
    Et=np.array([np.sum(Ew*np.exp(-1j*W*t)) for t in T])
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
        if method=='FWHM':
            """in case of mutiple peaks, the maximum width will be returned"""
            M=Y.max()
            NM=Y.argmax()
            N1=NM
            N2=NM
            for i in range(NM-1,-1,-1):
                if Y[i] > M/2: N1=i
            
            if N1-1 > 0:
                if Y[N1-1] == M/2:
                    X1=X[N1-1]
                else:
                    y1=Y[N1-1]
                    y2=Y[N1]
                    x1=X[N1-1]
                    x2=X[N1]
                    X1=x1+(M/2-y1)/(y2-y1)*(x2-x1) #linear interpolation for accuracy improvement
            else:
                X1=X[0]
                
            for i in range(NM+1,len(Y)):
                if Y[i] > M/2: N2=i
            if N2+1 < len(Y)-1:
                if Y[N2+1] == M/2:
                    X2=X[N2+1]
                else:
                    y1=Y[N2]
                    y2=Y[N2+1]
                    x1=X[N2]
                    x2=X[N2+1]
                    X2=x1+(M/2-y1)/(y2-y1)*(x2-x1) #linear interpolation for accuracy improvement
            else:
                X2=X[-1]
             
            Width=np.abs(X2-X1)
            return Width
        else:
            raise ER.SL_exception('unknown method')
    else:
        raise ER.SL_exception('no data for width calculation')

#print(width(np.array([0,1,2,3,4,5,6]),np.array([0.1,0,0.2,1,0.9,0,0]),method='FWHM'))