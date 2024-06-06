"""PCGPA
the code is based on the following papers
J. Opt. Soc. Am. B 25, A120-A132 (2008)
Opt. Express 26, 2643-2649 (2018)
Opt. Express 27, 2112-2124 (2019)
"""

#from the file directory find the first folder containing the name python and append it
import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)

import numpy as np
import matplotlib.pyplot as plt
Pi=np.pi
from myconstants import c, e0
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import classes.error_class as ER
from scipy import interpolate
import scipy.io
from multiprocessing import Pool, cpu_count
from scipy import interpolate

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

def PCGPA_ComM(OO):
    """colomn manipulation for time product
    from [0,-1,-2,-3,-4,3,2,1] to [-4,-3,-2,-1,0,1,2,3] or similar for larger arrays
    """
    OO1=np.roll(OO,-1,axis=0)
    #OO1=OO
    OO2=np.concatenate((OO1[int(len(OO1)/2)-1::-1,:],OO1[:int(len(OO1)/2)-1:-1,:]), axis=0)
    return OO2

def PCGPA_step(pulse_t,gate_t,frog,samepulseandgate=True,Type='SHG-FROG'):
    """frog is supposed to be oriented by delay, that means that each raw (frog[i]) 
    corresponds to a fixed dealy
    it is also assumed that frog has square dimentions propotinal to a power of 2 (2**N)
    pulse and gate are in temporal domain
    """
    
    OO=pulse_t[:,None]*gate_t #outer product
    OOshift=PCGPA_ComM(np.transpose(np.array([np.roll(OO[i],-i) for i in range(len(OO))])))
    frog_sim=np.array([fftshift(fft(fftshift(OOshift[i]))) for i in range(len(OOshift))])
    frog_opt=np.sqrt(frog)*np.exp(1j*np.angle(frog_sim)) #intensity constraint. take amgnitude from exp_frog and phase from simulation
    OOshift_new=np.array([ifftshift(ifft(ifftshift(frog_opt[i]))) for i in range(len(frog_opt))]) #inverse fourier transform
    A=np.transpose(PCGPA_ComM(OOshift_new))
    OO_new=np.array([np.roll(A[i],i) for i in range(len(A))]) #convert to new outer product form
    pulse_new=np.dot(np.dot(OO_new,np.transpose(OO_new)),pulse_t) #new guesses
    gate_new=np.dot(np.dot(np.transpose(OO_new),OO_new),gate_t)
    pulse_new=pulse_new*np.sqrt(np.sum(np.abs(pulse_t)**2)/np.sum(np.abs(pulse_new)**2)) #renormalize
    gate_new=gate_new*np.sqrt(np.sum(np.abs(gate_t)**2)/np.sum(np.abs(gate_new)**2))
    
    if samepulseandgate:
        """in case of autocorrelation FROG such as SHG-FROG (but not X-FROG)"""
        if Type=='SHG-FROG':
            gate_new=pulse_new
        elif Type=='TG-FROG':
            gate_new=np.abs(pulse_new)**2
        #calculate new proper frog_sim
        OOnew=pulse_new[:,None]*gate_new
        OOshift=PCGPA_ComM(np.transpose(np.array([np.roll(OOnew[i],-i) for i in range(len(OOnew))])))
        frog_sim=np.array([fftshift(fft(fftshift(OOshift[i]))) for i in range(len(OOshift))])
    
    frog_sim1=np.abs(frog_sim)**2*np.sum(frog)/np.sum(np.abs(frog_sim)**2) #normalization of simulated trace
    G=np.sqrt(np.sum((frog_sim1-frog)**2)/len(frog)/len(frog[0])) #rms difference between measured and retrieved FROGs
    return (pulse_new,gate_new,G,frog_sim1)

def PCGPA_G(pulse_t,gate_t,frog):
    """calculate G error and frog_sim"""
    OO=pulse_t[:,None]*gate_t
    OOshift=PCGPA_ComM(np.transpose(np.array([np.roll(OO[i],-i) for i in range(len(OO))])))
    frog_sim=np.array([fftshift(fft(fftshift(OOshift[i]))) for i in range(len(OOshift))])
    frog_sim1=np.abs(frog_sim)**2*np.sum(frog)/np.sum(np.abs(frog_sim)**2) #normalization of simulated trace
    G=np.sqrt(np.sum((frog_sim1-frog)**2)/len(frog)/len(frog[0])) #rms difference between measured and retrieved FROGs
    return (G,frog_sim1)

def shift2zerodelay(pulse,gate):
    """shifts pulse to about zero dealy"""
    pulse_w=fftshift(fft(fftshift(pulse)))
    gate_w=fftshift(fft(fftshift(pulse)))
    pulse_w , gate_w = remove_linear_phase(pulse_w,gate_w)
    pulse_tout=ifftshift(ifft(ifftshift(pulse_w)))
    gate_tout=ifftshift(ifft(ifftshift(gate_w)))
    return (pulse_tout,gate_tout)

def remove_linear_phase(pulse,gate):
    """removes the phase slope corresponding to a shift of the pulse from 0 position 
    (it doesnt change the FROG trace)
    the inputs are in spectral domain"""
    #take slope from the region containing 60% of the pulse energy
    
    Int_t=np.abs(pulse)**2
    Phase=remove_phase_jumps(np.angle(pulse))
    E_part=0.8 #to take 60% energy central part
    
    Et=np.sum(Int_t)
    It1=1
    while np.sum(Int_t[:It1]) < Et*E_part : It1 += 1
    It0=len(Int_t)
    while np.sum(Int_t[It0:]) < Et*E_part : It0 -= 1
    
    X=np.arange(It0,It1)
    Y=Phase[It0:It1]
    slope=np.polyfit(X,Y,1)[0] #!!! sometimes returns expected non-empty vector for x
    
    ind=np.arange(len(pulse))
    pulseout=np.abs(pulse)*np.exp(1j*(np.angle(pulse)-ind*slope))
    gateout=np.abs(gate)*np.exp(1j*(np.angle(gate)-ind*slope))
    
    return (pulseout,gateout)

def spectrum_fromFROG(T,W2,frog,Type='SHG-FROG'):
    """frog is supposed to be oriented by delay, that means that each raw (frog[i]) 
    corresponds to a fixed delay
    it is also assumed that frog has square dimentions propotinal to a power of 2 (2**N)"""
    
    SHG_w=np.sum(frog,axis=0) #get the integrated over delay SHG spectrum from the frog trace
#    plt.plot(W,SHG_w)
#    plt.show()
#    SHG_t=np.sqrt(ifftshift(ifft(SHG_w))) #*np.exp(-1j*W2[0]*T)
#    SHG_abs=np.abs(SHG_t)
#    SHG_phase=remove_discontinuity(np.angle(SHG_t),Pi/2)/2
#    S=fft(fftshift(SHG_abs*np.exp(1j*SHG_phase)*np.exp(1j*W2[0]/2*T)))
    dt=2*Pi/W2[int(len(W2)/2)]/8
    T1=np.linspace(T[0],T[-1],int((T[-1]-T[0])/dt+1)) #better time resolution
    SHG_t=ifft_fixed(T1,W2,SHG_w)
    SHG_abs=np.sqrt(np.abs(SHG_t))
    SHG_phase=remove_discontinuity(np.angle(SHG_t),Pi/2)/2
    dw=W2[1]-W2[0]
    if Type=='SHG-FROG':
        W1=W2[int(len(W2)/2)]/2+dw*np.array([i-int(len(W2)/2) for i in range(len(W2))]) #fundamental frequency range (good for recostraction)
    elif Type=='TG-FROG':
        W1=W2
    S=np.abs(fft_fixed(T1,W1,SHG_abs*np.exp(1j*SHG_phase)))
    return S

def fft_fixed(T,W,Et):
    Ew=np.array([np.sum(Et*np.exp(1j*w*T)) for w in W])
    return Ew

def ifft_fixed(T,W,Ew):
    Et=np.array([np.sum(Ew*np.exp(-1j*W*t)) for t in T])
    return Et
    
def remove_discontinuity(array,step_limit,nown_jump=True,jump_value=2*Pi):
    """removes discontinuities, such as 2Pi jumps in phase
    array should be a 1D np.array
    a fixed jump value is assumed
    """
    N0=0
    Nj=np.zeros(len(array))
    if nown_jump:
        Jump=jump_value
    else:
        Jump0=np.zeros(0)
    for i in range(2,len(array)):
        if np.abs(array[i]-array[i-1]) > step_limit:
            N0 += np.sign(array[i]-array[i-1])
            if not nown_jump:
                np.append(Jump0,np.abs(array[i]-array[i-1]))
        Nj[i]=N0
    if not nown_jump:
        Jump=Jump0.max()
    array1=np.array([array[i]-Jump*Nj[i] for i in range(len(array))])
    return array1
    
def load_frog(file):
    """loads a FROG scan"""
    
    if file[-3:]=='txt' or file[-3:]=='dat':
        """for program generated frogs (basically for test purposes)"""
        try:
            T0=open(file,'r').readline()
            T=np.fromstring(T0,sep='\t')
            Sp=np.genfromtxt(file,skip_header=1)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            W=2*Pi*c/Sp[0]*10**9*10**-15
            frog=Sp[1:]
            frog=frog/frog.max()
            return (T, W, frog)
        
    # elif file[-10:]=='h5SpecScan':
    #     """for files saved by akvlXFROG soft"""
    #     try:
    #         data=h5py.File(file, 'r')
    #     except OSError as er:
    #         raise ER.ReadError(er)
    #     else:
    #         T=data['delays'][:]
    #         L=data['wavelengths'][:]
    #         W=2*Pi*c/L*10**9*10**-15
    #         frog=data['trace'][:]
    #         frog=frog/frog.max()
    #         ind=W.argsort()
    #         W=W[ind]
    #         frog=frog[:,ind]
    #         return (T, W, frog)
           
    elif file[-21:]=='akSpecScantransformed':
        """for files saved by akXFROG soft and preprocessed to akSpecScantransformed"""
        try:
            M0=open(file,'r').readline()
            M=np.fromstring(M0,sep='\t') #motor positions
            T=(M-M[0])/c*10**-9*10**15 #delays
            Sp=np.genfromtxt(file,skip_header=1)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            W=2*Pi*c/Sp[0]*10**9*10**-15 #frequencies
            ind=W.argsort()
            W=W[ind]
            Sp1=Sp[1:]
            frog=Sp1[:,ind] #frog data
            frog=frog/frog.max()
            return (T, W, frog)
        
    elif file[-11:]=='txtSpecScan' :
        """for files saved by akvlXFROG_txt soft directly
        or by the catchFROG py software"""
        try:
            M0=open(file,'r').readline()
            T=np.fromstring(M0,sep='\t') #delays in fs
            Sp=np.genfromtxt(file,skip_header=1)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            W=2*Pi*c/Sp[0]*10**9*10**-15 #frequencies
            ind=W.argsort()
            W=W[ind]
            Sp1=Sp[1:]
            frog=Sp1[:,ind] #frog data
            frog=frog/frog.max()
            return (T, W, frog)
        
    elif file[-6:]=='pyfrog':
        """for files saved by akvlXFROG_txt soft directly
        or by the catchFROG py software"""
        try:
            F=open(file,'r')
            F.readline()
            M0=F.readline()
            T=np.fromstring(M0,sep='\t') #delays in fs
            Sp=np.genfromtxt(file,skip_header=2)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            W=2*Pi*c/Sp[0]*10**9*10**-15 #frequencies
            ind=W.argsort()
            W=W[ind]
            Sp1=Sp[1:]
            frog=Sp1[:,ind] #frog data
            frog=frog/frog.max()
            return (T, W, frog)
    
    elif file[-4:]=='frog':
        """for files saved by Sfrogger"""
        try:
            M0=open(file,'r').readline()
            T=np.fromstring(M0,sep='\t') #delays in fs
            Sp=np.genfromtxt(file,skip_header=1)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            W=2*Pi*Sp[0] #frequencies
            frog=Sp[1:] #frog data
            return (T, W, frog)
        
    elif file[-3:]=='frg':
        """for files prepared as for Trebino frogger"""
        try:
            par=np.fromstring(open(file,'r').readline(),sep='\t')
            Nbin=int(par[0])
            dt=par[2]
            dw=par[3]
            W0=par[4]*2*Pi
            Sp=np.genfromtxt(file,skip_header=1)
        except OSError as er:
            raise ER.ReadError(er)
        else:
            frog=np.transpose(Sp.reshape((Nbin,Nbin)))
            frog=frog/frog.max()
            #zero negative values
            ind0=frog < np.zeros(frog.shape)
            frog[ind0]=0
            
            T=dt*np.array([i-Nbin/2 for i in range(Nbin)])
            W=W0+dw*2*Pi*np.array([i-Nbin/2 for i in range(Nbin)])
            
            return (T, W, frog)
    elif file[-9:]=='frgav.npy':
        """for averaged frog traces"""
        In=np.load(file)
        W=In[0,1:]
        T=In[1:,0]
        frog=In[1:,1:]
        return (T, W, frog)
    else:
        raise ER.SL_exception('unknown file format')

def preprocess_frog(T,W,frog,Lmax=None,background=0):
    """Lmax in nm, maximum wavelength to keep (allow cutting noise background from fundamental)
    preprocessing of a frog trace
    removing w background and intensity background if present
    filtering could be optinally added"""
    
    if not Lmax==None:
        wmin=2*Pi*c/Lmax*10**9*10**-15
        ind=W>wmin
        Wout=W[ind]
        frog_out=frog[:,ind]
    
    Max=frog_out.max()
    frog_out=frog_out-Max*background
    Tout=T
    return (Tout,Wout,frog_out)

def resize_frog(T,W,frog,Nbin,Nmax):
    """prepares a proper .frg file with NbinxNbin size
    Nbis is the desired size of the array
    Nmax is the max size of the array
    """
    #determine size
    Int_t=np.sum(frog,axis=1)
    Int_w=np.sum(frog,axis=0)
#    print(len(T),len(Int_t),len(Int_w))
#    plt.plot(T,Int_t)
#    plt.show()
#    plt.plot(Int_w)
#    plt.show()
    E_part=0.9 #level to equalize size
    
    Et=np.sum(Int_t)
    It0=1
    while np.sum(Int_t[:It0]) < Et*E_part : It0 += 1
    It1=len(Int_t)
    while np.sum(Int_t[It1:]) < Et*E_part : It1 -= 1
    dt0=(T[It0]-T[It1])/Nbin
#    print(It0,It1)
    
    Ew=np.sum(Int_w)
    Iw0=1
    while np.sum(Int_w[:Iw0]) < Ew*E_part : Iw0 += 1
    Iw1=len(Int_w)
    while np.sum(Int_w[Iw1:]) < Ew*E_part : Iw1 -= 1
    dw0=(W[Iw0]-W[Iw1])/Nbin
#    print(Iw0,Iw1)
    
    Nbin0=2*Pi/dt0/dw0
    if np.log2(Nbin0)%1:
        Nbin=int(2**(np.floor(np.log2(Nbin0))+1)) #increase the number of points to the nearest 2**? value
    else:
        Nbin=Nbin0
    if Nbin > 2**Nmax:
        Nbin = 2**Nmax
    print(Nbin)
    #resize
    dw=dw0
    dt=2*Pi/dw/(Nbin-1)
    Nt0=Int_t.argmax() #taking frog centrum from the peak position
    Nw0=Int_w.argmax()

    #output trace
    frog2=np.zeros((Nbin,Nbin))
    Wout=np.array([W[Nw0]+dw*(i-Nbin/2) for i in range(Nbin)])
    Tout=np.array([T[Nt0]+dt*(i-Nbin/2) for i in range(Nbin)])
    indW=np.logical_and(np.ones(Nbin)*W[0] <= Wout, Wout <= np.ones(Nbin)*W[-1])
    Ffrog=interpolate.interp2d(T,W,np.transpose(frog),kind='cubic')
    for i in range(Nbin):
        if T[0]<=Tout[i]<=T[-1]:
            frog2[i][indW]=np.transpose(Ffrog(Tout[i],Wout[indW]))[0]
            ind2=frog2[i] < np.zeros(Nbin)
            frog2[i][ind2]=np.zeros(np.sum(ind2)) #remove negative values which could appear after interpolation
    return (Tout-T[Nt0],Wout,frog2)
            
def multigrig_resize(T,W,frog,Scale):
    """resizes a frog trace for the multi-grid iterative approach
    in order to accelerate the initial phase gess
    Scale determines the reduction factor: Nbin_new=Nbin_old/Scale.
    Scale should be a power of 2 (though doesnt have to) for the later fft
    """
    
    Nbin=int(len(frog)/Scale)
    dt=(T[1]-T[0])*Scale**0.5
    dw=(W[1]-W[0])*Scale**0.5
    Wout=np.array([W[int(len(W)/2)]+dw*(i-Nbin/2) for i in range(Nbin)])
    Tout=np.array([T[int(len(T)/2)]+dt*(i-Nbin/2) for i in range(Nbin)])
    if float(np.sqrt(Scale)).is_integer():
        #simplest option. might be useful to limit the case only to the powers of 2
        rescale=int(np.sqrt(Scale))
        frog_out=np.array([[np.sum(frog[k:k+rescale,i:i+rescale])/rescale**2 for i in 
                           range(int(len(frog)/2-len(frog)/rescale/2-int(rescale/2)),
                                 int(len(frog)/2+len(frog)/rescale/2-int(rescale/2)),rescale)] 
                           for k in range(int(len(frog)/2-len(frog)/rescale/2-int(rescale/2)),
                                 int(len(frog)/2+len(frog)/rescale/2-int(rescale/2)),rescale)])
        #print(len(frog_out))
    else:
        Ffrog=interpolate.interp2d(T,W,np.transpose(frog),kind='cubic')
        frog_out=np.zeros((Nbin,Nbin))
        for i in range(Nbin):
            frog_out[i]=np.transpose(Ffrog(Tout[i],Wout))[0]
            
    #zero negative values
    ind0=frog_out < np.zeros(frog_out.shape)
    frog_out[ind0]=0      
    
    return (Tout,Wout,frog_out)

def TBP_frog(T,W,frog):
    """time banswidth product for a frog
    the values defined as containing 70% of energy
    so be careful with applying (remove background and useless frequencies, such as fundumental)
    """
    Int_t=np.sum(frog,axis=1)
    Int_w=np.sum(frog,axis=0)
    E_part=0.85 #for 70% central part energy (15% left from each side)
    
    Et=np.sum(Int_t)
    It0=1
    while np.sum(Int_t[:It0]) < Et*E_part : It0 += 1
    It1=len(Int_t)
    while np.sum(Int_t[It1:]) < Et*E_part : It1 -= 1
    T0=(T[It0]-T[It1])
    
    Ew=np.sum(Int_w)
    Iw0=1
    while np.sum(Int_w[:Iw0]) < Ew*E_part : Iw0 += 1
    Iw1=len(Int_w)
    while np.sum(Int_w[Iw1:]) < Ew*E_part : Iw1 -= 1
    W0=(W[Iw0]-W[Iw1])
    #print(W0)
    
    return T0*W0


def PCGPA_reconstruct_SHG(T,W,frog,G_goal=10**-3,MaxStep=50,SpecFund=[],
                          keep_fundspec=False,MuliGrid=False,Type='SHG-FROG'):
    """PCGPA reconstraction function for SHG-FROG
    MuliGrid: use or not the multi-grid acceleration
    """
    #prepare starting parameters
    if len(SpecFund)==0:
        #retrieve fundamental spectrum
        Sf=spectrum_fromFROG(T,W,frog,Type)
    else:
        Sf=SpecFund
        
    if MuliGrid:
        pass
    else:
        phase=np.random.random(len(Sf))*2*Pi*1 #start from random phase
        pulse_w=np.sqrt(Sf)*np.exp(1j*phase) #in spectral domain
        pulse=ifftshift(ifft(ifftshift(pulse_w))) #convet to time domain
        if Type=='SHG-FROG':
            gate=np.copy(pulse)
        elif Type=='TG-FROG':
            gate=np.abs(np.copy(pulse))**2
    
    Step=0
    G=1
    frog_out=[]
    
    #reconstraction
    while(G > G_goal and Step < MaxStep):
        (pulse,gate,G,frog_out)=PCGPA_step(pulse,gate,frog,Type=Type)
        Step+=1
        if keep_fundspec:
            pulseW=np.sqrt(Sf)*np.exp(1j*np.angle(fftshift(fft(fftshift(pulse)))))
            pulse=ifftshift(ifft(ifftshift(pulseW)))
            if Type=='SHG-FROG':
                gate=np.copy(pulse)
            elif Type=='TG-FROG':
                gate=np.abs(np.copy(pulse))**2
            
    #output spectrum
    pulse_w=fftshift(fft(fftshift(pulse)))
    return (pulse_w,pulse,frog_out,G)

def one(a):
    return a

def parallel_IG(p,T,W,frog,SpecFund,keep_fundspec=False,max_population=12,NStep=25,parallel=True,
                Type='SHG-FROG'):
    """uses the idea from Opt. Express 27, 2112-2124 (2019)
    starts from a number of initial guesses (IG) and returns the best one
    uses the multi-grid approach
    p is a Pool object"""
    
    TBP=TBP_frog(T,W,frog)
    if TBP*4 < max_population:
        population=int(TBP*4)
    else:
        population=max_population
    
    
    #first step with Nbin/4 multigrid
    (T1,W1,frog1)=multigrig_resize(T,W,frog,4) #resize the FROG
    F1=interpolate.PchipInterpolator(W/2,SpecFund)
    Sf1=F1(W1/2)
    if parallel:
        Out1=np.array(p.starmap(PCGPA_reconstruct_IG,
                            [[T1,W1,frog1,NStep,Sf1,keep_fundspec,Type,[]] for i in range(population)]))
    else:
        Out1=list(map(PCGPA_reconstruct_IG,
                            [T1]*population,[W1]*population,[frog1]*population,
                            [NStep]*population,[Sf1]*population,
                            [keep_fundspec]*population,[Type]*population,[[]]*population))
    ind=np.array([Out1[i][0] for i in range(len(Out1))]).argsort()
    Out1=[Out1[m] for m in ind]
    
    #second step with Nbin/2 multigrid
    In2=Out1[:int(len(Out1)/2)] #take best results from the first step
    In21=[]
    for i in range(len(In2)):
        In21.append(shift2zerodelay(In2[i][1],In2[i][2]))
    In2=In21
    (T2,W2,frog2)=multigrig_resize(T,W,frog,2) #resize the FROG
    F2=interpolate.PchipInterpolator(W/2,SpecFund)
    Sf2=F2(W2/2)
    F_pulse=[interpolate.PchipInterpolator(T1,In2[i][1]) for i in range(len(In2))]
    Pulse=np.array([f(T2) for f in F_pulse])
    ind0=np.logical_or(T2<T1[0],T2>T1[-1])
    Pulse[:,ind0]=0
    if parallel:
        Out2=np.array(p.starmap(PCGPA_reconstruct_IG,
                            [[T2,W2,frog2,NStep,Sf2,keep_fundspec,Type,Pulse[i]] 
                            for i in range(len(Pulse))]))
    else:
        Out2=list(map(PCGPA_reconstruct_IG,
                            [T2]*len(Pulse),[W2]*len(Pulse),[frog2]*len(Pulse),[NStep]*len(Pulse),
                            [Sf2]*len(Pulse),[keep_fundspec]*len(Pulse),[Type]*len(Pulse),
                            Pulse))
    ind=np.array([Out1[i][0] for i in range(len(Out1))]).argsort()
    Out1=[Out1[m] for m in ind]
    #fit to the full original time window
    pulse_out=Out2[0][1]
    Fp=interpolate.PchipInterpolator(T2,pulse_out)
    pulse=Fp(T)
    ind0=np.logical_or(T<T2[0],T>T2[-1])
    pulse[ind0]=0
    return pulse
    
def PCGPA_reconstruct_IG(T,W,frog,Steps,SpecFund,keep_fundspec,Type,pulse):
    """calculates one initial guess for the multi-grid algorithm"""
    Sf=SpecFund
    if len(pulse) < 2:
        phase=np.random.random(len(Sf))*2*Pi*1 #start from random phase
        pulse_w=np.sqrt(Sf)*np.exp(1j*phase) #in spectral domain
        pulse=ifftshift(ifft(ifftshift(pulse_w))) #convet to time domain
        if Type=='SHG-FROG':
            gate=np.copy(pulse)
        elif Type=='TG-FROG':
            gate=np.abs(np.copy(pulse))**2
    else:
        gate=np.copy(pulse)
    Step=0
    G=1
    frog_out=[]
    
    #reconstraction
    while(Step < Steps):
        (pulse,gate,G,frog_out)=PCGPA_step(pulse,gate,frog,Type=Type)
        Step+=1
        if keep_fundspec:
            pulseW=np.sqrt(Sf)*np.exp(1j*np.angle(fftshift(fft(fftshift(pulse)))))
            pulse=ifftshift(ifft(ifftshift(pulseW)))
            if Type=='SHG-FROG':
                gate=np.copy(pulse)
            elif Type=='TG-FROG':
                gate=np.abs(np.abs(np.copy(pulse)))**2
            
    #output spectrum
    #pulse_w=fftshift(fft(fftshift(pulse)))
    return (G,pulse,gate)
    