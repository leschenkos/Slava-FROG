import os
import numpy as np
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)
Pi=np.pi
from constants import c
import classes.error_class as ER
from scipy import ndimage

def imp_dir(Dir,xcal='nm',filenames=True):
    """importing spectrum, calibrating to angular frequency (if input is in nm)
    return field
    expect *.dat files with material thickness as file names"""
    Fs=os.listdir(Dir)
    Sp=np.array([np.loadtxt(Dir+'\\'+fs) for fs in Fs])
    if filenames:
        pos=np.array([float(fs.split('.dat')[0]) for fs in Fs])
        Spout=[[pos[i],Sp[i]] for i in range(len(pos))]
        Spout.sort(key=lambda A: A[0])
    else:
        Spout=Sp
    return Spout

def imp_spec(FP,xcal='nm',normalize=False,axis=(0,1)):
    """importing spectrum, calibrating to angular frequency (if input is in nm)
    return [angular frequency in 2*Pi*Hz, normalized field (sqrt(intensity))]"""
    try:
        if FP[-4:] == 'spec':
            """load akspec file"""
            axis=(0,4)
            Sp=np.loadtxt(FP,skiprows=8)
            ind0=Sp < np.zeros(Sp.shape)
            Sp[ind0]=0
        elif FP[-3:] == 'txt' or FP[-3:] == 'dat': #standard case with a pure 2 colomn data
            if open(FP,'r').readline() == 'Wavelength\tIntensity\n':
                """file saved by a Wavescan (two column)"""
                Sp=Wavescan(FP,xout='nm')
            else: 
                L=np.fromstring(open(FP,'r').readline(),sep='\t')
                if len(L) == 2:
                    Sp=np.loadtxt(FP)
                else:
                    raise ER.SL_exception('unknown fundamental spectrum file format')
        elif FP[-11:] == 'IntSpectrum':
            """load Sfrogger spectrum"""
            axis=(0,1)
            Sp=np.loadtxt(FP,skiprows=1)
            
            xcal='PHz'            
        else:
            raise ER.SL_exception('unknown fundamental spectrum file format')
    except OSError as er:
        raise ER.ReadError(er)
    else:
        if xcal=='nm':
            Sp1=Sp[:,(axis[0],axis[1])]
            Sp1[:,0]=2*Pi*c/Sp1[:,0]*10**9
            Sp1[:,1]=np.abs(Sp1[:,1])**0.5
            Sp1=Sp1[::-1]
        elif xcal=='THz':
            Sp1=Sp[:,(axis[0],axis[1])]
            Sp1[:,0]=2*Pi*Sp1[:,0]*10**12
            Sp1[:,1]=np.abs(Sp1[:,1])**0.5
        elif xcal=='PHz':
            Sp1=Sp[:,(axis[0],axis[1])]
            Sp1[:,0]=2*Pi*Sp1[:,0]*10**15
            Sp1[:,1]=np.abs(Sp1[:,1])**0.5
        else:
            print("unknown x calibration")
            raise ER.CalibrationError(FP)
        if normalize:
            Sp1[:,1]=Sp1[:,1]/(Sp1[:,1]).max()
        return Sp1

def imp_phase(FP,xcal='nm',normalize=True,axis=(0,1)):
    """importing spectrum, calibrating to angular frequency (if input is in nm)
    return [angular frequency in 2*Pi*Hz, phase in rad]"""
    try:
        if FP[-10:] == 'PhSpectrum':
            """load Sfrogger phase"""
            axis=(0,1)
            Sp=np.loadtxt(FP,skiprows=1)
            xcal='PHz'
        else:
            Sp=np.loadtxt(FP)
    except OSError as er:
        raise ER.ReadError(er)
    else:
        if xcal=='nm':
            Sp1=np.array([[2*Pi*c/s[axis[0]]*10**9,s[axis[1]]] for s in Sp])
            Sp1=Sp1[::-1]
        elif xcal=='THz':
            Sp1=np.array([[2*Pi*s[axis[0]]*10**12,s[axis[1]]] for s in Sp])
        elif xcal=='PHz':
            Sp1=np.array([[2*Pi*s[axis[0]]*10**15,s[axis[1]]] for s in Sp])
        else:
            print("unknown x calibration")
        return Sp1
    
def Wavescan(FP,Filter=True,xout='um',SubstractBackground=True,Lam_background=[[0,0.1]]):
    """import and filter Wavescan data
    Lam_background is a list of ranges to be used for background calculation
    (if all are out of the data range, the background is calculated from the edges of the spectrum)"""
    Sp0=np.loadtxt(FP,skiprows=1)
    Y=Sp0[:,1]
    Y=Y/Y.max()
    
    if xout=='um':
        X=Sp0[:,0]*10**-3 #x in um
    elif xout=='nm':
        X=Sp0[:,0] #x in nm
    # print(X)
    if Filter:
        """filter out the scanning artifacts (mudulations)"""
        #Gaussian filter
        sigma=2 #standard deviation for Gaussian kernel
        Y1=ndimage.gaussian_filter1d(Y,sigma)
    if SubstractBackground:
        #===remove background
        Xmin=X.min()
        Xmax=X.max()
        Lbg=np.array(Lam_background)
        #remove wrong borders for the background calculation
        ind_min=Lbg<Xmin
        Lbg[ind_min]=Xmin
        ind_max=Lbg>Xmax
        Lbg[ind_max]=Xmax
        ind0=[np.logical_and(X>=l[0],X<=l[1]) for l in Lbg]
        if len(ind0) > 1:
            ind=np.sum(np.array(ind0),axis=0)
        else:
            ind=ind0[0]
        if np.sum(ind) < 3:
            #get background from both edges
            ind[:]=False
            ind[:9]=True
            ind[-10:]=True
        # print(np.sum(ind))
        Bg=np.sum(Y1[ind])/np.sum(ind)
        # print(Bg)
    Y1-=Bg
    #zero values below 0
    ind=Y1<0
    Y1[ind]=0
    Sp=np.ones((len(Y1),2))
    Sp[:,0]=X
    Sp[:,1]=Y1
    return Sp

def imp_time(FP,xcal='fs',normalize=True,axis=(0,1)):
    """import temporal pulse profile"""
    
    try:
        if FP[-7:] == 'IntTime':
            """load Sfrogger data"""
            axis=(0,1)
            Sp=np.loadtxt(FP,skiprows=1)
        
        else:
            raise ER.SL_exception('unknown temporal file format')
    except OSError as er:
        raise ER.ReadError(er)
    else:
        if xcal=='fs':
            return Sp
        else:
            print("unknown x calibration")