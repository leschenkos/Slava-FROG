import os
import numpy as np
#from the file directory find the first folder containing the name python and append it
Path=os.path.abspath(__file__)
SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
import sys
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)
Pi=np.pi
from constants import c
import classes.error_class as ER


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

def imp_spec(FP,xcal='nm',normalize=True,axis=(0,1)):
    """importing spectrum, calibrating to angular frequency (if input is in nm)
    return field"""
    try:
        if FP[-4:] == 'spec':
            """load akspec file"""
            axis=(0,4)
            Sp=np.loadtxt(FP,skiprows=8)
            ind0=Sp < np.zeros(Sp.shape)
            Sp[ind0]=0
        elif FP[-3:] == 'txt' or FP[-3:] == 'dat': #standard case with a pure 2 colomn data
            L=np.fromstring(open(FP,'r').readline(),sep='\t')
            if len(L) == 2:
                Sp=np.loadtxt(FP)
            else:
                raise ER.SL_exception('unknown fundamental spectrum file format')
        else:
            raise ER.SL_exception('unknown fundamental spectrum file format')
    except OSError as er:
        raise ER.ReadError(er)
    else:
        if xcal=='nm':
            Sp1=np.array([[2*Pi*c/s[axis[0]]*10**9,np.sqrt(np.abs(s[axis[1]]))] for s in Sp])
            Sp1=Sp1[::-1]
        elif xcal=='THz':
            Sp1=np.array([[2*Pi*s[axis[0]]*10**12,np.sqrt(np.abs(s[axis[1]]))] for s in Sp])
        else:
            print("unknown x calibration")
            raise ER.CalibrationError(FP)
        return Sp1

def imp_phase(FP,xcal='nm',normalize=True,axis=(0,1)):
    """importing spectrum, calibrating to angular frequency (if input is in nm)
    return field"""
    try:
        Sp=np.loadtxt(FP)
    except OSError as er:
        raise ER.ReadError(er)
    else:
        if xcal=='nm':
            Sp1=np.array([[2*Pi*c/s[axis[0]]*10**9,s[axis[1]]] for s in Sp])
            Sp1=Sp1[::-1]
        elif xcal=='THz':
            Sp1=np.array([[2*Pi*s[axis[0]]*10**12,s[axis[1]]] for s in Sp])
        else:
            print("unknown x calibration")
        return Sp1