"""my color maps for Qt aplications"""

import numpy as np
import pyqtgraph as pg
import os
Path=os.path.dirname((os.path.abspath(__file__)))
print(Path)

def colorset(N,Cmap='default'):
    """return color map for a given color scheme Cmap with the number of elements N
    if N is large than the color set length, color will be repeated"""
    if Cmap=='default':
        file=Path+'default_WM.dat'
        Cset0=np.loadtxt(file)
    
    N1=0
    DN=0
    Lset=len(Cset0)
    Cset=[]
    while N1<N:
        DN=int(np.floor(N1/Lset))
        Cset.append(tuple(Cset0[N1-DN*Lset]*255))
        N1+=1
    
    return Cset


def ImageColorMap(Cmap,grade):
    if Cmap=='Wh_rainbow':
        file=Path+'/Wh_rainbow.dat'
        Colors0=np.loadtxt(file)
    
    #Colors=[pg.QtGui.QColor(cl[0],cl[1],cl[2]) for cl in Colors0]
    Colors=[[int(cl[0]*255),int(cl[1]*255),int(cl[2]*255),255] for cl in Colors0]
    Val=np.linspace(0,1,len(Colors))
    CM=pg.ColorMap(Val,Colors)
    
    return CM.getLookupTable(0.0, 1.0, grade)

#print(ImageColorMap('Wh_rainbow',grade))