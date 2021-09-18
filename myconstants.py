"""physical constants"""
import numpy as np
Pi=np.pi

c=299792458
ccgs=c*10**3
e0=8.854187817*10**-12 #SI F/m vacuum permittivity or electric constant

hcgs = 4.135667696*10**-15 #eV*s
h = 6.62607015*10**-34 #J*s

me=9.1093837015*10**-31 #kg electron mass
mecgs=me*10**3
me_eV=0.51099895*10**6 #eV
e=1.602176634*10**-19 #electron charge
ecgs=4.80320425*10**-10
re=2.8179403262*10**-15 #m classical electron radius

mp=1.67262192369*10**-27 #kg proton mass
mp_eV=938.27208816*10**6 #eV

k=1.380649*10**-23 #J/K Boltzmann constant
kcgs= 8.617333262145*10**-5 #eV/K Boltzmann constant

Na=6.02214076*10**23 #Avogadro's constant [mol**-1]

Ry=27.211386245988/2 #eV  Rydberg constant

rb=5.291772109*10**-11#m Bohr radius

#___________________________________________
"""gas parameters"""

def P2rho(P,T):
    """convertion of pressure in bar to density in cm^-3 for an ideal gas
    at temperature T in K"""
    return P*10**5/k/T*10**-6

def rho2P(rho,T):
    """convertion of density in cm^-3 to pressure in bar for an ideal gas
    at temperature T in K"""
    return rho/10**-6*k*T/10**5

def bar2torr(P):
    """converts bar to torr"""
    return P*750.06

def torr2bar(P):
    return P/750.06

def cm2MB(area):
    """converst area in cm**2 to MB"""
    return area*10**-18

#___________________________________________
"""strong field parameters"""

def Up(I,lam):
    """ponderomotive energy in eV for I - intensity in W/cm^2, lam - wavelentgh in um"""
    w=2*Pi*c/lam/10**-6
    
    return 2*e**2/c/e0/me*I*10**4/4/w**2/e

def Keldysh(Ip,I,lam):
    """Keldych parameter for Ip in eV, 
    I - intensity in W/cm^2, lam - wavelentgh in um"""
    return (Ip/2/Up(I,lam))**0.5

def Up2I(Up,lam):
    """converst Up in eV to intenstiy in W/cm**2
    lam is a wavelenrgh in um"""
    Eup=Up*e
    w=2*Pi*c/lam/10**-6
    return Eup*c*e0*me/e**2*2*w**2/10**4

def lam2eV(lam):
    """converts wavelength lam in um to corresponding photon energy in eV"""
    w=2*Pi*c/lam/10**-6
    return h*w/e/2/Pi

def cm2eV(k):
    """converts wavenumber in cm**-1 to corresponding photon energy in eV"""
    lam=1/k
    w=2*Pi*c/lam/10**-2
    return h*w/e/2/Pi

def cm2Hz(k):
    """converts wavenumber in cm**-1 to corresponding frequency in Hz"""
    return eV2Hz(cm2eV(k))

def cm2lam(k):
    """converts wavenumber in cm**-1 to corresponding wavelength in um"""
    return eV2lam(cm2eV(k))

def eV2lam(hw):
    """converts photon enrgy in eV to wavelength in um"""
    w=hw*e/h*2*Pi
    return 2*Pi*c/w*10**6

def Hz2eV(v):
    """converts frequency v in Hz to corresponding photon energy in eV"""
    return h*v/e

def Hz2lam(v):
    """converts frequency v in Hz to corresponding wavelength in um"""
    return c/v*10**6

def eV2Hz(hw):
    """converts photon enrgy in eV to frequency in Hz"""
    lam=eV2lam(hw)
    return c/lam/10**-6

#___________________
"""atomic units"""

def Eau2eV(E):
    """converts energy in atomic units to eV"""
    return E*2*Ry

def Rua2m(r):
    """converts distance in atomic units to m"""
    return r*rb

#___________________________________________
"""plasma parameters"""
def nplasma_c(lam):
    """critical plasma density in cm**-3 for a given wavelength in um"""
    w=2*Pi*c/lam/10**-6
    return mecgs*w**2/(4*Pi*ecgs**2) #cgs
    # return e0*me*w**2/e**2/10**6 #Si
    
def wp(n):
    """plasma frequency (electrons oscillations) for a given plasma density in cm**-3
    in rad/s"""
    return (n*4*Pi*ecgs**2/mecgs)**0.5
    
def Mi_freq(rs):
    """Mie surface plasmon frequency of a neutral system 
    rs is Wigner-Seitz radius (2.21A for Ar)"""
    return e/(4*Pi*e0*me*rs**3)**0.5/2/Pi

def it2cluster(Mi,q,rs=2.21*10**-10):
    """time to double the cluster diameter after Coulomb explosion
    Mi ion mass in atomic units
    q average charge
    rs is Wigner-Seitz radius (2.21A for Ar)
    Rev. Mod. Phys. 82, 1793 – Published 8 June 2010.pdf"""
    return 2.3*(2*Pi*e0)**0.5/e*(Mi*mp)**0.5*rs**1.5/q

def v_plasma_sound(Te,Z,Mi):
    """plasma sound velocity in m/s
    (the speed of the longitudinal waves resulting from the mass of the ions 
     and the pressure of the electrons)
    Mi ion mass in atomic units
    Z average charge
    Te electron plasma temperature in eV"""
    return (Z*Te*e/Mi/mp)**0.5
    
def v_e(E):
    """electron velocity from energy in eV"""
    return (2*E*e/me)**0.5

def MB(E,T):
    """Maxwell Bolzman distribution for temperature T in eV and
    kinetic energy E in eV
    normalized to unity integral"""
    return 1/T*np.exp(-E/T)

def ionE2v(E,M):
    """ion velocity from energy in eV and mass M in proton masses
    in m/s"""
    E*=e
    m=mp*M
    v=(2*E/m)**0.5
    return v

def ionv2E(v,M):
    """ion energy in eV from velocity in m/s and mass M in proton masses
    in m/s"""
    m=mp*M
    E=m*v**2/2
    m=mp*M
    return E/e

def eE2v(E):
    """electron velocity from energy in eV
    in m/s"""
    E*=e
    v=(2*E/me)**0.5
    return v




#______________________________
"""pulse parameters"""

def pulse_dw2dt(dw,lam0=800,Input='frequency'):
    """pulse duration from frequency assuming Gaussian pulse and FWHM values
    dw in Hz (Input='frequency') or nm (Input='wavelength')
    if nm then lam0: central wavelength is required [in nm]"""
    if Input=='wavelength':
        dw=dw*c/(lam0*10**-9)**2
    return 4*np.log(2)/Pi/dw

def pulse_dt2dw(dt,lam0=800,Output='frequency'):
    """pulse spectral width from duration assuming Gaussian pulse and FWHM values
    dt in s
    Output in Hz (Output='frequency') or nm (Output='wavelength')
    if nm then lam0: central wavelength is required [in nm]"""
    dw=4*np.log(2)/Pi/dt
    if Output=='frequency':
        return dw
    else:
        return dw/c*(lam0*10**-9)**2
    
def pulse_power(E,T,TypeT='Gaussian',levelT='FWHM'):
    """pulse peak power in W from the pulse energy E in J and duration T in s"""
    if TypeT=='Gaussian':
        if levelT=='FWHM':
            PP=E/T/(Pi/np.log(16))**0.5
        else:
            print('unknown levelT (pulse duration definition level)')
    else:
        PP=E/T
    return PP

def fluence(E,D,TypeD='Gaussian',levelD='e**-2'):
    """pulse fluence in J/cm**2 from energy E in J and beam diameter D in m"""
    if TypeD=='Gaussian':
        if levelD=='e**-2':
            S=Pi*(D/2)**2/2 #m**2
            S*=10**4 #cm**2
            F0=E/S
        else:
            print('unknown levelD (beam size definition level)')
    return F0
    
def pulse_intensity(E,D,T,TypeD='Gaussian',TypeT='Gaussian',levelD='e**-2',levelT='FWHM'):
    """pulse peak intensity in W/cm**2 
    from pulse energy E in J, beam diameter D in m and duration T in s"""
    #peak power
    PP=pulse_power(E,T,TypeT,levelT)
    #peak intensity
    if TypeD=='Gaussian':
        if levelD=='e**-2':
            S=Pi*(D/2)**2/2 #m**2
            S*=10**4 #cm**2
            I0=PP/S
        else:
            print('unknown levelD (beam size definition level)')
    return I0

def I2E(I):
    """calculates field strentgh in V/cm from intensity in W/cm^2"""
    return (2*I/c/e0)**0.5

def a_rel(lam,I):
    """relativistic parameter (normalized vector potential) for a given wavelength in um
    and intensity in W/cm**2"""
    # w=2*Pi*c/lam/10**-6
    # A=I2E(I)/w
    # Acgs=A/299.792458
    # return ecgs*Acgs/mecgs/ccgs**2
    return (I*lam**2/1.38/10**18)**0.5

def Ifrom_a(lam,a):
    """intensity from wavelength in um and relativisitic field strength a (normalized vector potential)"""
    return 1.38*10**18*a**2/lam**2

def I2W(I0,E,T,TypeT='Gaussian',levelT='FWHM'):
    """returns beam diameter on e**-2 from given
    intensity I0 in W/cm**2, energy in J and pulse duration in s"""
    if TypeT=='Gaussian':
        if levelT=='FWHM':
            PP=E/T/(Pi/np.log(16))**0.5 #peak power
            S=PP/I0/10**4
            D=(2*S/Pi)**0.5*2
        else:
            print('unknown levelT (pulse duration definition level)')
    else:
        print('unknown TypeT')
    return D

def LRayleigh(lam,D,TypeD='Gaussian',levelD='e**-2'):
    """Rayleigh in m for a beam waist of D diameter in m and wavelenrgh lam in um"""
    if TypeD=='Gaussian':
        if levelD=='e**-2':
            R=Pi*(D/2)**2/(lam/10**6)
        else:
            print('unknown levelD (beam size definition level)')
    return R

def beam_divergence(lam,D,TypeD='Gaussian',levelD='e**-2'):
    """divergence (diameter) in rad on the e**-2 level for a beam of waist D and wavelenrgh lam in um"""
    if TypeD=='Gaussian':
        if levelD=='e**-2':
            Div=lam/10**6/Pi/(D/2)*2
        else:
            print('unknown levelD (beam size definition level)')
    return Div

def FWHM2E2(Dfwhm):
    """converts diameter from FWHM to e**-2"""
    return Dfwhm*(2/np.log(2))**0.5

def E2FWHM(De):
    """converts diameter from e**-2 to FWHM"""
    return De*(2/np.log(2))**-0.5