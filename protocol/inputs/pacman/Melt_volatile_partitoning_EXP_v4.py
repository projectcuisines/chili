import numpy as np
from scipy import optimize
import sys
import random
from numba import jit
import time
from scipy.optimize import fsolve
from Input_settings import *
from fe_speciation import Iron_speciation_smooth, Iron_speciation_smooth2

###############################################
### Specify planet mass and radius (relative to Earth) - must be changed here and in Main_New.py 
RE = 1 #0.95 for Venus  
ME = 1 #0.815 for Venus
###############################################
###############################################

x = 0.01550152865954013
# H2O solubility
# Constants from figure table 6 in Iacono-Marziano et al. 2012. Using Anhydrous constants
a_H2O = 0.54
b_H2O = 1.24
B_H2O = -2.95
C_H2O = 0.02

A1 = -0.4200250000201988
A2 = -2.59560737813789
M_CO2 = 44.01
C_CO2 = 0.14
C_H2O = 0.02
# CO2 Solubility
# Constants from table 5 in Iacono-Marziano et al. 2012. Using anhydrous
d_H2O = 2.3
d_AI = 3.8
d_FeO_MgO = -16.3
d_Na2O_K2O = 20.1
a_CO2 = 1
b_CO2 = 15.8
C_CO2 = 0.14
B_CO2 = -5.3

# molar masses in g/mol
M_SiO2 = 60
M_TiO2 = 79.866
M_Al2O3 = 101.96
M_FeO = 71.844
M_MgO = 40.3044
M_CaO = 56
M_Na2O = 61.97
M_K2O = 94.2
M_P2O5 = 141.94
M_H2O = 18.01528
M_CO2 = 44.01
M_O2 = 32.0
M_CO = 28.01
M_H2 = 2.016
M_CH4 = 16.04

XAl2O3 = 0.022423 
XCaO = 0.0335
XNa2O = 0.0024 
XK2O = 0.0001077 
XMgO = 0.478144  
XSiO2 =  0.4034   

#M_melt = 2e24
mu_magma = 64.52 #g magma/mol magma
NA = 6.022e23


'''
#optional true masses for Trappist planets
TRAPPIST_R_array = np.array([1.116,1.097,0.788,.920,1.045,1.129,0.775]) ## new agol
TRAPPIST_M_array = np.array([1.374,1.308,0.388,.692,1.039,1.321,0.326]) ## new agol
TRAPPIST_sep_array = np.array([0.01154, 0.0158,0.02227,0.02925,0.03849,0.04683,.06189 ]) # new agol separation
TRAPPIST_Rc_array = TRAPPIST_R_array * 3.4e6
RE = TRAPPIST_R_array[4]
ME = TRAPPIST_M_array[4]
core_radius = TRAPPIST_Rc_array[4]
#end optional true 
'''

'''
## LP 890-9 b and c options
LP9809b_R = 1.320
LP9809b_M = 2.3 #+1.7 - 0.7
LP9809b_sep = 0.01875
LP9809c_R = 1.367
LP9809c_M = 2.5 #+1.8 - 0.8
LP9809c_sep = 0.03984
LP9809_R_array = np.array([LP9809b_R,LP9809c_R]) 
LP9809_M_array = np.array([LP9809b_M,LP9809c_M])
LP9809_sep_array = np.array([LP9809b_sep, LP9809c_sep]) 
LP9809_Rc_array = LP9809_R_array * 3.4e6
core_radius = LP9809_Rc_array[0]
RE = LP9809_R_array[0]
ME = LP9809_M_array[0]
####
'''

##Proxima Centauri b
'''
RE = 1.2
ME = 1.07
'''

MEarth = 5.972e24 #Mass of Earth (kg)
G = 6.67e-11 #gravitational constant
R_planet = RE * 6.371e6 #Planet radius (m)
Mp = ME * MEarth #Planet mass (kg)
gravity = G*Mp/(R_planet**2) # gravity (m/s2)


'''
@jit(nopython=True) 
def Iron_speciation(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar

    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    deltaIW = np.log10(f_O2) - np.log10(fO2_IW) 
    #deltaIW = 2 * np.log10(1.5*xFeO_sil/0.8) ## definition of IW buffer
    #10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW))) =  np.log10(1.5*xFeO_sil/0.8)
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 
    if xFeO_sil>Total_Fe: #undersaturated wrt metallic iron, calcuate iron speciation as normal
        #m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.)  + (Total_Fe) * (56.0+16.0)
        terms1 = 0.196*np.log(f_O2) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        terms = terms1+terms2+terms3+terms4  
        return [(- 1.828 * Total_Fe + terms),Total_Fe] #log(XFe2O3/XFeO)
    else:
        return [-np.inf,xFeO_sil]

@jit(nopython=True) 
def Iron_speciation_smooth(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    #deltaIW_critical = np.log10(f_critical) - np.log10(fO2_IW) = 2 * np.log10(1.5*Total_Fe/0.8)
    #np.log10(f_critical) = 2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)
    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
    terms1 = 0.196*np.log(f_O2) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    terms = terms1+terms2+terms3+terms4  
    #final_log_ratio =  (- 1.828 * Total_Fe + terms)
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 
    if f_O2 >f_critical*10:
        return [np.exp(- 1.828 * Total_Fe + terms),Total_Fe]
    elif f_O2 > f_critical:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        #np.exp(- 1.828 * Total_Fe + terms)*fudge
        return [np.exp((- 1.828 * Total_Fe + terms)+fudge),Total_Fe]  
    else:
        return [np.exp(-np.inf),xFeO_sil]
        
def Iron_speciation_smooth2(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    #deltaIW_critical = np.log10(f_critical) - np.log10(fO2_IW) = 2 * np.log10(1.5*Total_Fe/0.8)
    #np.log10(f_critical) = 2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)
    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
    terms1 = 0.196*np.log(f_O2) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    terms = terms1+terms2+terms3+terms4  
    #final_log_ratio =  (- 1.828 * Total_Fe + terms)
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 
    if f_O2 >f_critical*10:
        return [np.exp(- 1.828 * Total_Fe + terms),Total_Fe]
    elif f_O2 > f_critical*1.00000000001:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        #np.exp(- 1.828 * Total_Fe + terms)*fudge
        return [np.exp((- 1.828 * Total_Fe + terms)+fudge),Total_Fe]  
    else:
        return [np.exp(-np.inf),xFeO_sil]


    #terms = terms1+terms2+terms3+terms4  
   #(- 1.828 * Total_Fe + fudge + 0.196*np.log(f_O2) + terms1+terms2+terms3+terms4)  = np.log(XFe2O3_over_XFeO)
   # 0.196*np.log(f_O2)   = np.log(XFe2O3_over_XFeO) + 1.828* Total_Fe - (terms1+terms2+terms3+terms4) - fudge
   # np.log(f_O2)   = (np.log(XFe2O3_over_XFeO) + 1.828* Total_Fe - (terms1+terms2+terms3+terms4)-fudge)/0.196
    #f_O2 = np.exp ((np.log(XFe2O3_over_XFeO) + 1.828* Total_Fe - (terms1+terms2+terms3+terms4)-fudge)/0.196 ) 

@jit(nopython=True) 
def Iron_speciation_smooth_Sossi(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)) #this remains unchanged
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 #this should also stay the same i think

    # new parameterization from Sossi et al 2020
    log_XFeO1pt5_over_XFeo = 0.252*(np.log10(f_O2) - np.log10(fO2_IW)) - 1.53
    log_XFe2O3_over_XFeO = log_XFeO1pt5_over_XFeo - np.log10(2) 

    if f_O2 >f_critical*10:
        return [10**(log_XFe2O3_over_XFeO),Total_Fe]        ### returns log_XFe2O3_over_XFeO as log10
    elif f_O2 > f_critical:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        return [10**(log_XFe2O3_over_XFeO)+fudge,Total_Fe]  ### returns log_XFe2O3_over_XFeO as log10
    else:
        return [10**(-np.inf),xFeO_sil]

@jit(nopython=True) 
def Iron_speciation_smooth2_Sossi(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)) #this remains unchanged
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 #this should also stay the same i think

    # new parameterization from Sossi et al 2020
    log_XFeO1pt5_over_XFeo = 0.252*(np.log10(f_O2) - np.log10(fO2_IW)) - 1.53
    log_XFe2O3_over_XFeO = log_XFeO1pt5_over_XFeo - np.log10(2)

    if f_O2 >f_critical*10:
        return [10**(log_XFe2O3_over_XFeO),Total_Fe]        ### returns log_XFe2O3_over_XFeO as log10
    elif f_O2 > f_critical*1.00000000001:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        return [10**((log_XFe2O3_over_XFeO)+fudge),Total_Fe]  ### returns log_XFe2O3_over_XFeO as log10
    else:
        return [10**(-np.inf),xFeO_sil]

def Iron_speciation_Sossi(f_O2,P,T,Total_Fe): #f_O2 needs to be in bar
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    log_XFeO1pt5_over_XFeo = 0.252*(np.log10(f_O2) - np.log10(fO2_IW)) - 1.53
    log_XFe2O3_over_XFeO = log_XFeO1pt5_over_XFeo - np.log10(2)
    return np.log(10**log_XFe2O3_over_XFeO)'''
    
    #log_XFeO1pt5_over_XFeo = 0.252*deltaIW - 1.53
    #log_XFeO1pt5_over_XFeo = log(XFeO1.5 / XFeO) = log(0.5 XFe2O3/XFeO) = log(0.5) + log(XFe2O3_over_XFeO)
    #log_XFeO1pt5_over_XFeo = np.log10(0.5) + np.log10(XFe2O3_over_XFeO)
    #[A,B,C] = [27215 ,6.57 ,0.0552]
    #fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    #deltaIW = np.log10(fO2) - np.log10(fO2_IW)
    #log_XFeO1pt5_over_XFeo = 0.252*(np.log10(fO2) - np.log10(fO2_IW)) - 1.53
    #fO2  = 10**( (log_XFeO1pt5_over_XFeo + 1.53)/.252)
    #fO2 =  np.exp( (np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe -(terms1+terms2+terms3+terms4) )/0.196)
    #return fO2 

@jit(nopython=True) 
def system1_graphite(y,d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Tinput,M_melt,Total_Fe): #n_O = y[4] all free oxygen
    T = Tinput#1500.0+273
    #Total_Fe = 0.1
    ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_natm,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2 = y
    f_O2 = np.exp(ln_fO2)
    #print('f_O2',f_O2)

    P = np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)+f_O2
    [P_CH4,P_CO2,P_CO,P_H2O,P_H2] = [np.exp(ln_CH4),np.exp(ln_CO2),np.exp(ln_CO),np.exp(ln_H2O),np.exp(ln_H2)]

    #free_oxygen = n_O - all_other_oxygen
    
    #[XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth_Sossi(f_O2,P*1e5,T,Total_Fe)      #Sossi
    [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)             #Kress & Carmich
#    np.exp(log_XFe2O3_over_XFeO) = XFe2O3/XFeO
#     
   # XFeO =(Total_Fe - 2*XFe2O3) = (Total_Fe - 2*XFeO*np.exp(log_XFe2O3_over_XFeO)) 
#    XFeO #(1 + 2 * np.exp(log_XFe2O3_over_XFeO)) = Total_Fe
    XFeO = Total_Fe_New / (1 + 2 * (XFe2O3_over_XFeO))
    XFe2O3 =XFeO * (XFe2O3_over_XFeO)
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0) + 0*(Total_Fe - Total_Fe_New) * 56.0
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.)  + Total_Fe * (56.0+16.0) 
    
    F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
    F_FeO = XFeO * (56.0 + 16.0) / m_sil 
    F_metal = (Total_Fe - Total_Fe_New)*56/m_sil #this is true metal amount

    if Metal_sink_switch == 1:
        F_metal = (Initial_FeO - Total_Fe_New)*56/m_sil ## this is just leftover oxygen, not actual metal in melt

    free_oxygen = M_melt*F_FeO1_5*0.5*16/(56.0+1.5*16.) - M_melt*F_metal*16/56.0 ## basically adding back oxygen from metal
    n_free_oxygen = free_oxygen/0.016


    C1 = K1/f_O2**0.5
    C2 = K2/f_O2**0.5
    C3 = K3/f_O2**2
    
    
    natm = np.exp(ln_natm)
    n_graphite = np.exp(ln_ngraphite)

#    np.exp(ln_m_H2O)*M_melt
    
    #Pfake = 1.0
    F1 = np.log(1/(M_CO2*x*10**6))+C_CO2*P/T+A1
    F2 = np.log(1/(M_H2O*x*100))+C_H2O*P/T+A2

    log10_K1 = 40.07639 - 2.53932e-2 * T + 5.27096e-6*T**2 + 0.0267 * (P - 1 )/T
    log10_K2 = - 6.24763 - 282.56/T - 0.119242 * (P - 1000)/T
    gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*f_O2)/(1+(10**log10_K1)*(10**log10_K2)*f_O2) 
    gXCO2_melt = grph_set*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction


    if gXCO2_melt>np.exp(ln_m_CO2):#n_C*(M_CO2/1000.)/M_melt: #undersaturated even if all CO2 in melt


        return (np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
            -np.log(np.exp(ln_m_CO2)*(mu_magma/M_CO2))+np.exp(ln_m_H2O)*(mu_magma/M_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -np.log(np.exp(ln_m_H2O)*(mu_magma/M_H2O))+a_H2O*ln_H2O+F2,\
            1000*(-n_H/natm + (4*P_CH4/P + 2 * P_H2O/P + 2 * P_H2/P) + (2/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0)+(2/natm)*np.exp(ln_m_H2)*M_melt/(M_H2/1000.0)),\
            1000*(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P) +  (1/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0)),\
            -P  + (gravity * natm /(4*np.pi*1e5*R_planet**2)) * (M_CO2*P_CO2/P + M_CO*P_CO/P + M_CH4*P_CH4/P + M_H2O*P_H2O/P + M_H2*P_H2/P+M_O2*f_O2/P)/1000.0,\
            1000*(-n_O/natm + n_free_oxygen/natm + (2*P_CO2/P + P_CO/P+ P_H2O/P + 2*f_O2/P ) +  (2/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0) +  (1/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0) ),\
             n_graphite/natm,\
             -np.log10(np.exp(ln_m_H2)*1e6)+0.524139*np.log10(P_H2)+1.100836 ) #np.log10(ppmw) = 0.524139*np.log10( P_H2) + 1.100836, where ppmw = np.exp(ln_m_H2)*1e6
             #so np.log10(np.exp(ln_m_H2)*1e6) = 0.524139*np.log10( P_H2) + 1.100836
  
    else:
        #now 'ln_m_CO2' is not unknown, and n_graphite is
        fixed_m_CO2 =gXCO2_melt
        #print('mco2',fixed_m_CO2,np.exp(ln_m_CO2))
        #print('n_graphite',n_graphite,n_C)
        #print(-n_C/natm , (P_CH4/P + P_CO2/P + P_CO/P) , (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0),n_graphite/natm)
        #print(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P) +  (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0)+n_graphite/natm)
        #n_graphite = natm*(n_C/natm - (P_CH4/P + P_CO2/P + P_CO/P) -  (1/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0) )
        #print(np.log(C1)+ln_H2O-ln_H2,\
        #    np.log(C2)+ln_CO2-ln_CO,\
        #    np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
        #    -np.log(fixed_m_CO2*(mu_magma/M_CO2))+np.exp(ln_m_H2O)*(mu_magma/M_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
        #    -np.log(np.exp(ln_m_H2O)*(mu_magma/M_H2O))+a_H2O*ln_H2O+F2,\
        #    1000*(-n_H/natm + (4*P_CH4/P + 2 * P_H2O/P + 2 * P_H2/P) + (2/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0)),\
        #    1000*(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P) +  (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0)+n_graphite/natm),\
        #    -P  + (gravity * natm /(4*np.pi*1e5*R_planet**2)) * (M_CO2*P_CO2/P + M_CO*P_CO/P + M_CH4*P_CH4/P + M_H2O*P_H2O/P + M_H2*P_H2/P)/1000.0,\
        #    -fixed_m_CO2+np.exp(ln_m_CO2))
        return (np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
            -np.log(fixed_m_CO2*(mu_magma/M_CO2))+np.exp(ln_m_H2O)*(mu_magma/M_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -np.log(np.exp(ln_m_H2O)*(mu_magma/M_H2O))+a_H2O*ln_H2O+F2,\
            1000*(-n_H/natm + (4*P_CH4/P + 2 * P_H2O/P + 2 * P_H2/P) + (2/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0)+(2/natm)*np.exp(ln_m_H2)*M_melt/(M_H2/1000.0)),\
            1000*(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P) +  (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0)+n_graphite/natm),\
            -P  + (gravity * natm /(4*np.pi*1e5*R_planet**2)) * (M_CO2*P_CO2/P + M_CO*P_CO/P + M_CH4*P_CH4/P + M_H2O*P_H2O/P + M_H2*P_H2/P + M_O2*f_O2/P)/1000.0,\
            1000*(-n_O/natm + n_free_oxygen/natm + (2*P_CO2/P + P_CO/P+ P_H2O/P + 2*f_O2/P ) +  (2/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0) +  (1/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0) ),\
            -fixed_m_CO2+np.exp(ln_m_CO2),\
            -np.log10(np.exp(ln_m_H2)*1e6)+0.524139*np.log10(P_H2)+1.100836)


@jit(nopython=True) 
def system1_GASONLY(y,d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Tinput,Total_Fe): # 
    T = Tinput
    #Total_Fe = 0.1
    ln_H2O,ln_CO2,ln_natm,ln_H2,ln_CH4,ln_CO,ln_fO2 = y
    f_O2 = np.exp(ln_fO2)
    #print('f_O2',f_O2)
    natm = np.exp(ln_natm)
    P = np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)+f_O2
    [P_CH4,P_CO2,P_CO,P_H2O,P_H2] = [np.exp(ln_CH4),np.exp(ln_CO2),np.exp(ln_CO),np.exp(ln_H2O),np.exp(ln_H2)]

    C1 = K1/f_O2**0.5
    C2 = K2/f_O2**0.5
    C3 = K3/f_O2**2

    if 2>1:#n_C*(M_CO2/1000.)/M_melt: #undersaturated even if all CO2 in melt
        return (np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
            1000*(-n_H/natm + (4*P_CH4/P + 2 * P_H2O/P + 2 * P_H2/P)),\
            1000*(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P)),\
            -P  + (gravity * natm /(4*np.pi*1e5*R_planet**2)) * (M_CO2*P_CO2/P + M_CO*P_CO/P + M_CH4*P_CH4/P + M_H2O*P_H2O/P + M_H2*P_H2/P+M_O2*f_O2/P)/1000.0,\
            1000*(-n_O/natm + (2*P_CO2/P + P_CO/P+ P_H2O/P + 2*f_O2/P )) )
  



@jit(nopython=True) 
def simple_system1_linear_graphite(y,d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Pfixed,Mfudge,Tinput,M_melt,Total_Fe):
    T = Tinput
    #Total_Fe = 0.1
    ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2 = y 

    f_O2 = np.exp(ln_fO2)

    P = np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)+f_O2
    [P_CH4,P_CO2,P_CO,P_H2O,P_H2] = [np.exp(ln_CH4),np.exp(ln_CO2),np.exp(ln_CO),np.exp(ln_H2O),np.exp(ln_H2)]

    #free_oxygen = n_O - all_other_oxygen
    
    #print(f_O2,P*1e5,T,Total_Fe)
    #[XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth_Sossi(f_O2,P*1e5,T,Total_Fe)          #Sossi
    [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)                 #Kress & Carmich

    XFeO = Total_Fe_New / (1 + 2 * (XFe2O3_over_XFeO))
    XFe2O3 =XFeO * (XFe2O3_over_XFeO)
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0) + 0*(Total_Fe - Total_Fe_New) * 56.0
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  XFeO * (56.0+16.0) 
    F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
    F_FeO = XFeO * (56.0 + 16.0) / m_sil 
    F_metal = (Total_Fe - Total_Fe_New)*56/m_sil


    if Metal_sink_switch == 1:
        F_metal = (Initial_FeO - Total_Fe_New)*56/m_sil

    #free_oxygen = M_melt*F_FeO1_5*0.5*16/(56.0+1.5*16.) #
    free_oxygen = M_melt*F_FeO1_5*0.5*16/(56.0+1.5*16.) - M_melt*F_metal*16/56.0
    n_free_oxygen = free_oxygen/0.016

    C1 = K1/f_O2**0.5
    C2 = K2/f_O2**0.5
    C3 = K3/f_O2**2

    n_graphite = np.exp(ln_ngraphite)
    #natm = np.exp(ln_natm)
#    np.exp(ln_m_H2O)*M_melt
    
    #Pfake = 1.0
    F1 = np.log(1/(M_CO2*x*10**6))+C_CO2*Pfixed/T+A1
    F2 = np.log(1/(M_H2O*x*100))+C_H2O*Pfixed/T+A2

    natm = Pfixed*(4*np.pi*R_planet**2*1e5)/(Mfudge*gravity)

    log10_K1 = 40.07639 - 2.53932e-2 * T + 5.27096e-6*T**2 + 0.0267 * (Pfixed - 1 )/T
    log10_K2 = - 6.24763 - 282.56/T - 0.119242 * (Pfixed - 1000)/T
    gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*f_O2)/(1+(10**log10_K1)*(10**log10_K2)*f_O2) 
    gXCO2_melt = grph_set*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction

    if gXCO2_melt>np.exp(ln_m_CO2):#n_C*(M_CO2/1000.)/M_melt: #undersaturated even if all CO2 in melt
        return (np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
            -np.log(np.exp(ln_m_CO2)*(mu_magma/M_CO2))+np.exp(ln_m_H2O)*(mu_magma/M_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -np.log(np.exp(ln_m_H2O)*(mu_magma/M_H2O))+a_H2O*ln_H2O+F2,\
            1000*(-n_H/natm + (4*P_CH4/Pfixed + 2 * P_H2O/Pfixed + 2 * P_H2/Pfixed) + (2/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0)+(2/natm)*np.exp(ln_m_H2)*M_melt/(M_H2/1000.0)),\
            1000*(-n_C/natm + (P_CH4/Pfixed + P_CO2/Pfixed + P_CO/Pfixed) +  (1/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0)),\
            -P  + Pfixed,\
            1000*(-n_O/natm + n_free_oxygen/natm + (2*P_CO2/Pfixed + P_CO/Pfixed+ P_H2O/Pfixed + 2*f_O2/Pfixed) +  (2/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0) +  (1/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0) ),\
             n_graphite/natm,\
             -np.log10(np.exp(ln_m_H2)*1e6)+0.524139*np.log10(P_H2)+1.100836) 
    else:
        #now 'ln_m_CO2' is not unknown, and n_graphite is
        fixed_m_CO2 = gXCO2_melt
        #print('mco2',fixed_m_CO2,np.exp(ln_m_CO2))
        #print('n_graphite',n_graphite,n_C)
        #print(-n_C/natm , (P_CH4/P + P_CO2/P + P_CO/P) , (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0),n_graphite/natm)
        #print(-n_C/natm + (P_CH4/P + P_CO2/P + P_CO/P) +  (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0)+n_graphite/natm)
        #n_graphite = natm*(n_C/natm - (P_CH4/P + P_CO2/P + P_CO/P) -  (1/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0) )
        return (np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4,\
            -np.log(fixed_m_CO2*(mu_magma/M_CO2))+np.exp(ln_m_H2O)*(mu_magma/M_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -np.log(np.exp(ln_m_H2O)*(mu_magma/M_H2O))+a_H2O*ln_H2O+F2,\
            1000*(-n_H/natm + (4*P_CH4/Pfixed + 2 * P_H2O/Pfixed + 2 * P_H2/Pfixed) + (2/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0)+(2/natm)*np.exp(ln_m_H2)*M_melt/(M_H2/1000.0)),\
            1000*(-n_C/natm + (P_CH4/Pfixed + P_CO2/Pfixed + P_CO/Pfixed) +  (1/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0)+n_graphite/natm),\
            -P  + Pfixed,\
            1000*(-n_O/natm + n_free_oxygen/natm + (2*P_CO2/Pfixed + P_CO/Pfixed+ P_H2O/Pfixed + 2*f_O2/Pfixed) +  (2/natm)*fixed_m_CO2*M_melt/(M_CO2/1000.0) +  (1/natm)*np.exp(ln_m_H2O)*M_melt/(M_H2O/1000.0) ),\
            -fixed_m_CO2+np.exp(ln_m_CO2),\
            -np.log10(np.exp(ln_m_H2)*1e6)+0.524139*np.log10(P_H2)+1.100836)




def solve_gases_new(T,y4_input,mf_C,mf_H,Guess,Guess_ar2,M_melt,Total_Fe):
    #global int_time0,int_time1,int_time2,int_time3,int_time4,int_time5

    int_time0 = 0
    int_time1 = 0
    int_time2 = 0
    int_time3 = 0
    int_time4 = 0
    int_time5 = 0

    #print('int_times',int_time0,int_time1,int_time2,int_time3,int_time4,int_time5)
    '''
    This function solves for the speciation of gases produced by
    a volcano. This code assumes magma composition of the lava erupting at
    Mt. Etna Italy.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt
    mf_C = mass fraction of C in the magma
    mf_H = mass fraction of H in the magma

    Outputs:
    an array which contains
    [PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O, mH2O,mH2,mCO2,mCO,mCH4,mO2]
     where
     PH2O = partial pressure of H2O in the gas in bar
     PH2 = partial pressure of H2 in the gas in bar
     PCO2 = partial pressure of CO2 in the gas in bar
     PCO = partial pressure of CO in the gas in bar
     PCH4 = partial pressure of CH4 in the gas in bar
     alphaG = moles of gas divide by total moles in gas and magma combined
     xCO2 = mol fraction of the CO2 in the magma
     xH2O = mol fraction of the H2O in the magma
     mH2O = fraction of H2O gas (mixing ratio), mH2O = PH2O/P
     mH2 = fraction of H2 gas (mixing ratio), mH2 = pH2/P etc.
     mCO2 = fraction of CO2 gas (mixing ratio)
     mCO = fraction of CO gas (mixing ratio)
     mCH4 = fraction of CH4 gas (mixing ratio)
     mO2 = fraction of O2 gas (mixing ratio, what you put in)
    '''
    tm0= time.time()
    #print('sol inside 1')
    ###### Solubility constants
    a_H2O = 0.54
    a_CO2 = 1
    d_H2O = 2.3

    # mol of magma/g of magma
    x = 0.01550152865954013

    # molar mass in g/mol
    M_H2O = 18.01528
    M_CO2 = 44.01

    A1 = -0.4200250000201988
    A2 = -2.59560737813789
    M_CO2 = 44.01
    C_CO2 = 0.14
    C_H2O = 0.02
    #F1 = np.log(1/(M_CO2*x*10**6))+C_CO2*P/T+A1
    #F2 = np.log(1/(M_H2O*x*100))+C_H2O*P/T+A2

    # calculate mol fraction of CO2 and H2O in the magma

    # equilibrium constants
    # made with Nasa thermodynamic database (Burcat database)
    K1 = np.exp(-29755.11319228574/T+6.652127716162998)
    K2 = np.exp(-33979.12369002451/T+10.418882755464773)
    K3 = np.exp(-96444.47151911151/T+0.22260815074146403)

    #constants
    #C1 = K1/f_O2**0.5
    #C2 = K2/f_O2**0.5
    #C3 = K3/f_O2**2

    # now use the solution of the simple system to solve the
    # harder problem. I will try to solve it two different ways to
    # make sure I avoid errors.

    # error tolerance
    tol = 1e-5
    [x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-2,1e-4,100.,10.,0.1,10.,1.,10.] 
    [x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-7,1e-3,0.01,10.,0.1,0.001,.001,.00010] 
    #[x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-7,1e-8,0.01,0.001,0.1,0.001,.001,.00010]
    #import pdb
    #pdb.set_trace()

    n_H = mf_H*M_melt/0.001 # mass fraction H -> mol H
    n_C = mf_C*M_melt/0.012 # mass fraction C -> mol C
    n_O = y4_input/0.016
    #first see if can solve using guess
    #print(Guess)
    #import pdb
    #pdb.set_trace()

    [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check,f_O2_g,m_H2_g] = Guess
    #print ('graphite guess 1',np.log(solid_graph_g*M_melt/0.012))
    init_cond_guess = [np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012),np.log(f_O2_g),np.log(m_H2_g)] 
    #print ('always')
    #print ('init_cond_guess',init_cond_guess)

    #if (i>=524):
    #    import pdb
    #    pdb.set_trace()
    try:
        #import pdb
        #pdb.set_trace()
        sol_guess = optimize.root(system1_graphite,init_cond_guess,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
        error_guess = np.linalg.norm(system1_graphite(sol_guess['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe))
        ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2   = sol_guess['x']
        f_O2 = np.exp(ln_fO2)
        n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
        P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4) + f_O2
        natm = np.exp(ln_natm)
        solid_graphite = n_graphite * 0.012/M_melt
        [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)         #Kress & Carm
        #print ('final_first_attmpt',[ ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2])

        if error_guess<tol:
            ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2 = sol_guess['x']
            n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
            #print ('final graphite 1',ln_ngraphite)
            P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+ np.exp(ln_fO2)
            natm = np.exp(ln_natm)
            f_O2 = np.exp(ln_fO2)
            solid_graphite = n_graphite * 0.012/M_melt
            return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
                np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
                np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite,np.array([int_time0,int_time1,int_time2,int_time3,int_time4,int_time5]),np.exp(ln_m_H2),Total_Fe_New)
    except:
        abc=1

    tm1= time.time()
    int_time0 = int_time0 + tm1-tm0

    #print ('OH NO HAPPENED HERE Guess failed')
    #print('error_guess',error_guess)
    #ace = np.array([np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
    #        np.exp(ln_P_CH4),natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
    #        np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite])
    #print ('outputs first attempt',ace)

    #import pdb
    #pdb.set_trace()

    #print('sol inside 2')


    Pfixedg = PH2O_g+PH2_g+PCO2_g+PCO_g+PCH4_g+f_O2_g
    Mfudge = (PH2O_g*0.018+PH2_g*0.02+PCO2_g*0.044+PCO_g*0.028+PCH4_g*0.016+f_O2_g*0.032)/(PH2O_g+PH2_g+PCO2_g+PCO_g+PCH4_g+f_O2_g)
    init_cond1 = [np.log(m_H2O_g),np.log(m_CO2_g),np.log(P_H2O),np.log(PCO2_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012),np.log(f_O2_g),np.log(m_H2_g)] 
    #print ('graphite guess 2',np.log(solid_graph_g*M_melt/0.012))
    sol1 = optimize.root(simple_system1_linear_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Pfixedg,Mfudge,T,M_melt,Total_Fe),method='lm',options={'maxiter': 1000})
    error = np.linalg.norm(simple_system1_linear_graphite(sol1['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Pfixedg,Mfudge,T,M_melt,Total_Fe))
    
    ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2 = sol1['x']
    #print ('final graphite 2',ln_ngraphite)
    Pfixed = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2) + np.exp(ln_P_CH4)+np.exp(ln_P_CO)+np.exp(ln_fO2)
    Mfudge_fin = (np.exp(ln_P_H2O)*0.018+np.exp(ln_P_H2)*0.02+np.exp(ln_P_CO2)*0.044+np.exp(ln_P_CO)*0.028+np.exp(ln_P_CH4)*0.016+np.exp(ln_fO2)*0.032)/Pfixed
    ln_natm = np.log(Pfixed*(4*np.pi*R_planet**2*1e5)/(Mfudge_fin*gravity))
    #log_XFe2O3_over_XFeO,Total_Fe_New = Iron_speciation(np.exp(ln_fO2),Pfixed*1e5,T,Total_Fe)

    init_cond1 = [ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2]
    #print ('graphite guess 3',ln_ngraphite)
    sola = optimize.root(system1_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
    errora = np.linalg.norm(system1_graphite(sola['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe))
    
    tm2= time.time()
    int_time1 = int_time1 + tm2-tm1
    #import pdb
    #pdb.set_trace()
    #print ('second attempt')
    if errora<tol:
        ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2   = sola['x']
        n_graphite = np.exp(ln_ngraphite)
        #print ('final graphite 3',ln_ngraphite)
        P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+ np.exp(ln_fO2)
        f_O2 = np.exp(ln_fO2)
        natm = np.exp(ln_natm)
        solid_graphite = n_graphite * 0.012/M_melt
        #[XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth_Sossi(f_O2,P*1e5,T,Total_Fe)  #Sossi
        [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)         #Kress & Carmich
        #print ('Guess worked')
        #print ('fixed it the second time')
        return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
            np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
            np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite,np.array([int_time0,int_time1,int_time2,int_time3,int_time4,int_time5]),np.exp(ln_m_H2),Total_Fe_New)
    else:
        Guess_mix = np.copy(Guess)
        random_loop = 0
        while random_loop <5:
            #import pdb
            #pdb.set_trace()
            #print('Guess',Guess)
            Guess_mix = Guess*10**np.random.uniform(-1,1,len(Guess))
            #print('Guess_mix',Guess_mix)
            [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check,fO2_g,m_H2_g] = Guess_mix
            init_cond_guess = [np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012),np.log(fO2_g),np.log(m_H2_g)] 
            check_work = 1.0
            #print ('graphite guess 4',np.log(solid_graph_g*M_melt/0.012))
            try: #for some reason randomizing inputs messes things up sometimes
                sol_guess = optimize.root(system1_graphite,init_cond_guess,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
                error_guess = np.linalg.norm(system1_graphite(sol_guess['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe))        
            except:               
                check_work=0.0

            random_loop = random_loop+1
            if (error_guess<tol) and (check_work>0.0) and (sol_guess['success'])==True:
                random_loop = 10
                ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2   = sol_guess['x']
                n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
                #print ('final graphite 4',ln_ngraphite)
                f_O2 = np.exp(ln_fO2)
                P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+f_O2
                natm = np.exp(ln_natm)
                solid_graphite = n_graphite * 0.012/M_melt
                #[XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth_Sossi(f_O2,P*1e5,T,Total_Fe)      #Sossi
                [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)             #Kress & Carmich   
        #print ('Guess worked')
                return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
                    np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
                    np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite,np.array([int_time0,int_time1,int_time2,int_time3,int_time4,int_time5]),np.exp(ln_m_H2),Total_Fe_New)
             
    
    tm3= time.time()
    int_time2 = int_time2 + tm3-tm2
    #print('sol inside 3')

    '''

    [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check] = Guess_ar2
    init_cond_guess = [np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012)] 
    sol_guess2 = optimize.root(system1_graphite,init_cond_guess,args = (d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,T,M_melt),method='lm',tol=1e-12,options={'maxiter': 10000})
    error_guess2 = np.linalg.norm(system1_graphite(sol_guess2['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,T,M_melt))
    ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite   = sol_guess2['x']
    n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
    P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)
    natm = np.exp(ln_natm)
    solid_graphite = n_graphite * 0.012/M_melt
    if error_guess2<tol:
        ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite   = sol_guess2['x']
        n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
        P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)
        natm = np.exp(ln_natm)
        solid_graphite = n_graphite * 0.012/M_melt
        #print ('Guess worked')
        print ('fixed it the second time')
        return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
            np.exp(ln_P_CH4),natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
            np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite)

    '''
    #print ('tragically, second guess failed too')
    #print('errora',errora)
    #ace = np.array([np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
    #        np.exp(ln_P_CH4),natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
    #        np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite])
    #print ('outputs second attempt',ace)
    

    lowerP = np.log10(Pfixedg) - 1.5
    upperP = np.log10(Pfixedg)+ 1.5
    #P_space = np.logspace(-5,4,100)
    P_space = np.logspace(lowerP,upperP,10)
    error_array=[]
    sol_array=[]
    [m_H2O,m_CO2,P_H2O,P_CO2,natm,P_H2,P_CH4,P_CO,f_O2,m_H2 ] = [2e-3,2e-5,1e-4,1e-10,1e20,1.0,5.0,.00010,1e-9,1e-4] 
    smallest_error = 1e20
    #init_cond1 = [np.log(m_H2O),np.log(m_CO2),np.log(P_H2O),np.log(P_CO2),np.log(natm),np.log(P_H2),np.log(P_CH4),np.log(P_CO)] 
    #init_cond1 = [np.log(m_H2O),np.log(m_CO2),np.log(P_H2O),np.log(P_CO2),np.log(P_H2),np.log(P_CH4),np.log(P_CO)] 
    init_cond1 = [np.log(m_H2O),np.log(m_CO2),np.log(P_H2O),np.log(P_CO2),np.log(P_H2),np.log(P_CH4),np.log(P_CO),np.log(n_C),np.log(f_O2),np.log(m_H2)] 
    #best_sol=[]
    #print ('in loop')   
    best_sol = optimize.root(simple_system1_linear_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Pfixed,Mfudge_fin,T,M_melt,Total_Fe),method='lm',options={'maxiter': 1000})  
    smallest_error = np.linalg.norm(simple_system1_linear_graphite(sol1['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Pfixed,Mfudge_fin,T,M_melt,Total_Fe))
    for j in range(0,len(P_space)):
        #print ('in loop')
        #print('j',j)
        #P_space[j]
        '''
        Mfudge = 0.018
        sol1 = optimize.root(simple_system1_linear,init_cond1,args = (d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,P_space[j],Mfudge),method='lm',options={'maxiter': 1000})
        error = np.linalg.norm(simple_system1_linear(sol1['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,P_space[j],Mfudge))
        '''
        try:
            Mfudge = 0.018
            sol1 = optimize.root(simple_system1_linear_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,P_space[j],Mfudge,T,M_melt,Total_Fe),method='lm',options={'maxiter': 1000})
            error = np.linalg.norm(simple_system1_linear_graphite(sol1['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,P_space[j],Mfudge,T,M_melt,Total_Fe))

            '''
            Mfudge = 0.002
            sol2 = optimize.root(simple_system1_linear,init_cond1,args = (d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,P_space[j],Mfudge),method='lm',options={'maxiter': 1000})
            error2 = np.linalg.norm(simple_system1_linear(sol2['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2,P_space[j],Mfudge))
            '''

            Mfudge = 0.002
            sol2 = optimize.root(simple_system1_linear_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,P_space[j],Mfudge,T,M_melt,Total_Fe),method='lm',options={'maxiter': 1000})
            error2 = np.linalg.norm(simple_system1_linear_graphite(sol2['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,P_space[j],Mfudge,T,M_melt,Total_Fe))

            if error<error2:
                error_array.append(error)
                sol_array.append(sol1)
                if error<smallest_error:
                    smallest_error = error
                    best_sol = sol1
                    ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2  = sol1['x']
                    init_cond1 = [ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2]
                    Mfudge_fin = 0.018
            else:
                error_array.append(error2)
                sol_array.append(sol2)
                if error2<smallest_error:
                    smallest_error = error2
                    best_sol = sol2
                    ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2 = sol2['x']
                    init_cond1 = [ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,ln_H2,ln_CH4,ln_CO,ln_ngraphite,ln_fO2,ln_m_H2]
                    Mfudge_fin = 0.002
        except:
            abc = 123

    tm4= time.time()    
    int_time3 = int_time3 + tm4-tm3
    #import pylab
    #pylab.figure()
    #pylab.semilogy(error_array)
    #pylab.show()
    #import pdb
    #pdb.set_trace()

    #ln_x_H2O,ln_x_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO = best_sol['x']
    #print('sol inside 4')

    ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2 = best_sol['x']
    Pfixed = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2) + np.exp(ln_P_CH4)+np.exp(ln_P_CO)+ np.exp(ln_fO2)
    ln_natm = np.log(Pfixed*(4*np.pi*R_planet**2*1e5)/(Mfudge_fin*gravity))
    #[ln_m_H2O,ln_m_CO2,ln_H2O,ln_CO2,natm,ln_H2,ln_CH4,ln_CO ] = [np.log(2e-3),np.log(2e-3),1.0,1.0,1e18,0.001,.001,.00010] 


    init_cond1 = [ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2] #np.log(n_C)]#[np.log(2e-3),np.log(2e-5),np.log(1e-4),np.log(1e-10),np.log(1e20),np.log(1.0),np.log(5.0),np.log(.00010)] 
    sola = optimize.root(system1_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
    errora = np.linalg.norm(system1_graphite(sola['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe))
    sol = sola
    ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2 = sol['x']
    error = errora
    errorc = []
    errorb = []

    #[ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]   = sol_guess['x']
    #pg = np.exp(np.array([ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]))
    #print ('g',pg)
    #[ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]   = sola['x']
    #pa = np.exp(np.array([ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]))
    #print ('a',pa)
    #print ('deep in loop')   
    tm5= time.time()
    int_time4 = int_time4 + tm5-tm4

    if errora>tol or sola['success']==False:
        #print ('1st failure')
        [m_H2O,m_CO2,P_H2O,P_CO2,natm,P_H2,P_CH4,P_CO,ngraphite,f_O2,m_H2 ] = [1e-3,1e-3,0.01,0.001,1e20,0.001,.001,.00010,n_C,1e-10,1e-4]
        #[m_H2O,m_CO2,P_H2O,P_CO2,natm,P_H2,P_CH4,P_CO,ngraphite ] = [1e-3,1e-9,1e-30,1e-30,1e15,0.001,1e-30,1e-30,n_C]
        init_cond1 = [np.log(m_H2O),np.log(m_CO2),np.log(P_H2O),np.log(P_CO2),np.log(natm),np.log(P_H2),np.log(P_CH4),np.log(P_CO),np.log(ngraphite),np.log(f_O2),np.log(m_H2)] 
        solb = optimize.root(system1_graphite,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
        errorb = np.linalg.norm(system1_graphite(solb['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe))

        #[ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]   = solb['x']
        #pb = np.exp(np.array([ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]))
        #print ('b',pb)
        if errorb<errora:
            sol=solb
            error=errorb
            ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2   = sol['x']
        #print('sol inside 5')
        if errorb>tol or solb['success']==False:
            #print ('2nd failure')
            #init_cond1 = np.load('ICs2.npy')
            [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check,fO2_g,m_H2_g] = Guess
            init_cond_guess = [np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012),np.log(fO2_g),np.log(m_H2_g)] 
            solc = optimize.root(system1_graphite,init_cond_guess,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
            errorc = np.linalg.norm(system1_graphite(solc['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,M_melt,Total_Fe)) 
  
            #[ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]   = solc['x']
            #pc = np.exp(np.array([ln_m_H2O_pl,ln_m_CO2_pl,ln_P_H2O_pl,ln_P_CO2_pl,ln_natm_pl,ln_P_H2_pl,ln_P_CH4_pl,ln_P_CO_pl,ln_ngraphite_pl]))
            #print ('c',pc)
            if (errorc<errorb)and(errorc<errora):
                sol=solc
                error = errorc
                ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite,ln_fO2,ln_m_H2   = sol['x']
        

    #np.save('ICs2',sol['x'])
    #print ('errors',error_guess,errora,errorb,errorc)



    #print ('final',np.exp(np.array([ln_m_H2O,ln_m_CO2,ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_ngraphite])))

    #y = ln_x_H2O,ln_x_CO2,ln_P_H2O,ln_P_CO2,ln_alphaG,ln_P_H2,ln_P_CH4,ln_P_CO
    n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
    P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+np.exp(ln_fO2)
    natm = np.exp(ln_natm)
    f_O2= np.exp(ln_fO2)
    #n_graphite = natm *(n_C/natm - (np.exp(ln_P_CH4)/P + np.exp(ln_P_CO2)/P + np.exp(ln_P_CO)/P) - (1/natm)*np.exp(ln_m_CO2)*M_melt/(M_CO2/1000.0))
    solid_graphite = n_graphite * 0.012/M_melt
    #[XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth_Sossi(f_O2,P*1e5,T,Total_Fe)      #Sossi
    [XFe2O3_over_XFeO,Total_Fe_New] = Iron_speciation_smooth(f_O2,P*1e5,T,Total_Fe)             #Kress & Carmich  
    #return_out = np.array([np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
    #        np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
    #        np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite])
    #print('long return',return_out)
    #import pdb
    #pdb.set_trace()
    #print('sol inside 6 - ret')
        
    tm6= time.time()

    
    
    
   
    int_time5 = int_time5 + tm6-tm5
    #print (tm1,tm2,tm3,int_time0)
    return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
            np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_m_CO2),np.exp(ln_m_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
            np.exp(ln_P_CH4)/P,f_O2/P ,solid_graphite,np.array([int_time0,int_time1,int_time2,int_time3,int_time4,int_time5]),np.exp(ln_m_H2),Total_Fe_New)


def solve_gases_ONLY(T_input,y4_input,mf_C,mf_H,Guess,Guess_ar2,M_melt,Total_Fe): ##now just total mass input
    #global int_time0,int_time1,int_time2,int_time3,int_time4,int_time5

    #import pdb
    #pdb.set_trace()
    int_time0 = 0
    int_time1 = 0
    int_time2 = 0
    int_time3 = 0
    int_time4 = 0
    int_time5 = 0

    tm0= time.time()

    # calculate mol fraction of CO2 and H2O in the magma

    if T_input <1000.0: #quench temp, breaks things
        T = 1000.0
    else:
        T = T_input
    #T = T_input

    # equilibrium constants
    # made with Nasa thermodynamic database (Burcat database)
    K1 = np.exp(-29755.11319228574/T+6.652127716162998)
    K2 = np.exp(-33979.12369002451/T+10.418882755464773)
    K3 = np.exp(-96444.47151911151/T+0.22260815074146403)

    #constants
    #C1 = K1/f_O2**0.5
    #C2 = K2/f_O2**0.5
    #C3 = K3/f_O2**2

    # now use the solution of the simple system to solve the
    # harder problem. I will try to solve it two different ways to
    # make sure I avoid errors.

    # error tolerance
    tol = 1e-5
    [x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-2,1e-4,100.,10.,0.1,10.,1.,10.] 
    [x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-7,1e-3,0.01,10.,0.1,0.001,.001,.00010] 
    #[x_H2O,x_CO2,P_H2O,P_CO2,alphaG,P_H2,P_CH4,P_CO ] = [1e-7,1e-8,0.01,0.001,0.1,0.001,.001,.00010]
    #import pdb
    #pdb.set_trace()

    n_H = mf_H/0.001 # mass fraction H -> mol H
    n_C = mf_C/0.012 # mass fraction C -> mol C
    n_O = y4_input/0.016
    #first see if can solve using guess
    #print(Guess)
    #import pdb
    #pdb.set_trace()

    [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check,f_O2_g,m_H2_g] = Guess
    #print ('graphite guess 1',np.log(solid_graph_g*M_melt/0.012))
    init_cond_guess = [np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(f_O2_g)] 
    #print ('always')
    #print ('init_cond_guess',init_cond_guess)

    try:
        #import pdb
        #pdb.set_trace() #d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,Tinput,Total_Fe
        sol_guess = optimize.root(system1_GASONLY,init_cond_guess,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
        error_guess = np.linalg.norm(system1_GASONLY(sol_guess['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe))
        #ln_H2O,ln_CO2,ln_natm,ln_H2,ln_CH4,ln_CO,ln_fO2   = sol_guess['x']
        ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2 = sol_guess['x']
        f_O2 = np.exp(ln_fO2)
        P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4) + f_O2
        natm = np.exp(ln_natm)
        if error_guess<tol:
            ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2 = sol_guess['x']
            P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+ np.exp(ln_fO2)
            natm = np.exp(ln_natm)
            f_O2 = np.exp(ln_fO2)
            return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
                np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
                np.exp(ln_P_CH4)/P,f_O2/P)
    except:
        abc=1


    Pfixedg = PH2O_g+PH2_g+PCO2_g+PCO_g+PCH4_g+f_O2_g
    Mfudge = (PH2O_g*0.018+PH2_g*0.02+PCO2_g*0.044+PCO_g*0.028+PCH4_g*0.016+f_O2_g*0.032)/(PH2O_g+PH2_g+PCO2_g+PCO_g+PCH4_g+f_O2_g)
    init_cond1 = [np.log(P_H2O),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(f_O2_g)] 
    #print ('graphite guess 2',np.log(solid_graph_g*M_melt/0.012))
    sol1 = optimize.root(system1_GASONLY,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe),method='lm',options={'maxiter': 1000})
    error = np.linalg.norm(system1_GASONLY(sol1['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe))
    
    ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2 = sol1['x']
    #print ('final graphite 2',ln_ngraphite)
    Pfixed = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2) + np.exp(ln_P_CH4)+np.exp(ln_P_CO)+np.exp(ln_fO2)
    Mfudge_fin = (np.exp(ln_P_H2O)*0.018+np.exp(ln_P_H2)*0.02+np.exp(ln_P_CO2)*0.044+np.exp(ln_P_CO)*0.028+np.exp(ln_P_CH4)*0.016+np.exp(ln_fO2)*0.032)/Pfixed
    ln_natm = np.log(Pfixed*(4*np.pi*R_planet**2*1e5)/(Mfudge_fin*gravity))
    #log_XFe2O3_over_XFeO,Total_Fe_New = Iron_speciation(np.exp(ln_fO2),Pfixed*1e5,T,Total_Fe)

    init_cond1 = [ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2]
    #print ('graphite guess 3',ln_ngraphite)
    sola = optimize.root(system1_GASONLY,init_cond1,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})
    errora = np.linalg.norm(system1_GASONLY(sola['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe))
    
    #import pdb
    #pdb.set_trace()
    #print ('second attempt')
    if errora<tol:
        ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2   = sola['x']
        P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+ np.exp(ln_fO2)
        f_O2 = np.exp(ln_fO2)
        natm = np.exp(ln_natm)
        return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
            np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
            np.exp(ln_P_CH4)/P,f_O2/P)
    else:
        Guess_mix = np.copy(Guess)
        random_loop = 0
        while random_loop <5:
            #import pdb
            #pdb.set_trace()
            #print('Guess',Guess)
            Guess_mix = Guess*10**np.random.uniform(-1,1,len(Guess))
            #print('Guess_mix',Guess_mix)
            [PH2O_g, PH2_g, PCO2_g, PCO_g, PCH4_g, natm_g, m_CO2_g, m_H2O_g, mH2O_g,mH2_g,mCO2_g,mCO_g,mCH4_g,mO2_g,solid_graph_g,grp_check,fO2_g,m_H2_g] = Guess_mix

            init_cond_guess = [np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(fO2_g)] 
            check_work = 1.0
            #print ('graphite guess 4',np.log(solid_graph_g*M_melt/0.012))
            try: #for some reason randomizing inputs messes things up sometimes
                sol_guess = optimize.root(system1_GASONLY,init_cond_guess,args = (d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe),method='lm',tol=1e-12,options={'maxiter': 10000})

                error_guess = np.linalg.norm(system1_GASONLY(sol_guess['x'],d_H2O,a_H2O,K1,K2,K3,a_CO2,n_H,n_C,n_O,T,Total_Fe))        
            except:               
                check_work=0.0

            random_loop = random_loop+1
            if (error_guess<tol) and (check_work>0.0) and (sol_guess['success'])==True:
                random_loop = 10
                ln_P_H2O,ln_P_CO2,ln_natm,ln_P_H2,ln_P_CH4,ln_P_CO,ln_fO2   = sol_guess['x']
                n_graphite = np.exp(ln_ngraphite)#0.0#just_the_graphite(sol['x'],d_H2O,a_H2O,C1,C2,C3,a_CO2,n_H,n_C,f_O2)
                #print ('final graphite 4',ln_ngraphite)
                f_O2 = np.exp(ln_fO2)
                P = np.exp(ln_P_H2O)+np.exp(ln_P_H2)+np.exp(ln_P_CO2)+np.exp(ln_P_CO)+ np.exp(ln_P_CH4)+f_O2
                natm = np.exp(ln_natm)
                return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
                    np.exp(ln_P_CH4),f_O2,natm,np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
                    np.exp(ln_P_CH4)/P,f_O2/P)





def Call_MO_Atmo_equil(T_surface, y4_input,mf_C_input,mf_H_input,M_melt_input,Guess,Guess_ar2,Total_Fe): #not making use of ICs from last run
    #global int_time0,int_time1,int_time2,int_time3,int_time4,int_time5
    #print ('T_surface,y4_input,mf_C_input,mf_H_input,Guess,Guess_ar2,M_melt_input,Total_Fe')
    #print (T_surface,y4_input,mf_C_input,mf_H_input,Guess,Guess_ar2,M_melt_input,Total_Fe)
    try:
        #print ('inputs ded')
        if M_melt_input==0:
            #import pdb
            #pdb.set_trace()
            [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, mH2O,mH2,mCO2,mCO,mCH4,mO2] = solve_gases_ONLY(T_surface,y4_input,mf_C_input,mf_H_input,Guess,Guess_ar2,M_melt_input,Total_Fe)
            new_additions =np.array([1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])
            return [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, 0.0, 0.0, mH2O,mH2,mCO2,mCO,mCH4,mO2,0.0,0.0,new_additions,0.0,Total_Fe] ## returns mass fraction H2O and CO2 in melt

        else:
        #print ([T_surface,y4_input,mf_C_input,mf_H_input,Guess,Guess_ar2,M_melt_input,Total_Fe])
            [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,new_additions,m_H2,Total_Fe_New] = solve_gases_new(T_surface,y4_input,mf_C_input,mf_H_input,Guess,Guess_ar2,M_melt_input,Total_Fe)
        #print ('Okkk')
    #f_O2 = fO2_input
    except:
        [PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2] = Guess
        Total_Fe_New = Total_Fe
        new_additions =np.array([1e-10,1e-10,1e-10,1e-10,1e-10,1e-10])
        #print ('lol ded')
    P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
    n_C_atm = natm *(PCH4/P + PCO2/P + PCO/P)
    n_C_diss =  m_CO2 * M_melt_input/(M_CO2/1000.0)
    n_C_graphite = solid_graph/ (0.012/M_melt_input)
    n_H_atm = natm * (4*PCH4/P + 2 * PH2O/P + 2 * PH2/P) 
    n_H_diss =  2 * m_H2O*M_melt_input/(M_H2O/1000.0)+ 2 * m_H2*M_melt_input/(M_H2/1000.0)
    log10_K1 = 40.07639 - 2.53932e-2 * T_surface + 5.27096e-6*T_surface**2 + 0.0267 * (P - 1 )/T_surface
    log10_K2 = - 6.24763 - 282.56/T_surface - 0.119242 * (P - 1000)/T_surface
    gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*PO2)/(1+(10**log10_K1)*(10**log10_K2)*PO2) 
    gXCO2_melt = grph_set*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction
    graph_check = gXCO2_melt*M_melt_input/(M_CO2/1000.0)
    #new_additions = np.array([int_time0,int_time1,int_time2,int_time3,int_time4,int_time5])
    #print('new_additions',new_additions)
    return [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,Total_Fe_New] ## returns mass fraction H2O and CO2 in melt


def buffer_fO2(T,Press,redox_buffer): # Estimating oxygen fugacity for common buffers (FMQ, IW, and MH). T in K, P in bar
    if redox_buffer == 'FMQ':
        [A,B,C] = [25738.0, 9.0, 0.092]
    elif redox_buffer == 'IW':
        [A,B,C] = [27215 ,6.57 ,0.0552]
    elif redox_buffer == 'MH':
        [A,B,C] = [25700.6,14.558,0.019] 
    else:
        #print ('error, no such redox buffer')
        return -999
    return 10**(-A/T + B + C*(Press-1)/T)


PH2O_ar=[]
PH2_ar=[]
PCO2_ar=[]
PCO_ar=[]
PCH4_ar=[]
PO2_ar = []
mCO2_ar=[]
mH2O_ar=[]
solid_graph_array = []

XFeO_ar = []
XFe2O3_ar = []
Metal_frac_ar = []


nFeO_ar= []
nFeO1_5_ar = []
Metal_n_ar = []

arr_n_C_atm=[]
arr_n_C_diss=[]
arr_n_C_graphite=[]
arr_n_H_atm=[]
arr_n_H_diss=[]
arr_n_H2O_diss=[]

m_H2_ar = []

arr_n_O_diss = []
arr_n_O_atm = []
arr_n_O_Fe = []

arr_n_FeO1_5 = []#.append( (F_FeO1_5 * M_melt * 56./(1.5*16+56))/0.056 )
arr_n_FeO = []#.append( (F_FeO * M_melt * 56./(16+56))/0.056)
arr_n_FeMetal = []#.append( F_metal*M_melt )/0.056)

graph_check_ar = []

mf_C = 102./1e6
mf_H = 90./1e6
M_melt = 0.5*4e24



T_surface = 1500.0+273
PO2 = 1e-10
P = 1.0
log10_K1 = 40.07639 - 2.53932e-2 * T_surface + 5.27096e-6*T_surface**2 + 0.0267 * (P - 1 )/T_surface
log10_K2 = - 6.24763 - 282.56/T_surface - 0.119242 * (P - 1000)/T_surface
gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*PO2)/(1+(10**log10_K1)*(10**log10_K2)*PO2) 
gXCO2_melt = grph_set*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction
#graph_check = gXCO2_melt*M_melt/(M_CO2/1000.0)

Guess_ar = np.array([1., 1., 1., 1., 1., 1e20, mf_H, mf_C, .1,.1,.1,.1,.1,.5,1e-3,gXCO2_melt,1e-10,1e-4])

Guess_ar = np.array([1000., 1000., 1000., 1000., 100., 1e20, mf_H, mf_C, .1,.1,.1,.1,.1,.5,1e-3,gXCO2_melt,1e-10,1e-4])
Guess_ar2 = np.copy(Guess_ar)

y4_ar = np.logspace(10,23,1000)
fO2_ar = np.copy(y4_ar)
t0 = time.time()
mf_H_ar = np.logspace(-5,-1,1000)


#fO2_ar =np.logspace(-14,-3,1000)
#mf_H_ar = np.copy(fO2_ar)

Total_Fe = 0.06
for i in range(0,len(mf_H_ar)):
    print('i',i,mf_H_ar[i])
    #print (Guess_ar)
    try:
        [PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,new_additions,m_H2,Total_Fe_New] = solve_gases_new(1500.0+273,1e22,mf_C,mf_H_ar[i],Guess_ar,Guess_ar2,M_melt,Total_Fe)
        '''PH2O_ar.append(PH2O)
        PH2_ar.append(PH2)
        PCO2_ar.append(PCO2)
        PCO_ar.append(PCO)
        PCH4_ar.append(PCH4)
        PO2_ar.append(PO2)
        mCO2_ar.append(m_CO2)
        mH2O_ar.append(m_H2O)
        m_H2_ar.append(m_H2)
        solid_graph_array.append(solid_graph)'''
    except Exception as e:
        print(e)
        abc = 2
        #print("CRASH IN solve_gases_new:", e)
        # Either skip this iteration:
        #continue


    #print (1500.0+273,1e21,mf_C*M_melt,mf_H_ar[i]*M_melt,Guess_ar,Guess_ar2,0.0,Total_Fe)
    #[PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, mH2O,mH2,mCO2,mCO,mCH4,mO2] = solve_gases_ONLY(1500.0+273,1e21,mf_C*M_melt,mf_H_ar[i]*M_melt,Guess_ar,Guess_ar2,0.0,Total_Fe)
    #try:
    #    [PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, mH2O,mH2,mCO2,mCO,mCH4,mO2] = solve_gases_ONLY(1500.0+273,1e21,mf_C*M_melt,mf_H_ar[i]*M_melt,Guess_ar,Guess_ar2,0.0,Total_Fe)
    #    m_CO2 = 0.0
    #    m_H2O = 0.0
    #    m_H2 = 0.0
    #    solid_graph = 0.0
    #except Exception as e:
    #    print (e)
    #    abc = 2

    PH2O_ar.append(PH2O)
    PH2_ar.append(PH2)
    PCO2_ar.append(PCO2)
    PCO_ar.append(PCO)
    PCH4_ar.append(PCH4)
    PO2_ar.append(PO2)
    mCO2_ar.append(m_CO2)
    mH2O_ar.append(m_H2O)
    m_H2_ar.append(m_H2)
    solid_graph_array.append(solid_graph)


    P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
    n_C_atm = natm *(PCH4/P + PCO2/P + PCO/P)
    n_C_diss =  m_CO2 * M_melt/(M_CO2/1000.0)
#    n_C_graphite = solid_graph/ (0.012/M_melt)
    n_C_graphite = M_melt*solid_graph/0.012 #/ (0.012/M_melt)
    n_H_atm = natm * (4*PCH4/P + 2 * PH2O/P + 2 * PH2/P) 
    n_H_diss =  2 * m_H2O*M_melt/(M_H2O/1000.0)+ 2 * m_H2*M_melt/(M_H2/1000.0)
    n_H2O_diss = 2 * m_H2O*M_melt/(M_H2O/1000.0)



    
    #[XFe2O3_over_XFeO,Total_Fe_New_check] = Iron_speciation_smooth_Sossi(PO2,P*1e5,1500.0+273,Total_Fe)              #Sossi
    [XFe2O3_over_XFeO,Total_Fe_New_check] = Iron_speciation_smooth(PO2,P*1e5,1500.0+273,Total_Fe)         #Kress & Carmich
    m_sil_og = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + Total_Fe * (56.0+16.0) 
    F_FeO_og = Total_Fe * (56.0 + 16.0) / m_sil_og 
    n_FeO_og = F_FeO_og * M_melt * (56./(16+56))/0.056

    F_TotalFeO_new = Total_Fe_New_check * (56.0 + 16.0) / m_sil_og 
    n_TotalFeO_new = F_TotalFeO_new * M_melt * (56./(16+56))/0.056
    Metal_n = n_FeO_og - n_TotalFeO_new

    #nFeO + nFeO1.5 = n_TotalFeO_new
    #2*np.exp(log_XFe2O3_over_XFeO) = nFeO1.5/nFeO
    #nFeO1.5 = nFeO * 2 * np.exp(log_XFe2O3_over_XFeO)
    #nFeO + nFeO * 2 * np.exp(log_XFe2O3_over_XFeO) = n_TotalFeO_new
    #nFeO ( 1 + 2 * np.exp(log_XFe2O3_over_XFeO)) = n_TotalFeO_new
    nFeO = n_TotalFeO_new / ( 1 + 2 * (XFe2O3_over_XFeO)) 
    nFeO1_5 = n_TotalFeO_new - nFeO
  
    #XFe2O3 =XFeO * np.exp(log_XFe2O3_over_XFeO)
    
    F_FeO1_5 = (nFeO1_5*(56.0+1.5*16.0)/1000.0)/M_melt
    F_FeO = (nFeO*(56.0+16.0)/1000.0)/M_melt
    F_Fe = (Metal_n*56/1000.0)/M_melt

    nFeO_ar.append(nFeO)
    nFeO1_5_ar.append(nFeO1_5)
    Metal_n_ar.append(Metal_n)

    '''
    F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
    F_FeO = XFeO * (56.0 + 16.0) / m_sil 
    
    Metal_frac = Total_Fe - Total_Fe_New_check
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFeO * (56.0+16.0) + XFe2O3*(2*56.0+3*16.0) #+ Metal_frac*56.0
    F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
    F_FeO = XFeO * (56.0 + 16.0) / m_sil 
    F_metal = Metal_frac*56/m_sil

    #XFeO_ar.append(XFeO)
    #XFe2O3_ar.append(XFe2O3)
    #Metal_frac_ar.append(Metal_frac)
    '''
    n_O_diss = 1 * m_H2O*M_melt/(M_H2O/1000.0) + 2*m_CO2 * M_melt/(M_CO2/1000.0)
    n_O_atm = natm * (PH2O/P + 2*PCO2/P + PCO/P + 2*PO2/P)
    n_O_Fe = (F_FeO1_5 * M_melt * 0.5*16/(1.5*16+56))/0.016

    arr_n_FeO1_5.append( (F_FeO1_5 * M_melt * 56./(1.5*16+56))/0.056 )
    arr_n_FeO.append( (F_FeO * M_melt * 56./(16+56))/0.056)
    arr_n_FeMetal.append( (F_Fe*M_melt )/0.056)

    arr_n_O_diss.append(n_O_diss)
    arr_n_O_atm.append(n_O_atm)
    arr_n_O_Fe.append(n_O_Fe)

    arr_n_C_atm.append(n_C_atm)
    arr_n_C_diss.append(n_C_diss)
    arr_n_C_graphite.append(n_C_graphite)
    arr_n_H_atm.append(n_H_atm)
    arr_n_H_diss.append(n_H_diss)
    arr_n_H2O_diss.append(n_H2O_diss)


    fO2_ar[i] = PO2
    T = 1500.0+273
    log10_K1 = 40.07639 - 2.53932e-2 * T + 5.27096e-6*T**2 + 0.0267 * (P - 1 )/T
    log10_K2 = - 6.24763 - 282.56/T - 0.119242 * (P - 1000)/T
    gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*PO2)/(1+(10**log10_K1)*(10**log10_K2)*PO2) 
    gXCO2_melt = grph_set*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction
    graph_check_ar.append(gXCO2_melt*M_melt/(M_CO2/1000.0))

    Guess_ar = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,gXCO2_melt*M_melt/(M_CO2/1000.0),PO2,m_H2])
    Guess_ar2 = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,gXCO2_melt*M_melt/(M_CO2/1000.0),PO2,m_H2])



'''
XFeO_ar = np.array(nFeO_ar)
XFe2O3_ar = np.array(nFeO1_5_ar)
Metal_frac_ar = np.array(Metal_n_ar)


t1 = time.time()

#import pdb
#pdb.set_trace()

n_H = mf_H*M_melt/0.001
n_C = mf_C*M_melt/0.012
n_O = y4_ar/0.016

P_ar = np.array(PH2O_ar) + np.array(PH2_ar) + np.array(PCO2_ar) + np.array(PCO_ar) + np.array(PCH4_ar) + np.array(PO2_ar)


new_O2 = np.copy(fO2_ar)
for i in range(0,len(new_O2)):
    new_O2[i] = np.log10(fO2_ar[i]) - np.log10(buffer_fO2(1500+273,P_ar[i],'IW'))
#fO2_ar = new_O2

print ('total loop time',t1-t0)
#print('int_times',int_time0,int_time1,int_time2,int_time3,int_time4,int_time5)
y4_ar = np.copy(mf_H_ar)
import pylab
pylab.figure()
pylab.subplot(2,1,1)
pylab.loglog(y4_ar,PH2O_ar,'b',label='H2O')
pylab.loglog(y4_ar,PH2_ar,'c',label='H2')
pylab.loglog(y4_ar,PCO2_ar,'r',label='CO2')
pylab.loglog(y4_ar,PCO_ar,'g',label='CO')
pylab.loglog(y4_ar,PO2_ar,'y',label='O2')
pylab.loglog(y4_ar,P_ar,'k--',label='P_total')
pylab.loglog(y4_ar,PCH4_ar,'m',label='CH4')
pylab.subplot(2,1,2)
pylab.loglog(y4_ar,mCO2_ar,'r',label='mCO2')
pylab.loglog(y4_ar,mH2O_ar,'b',label='mH2O')
pylab.loglog(y4_ar,solid_graph_array,'k--',label='Graphite')

pylab.figure()
pylab.subplot(2,1,1)
pylab.loglog(y4_ar,arr_n_C_atm,'b')
pylab.loglog(y4_ar,arr_n_C_diss,'g')
pylab.loglog(y4_ar,arr_n_C_graphite,'r')
pylab.loglog(y4_ar,graph_check_ar,'m--')
pylab.loglog(y4_ar,np.array(arr_n_C_graphite)*0+n_C,'c')
pylab.loglog(y4_ar,np.array(arr_n_C_graphite)+np.array(arr_n_C_diss)+np.array(arr_n_C_atm),'k--')
pylab.subplot(2,1,2)
pylab.loglog(y4_ar,arr_n_H_atm,'b')
pylab.loglog(y4_ar,arr_n_H_diss,'g')
pylab.loglog(y4_ar,arr_n_H2O_diss,'y--')
pylab.loglog(y4_ar,np.array(arr_n_H_diss)*0+n_H,'c')
pylab.loglog(y4_ar,np.array(arr_n_H_diss)+np.array(arr_n_H_atm),'k--')

pylab.figure()
pylab.subplot(3,1,1)
pylab.loglog(y4_ar,XFe2O3_ar,'b',label='XFeO_1pt5_ar')
pylab.loglog(y4_ar,XFeO_ar,'g',label='XFeO_ar')
pylab.loglog(y4_ar,Metal_frac_ar,'r',label='Metal_frac_ar')
pylab.loglog(y4_ar,np.array(Metal_frac_ar)+np.array(XFeO_ar)+np.array(XFe2O3_ar),'k--')
pylab.ylabel('metal frac')
pylab.legend()
pylab.subplot(3,1,2)
pylab.loglog(y4_ar,arr_n_O_diss,label='n_O_diss')
pylab.loglog(y4_ar,arr_n_O_atm,label='n_O_atm')
pylab.loglog(y4_ar,arr_n_O_Fe,label='n_O_Fe')
pylab.loglog(y4_ar,np.array(arr_n_O_Fe)+np.array(arr_n_O_atm)+np.array(arr_n_O_diss),label='Sum')
pylab.loglog(y4_ar,n_O,label='Total',linestyle='--')
pylab.legend()
pylab.subplot(3,1,3)
pylab.loglog(y4_ar,arr_n_FeO1_5,'r',label='FeO1.5')
pylab.loglog(y4_ar,arr_n_FeO,'g',label='FeO')
pylab.loglog(y4_ar,arr_n_FeMetal,'b',label='Fe')
pylab.loglog(y4_ar,np.array(arr_n_FeO1_5) + np.array(arr_n_FeO) + np.array(arr_n_FeMetal),'k--')
pylab.legend()

pylab.figure()
pylab.subplot(3,1,1)
pylab.semilogy(new_O2,PH2O_ar,'b',label='H2O')
pylab.semilogy(new_O2,PH2_ar,'r',label='H2')
pylab.semilogy(new_O2,PCO2_ar,'grey',label='CO2')
pylab.semilogy(new_O2,PCO_ar,'k',label='CO')
pylab.semilogy(new_O2,PCH4_ar,'y',label='CH4')
pylab.semilogy(new_O2,P_ar,'k--',label='Ptotal')
#pylab.xlim([-6,4])
#pylab.ylim([0.01,250])
pylab.legend()
pylab.subplot(3,1,2)
pylab.semilogy(new_O2,1e6*np.array(mCO2_ar),'grey',label='mCO2')
pylab.semilogy(new_O2,1e6*np.array(mH2O_ar),'b',label='mH2O')
pylab.semilogy(new_O2,1e6*np.array(solid_graph_array),'k--',label='Graphite')
pylab.semilogy(new_O2,1e6*np.array(m_H2_ar),'r',label='mH2')
#pylab.xlim([-6,4])
#pylab.ylim([0.1,1000])
pylab.legend()
pylab.subplot(3,1,3)
pylab.semilogx(y4_ar,new_O2,label='oxygen fugacity')
#pylab.xlim([-6,4])
#pylab.ylim([0.1,1000])
pylab.legend()
pylab.show()

'''
#import pdb
#pdb.set_trace()
#raise Exception



