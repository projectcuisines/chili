import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import pdb
import scipy.interpolate
from numba import jit
import time



T_surf_grid = np.linspace(250,3000,120)
Te_grid = np.linspace(150,350,10)
M_H2O_grid_new = np.logspace(15,23,32)
M_H2_grid_new = np.logspace(15,23,8)
M_CO2_grid_new = np.logspace(15,22,14) 
M_CO_grid_new = np.logspace(15,22,8) 
Ev_array=np.load("Ev_array_improved_corrected_CO_as_N2.npy") #load precomputed climate grid #Ev_array_improved_corrected_CO_as_N2.npy for original climate grid 
#Ev_array=np.load("Ev_array_improved_corrected_CO_as_N2_deepIsotherm.npy") #load precomputed DEEP ISOTHERMAL climate grid as end-member test
#Ev_array=np.load("Clima_grid_rearranged.npy") #Clima_grid_rearranged.npy for clima comparison

# change from Ts,H2O,CO2,CO,Te,H2 to Ts,Te,H2O,CO2,CO,H2
Ev_array_load = np.moveaxis(Ev_array,[4],[1])
OLR_hybrid_FIX = np.log10(Ev_array_load[:,:,:,:,:,:,0])
water_frac_multi_new=Ev_array_load[:,:,:,:,:,:,6]
fH2O_new = Ev_array_load[:,:,:,:,:,:,3] 
##commented these out for proper escape
denom = (Ev_array_load[:,:,:,:,:,:,1]+Ev_array_load[:,:,:,:,:,:,2]+Ev_array_load[:,:,:,:,:,:,3]+Ev_array_load[:,:,:,:,:,:,4]+Ev_array_load[:,:,:,:,:,:,5])
#fH2O_new = (Ev_array_load[:,:,:,:,:,:,3]+Ev_array_load[:,:,:,:,:,:,5])/ denom

fH2O_new = np.log10(fH2O_new/denom)
fH2_new = np.log10(Ev_array_load[:,:,:,:,:,:,5]/denom)
fCO_new = np.log10(Ev_array_load[:,:,:,:,:,:,1]/denom)
fCO2_new = np.log10(Ev_array_load[:,:,:,:,:,:,4]/denom)
fN2_new = np.log10(Ev_array_load[:,:,:,:,:,:,2]/denom)
#T_surf_grid = np.linspace(250,4000,200)
#Te_grid = np.copy(Te_grid)
P_H2O_grid_new = M_H2O_grid_new
P_CO2_grid_new = M_CO2_grid_new
P_CO_grid_new = M_CO_grid_new
P_H2_grid_new = M_H2_grid_new

#
#    fCO = f_i[0,:]
#    fN2 = f_i[1,:]
#    fH2O = f_i[2,:]
#    fCO2 = f_i[3,:]
#    fH2 = f_i[4,:]


@jit(nopython=True)
def my_interp(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
   
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2 #98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2 #4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2

    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD

    if Tsurf < np.min(T_surf_grid):
#         return (Tsurf/np.min(T_surf_grid)) * 10**answer #default
         return (Tsurf/np.min(T_surf_grid))**4 * 10**answer #not as cold hopefully, works OK for low CO2, need to consider Co2 condensation
    elif Tsurf > np.max(T_surf_grid):
         return 1000*5.678e-8*(Tsurf-3000.0)**4 + ((Tsurf/np.max(T_surf_grid))**5) * 10**answer   
    else:
        return 10**answer
        
#print ('compare',my_interp(160,230,1e5,1e7),OLR_True(160,230,1e5,1e7,10))
'''
inter4D_new =scipy.interpolate.RegularGridInterpolator((T_surf_grid,Te_grid,P_H2O_grid_new,P_CO2_grid_new,P_CO_grid_new,P_H2_grid_new),OLR_hybrid_FIX,method='linear',bounds_error=False,fill_value = None)

print ('Compare')
print (my_interp(1300,230,1e21,1e19,1e16,1e16),10**inter4D_new((1300,230,1e21,1e19,1e16,1e16)))
'''


@jit(nopython=True)
def my_water_frac(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
    
    if Tsurf >=647: # fudge because es(Ts) is clearly broken in some way for high CO2, high H2O atmospheres
        return 1.0    
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2 #98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2



    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD
   
    if answer <0:
        return 0.0
    elif answer>1:
        return 1.0
    
    return answer

@jit(nopython=True)
def my_fH2O(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2




    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD


    if H2O_index ==0:
        return -9.9

    if Tsurf < np.min(T_surf_grid): ## added fudge to zero out escape
        #print('do nothing')
        ###return np.exp(15*(-1+Tsurf/np.min(T_surf_grid))) * 10**answer 
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer #looks like a decent extrapolation, need to see what it does to false positives

#    print (Te,answer) extrapolating fH2O low Tskin
    #if Te < np.min(Te_grid): #extrapolate down to lower Te
    #     new_fh2o = answer*np.exp((2.834e6/461.5)*(-1/Te + 1/np.min(Te_grid)))
#         new_fh2o = answer*np.exp((-1/Te + 1/np.min(Te_grid)))
#         print ('extrap',new_fh2o)#np.exp((Te/np.min(Te_grid))-1) * answer)
    #     return new_fh2o#np.exp((Te/np.min(Te_grid))-1.0) * answer

    #if answer <0:
    #    return 0.0
    #elif answer>1:
    #    return 1.0
    #return answer    
    
    return 10**answer




@jit(nopython=True)
def my_fH2(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):

                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2




    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = fH2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = fH2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = fH2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = fH2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD

    if H2_index ==0:
        return -9.9

    if Tsurf < np.min(T_surf_grid): ## added fudge to zero out escape
        #print('do nothing')
        ###return np.exp(15*(-1+Tsurf/np.min(T_surf_grid))) * 10**answer 
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer #looks like a decent extrapolation, need to see what it does to false positives

#    print (Te,answer) extrapolating fH2O low Tskin
    #if Te < np.min(Te_grid): #extrapolate down to lower Te
    #     new_fh2o = answer*np.exp((2.834e6/461.5)*(-1/Te + 1/np.min(Te_grid)))
#         new_fh2o = answer*np.exp((-1/Te + 1/np.min(Te_grid)))
#         print ('extrap',new_fh2o)#np.exp((Te/np.min(Te_grid))-1) * answer)
    #     return new_fh2o#np.exp((Te/np.min(Te_grid))-1.0) * answer

    #if answer <0:
    #    return 0.0
    #elif answer>1:
    #    return 1.0
    #return answer    
    
    return 10**answer



@jit(nopython=True)
def my_fCO(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):

                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2




    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = fCO_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = fCO_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = fCO_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = fCO_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD


    if CO_index ==0:
        return -9.9

    if Tsurf < np.min(T_surf_grid): ## added fudge to zero out escape
        #print('do nothing')
        ###return np.exp(15*(-1+Tsurf/np.min(T_surf_grid))) * 10**answer 
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer #looks like a decent extrapolation, need to see what it does to false positives

#    print (Te,answer) extrapolating fH2O low Tskin
    #if Te < np.min(Te_grid): #extrapolate down to lower Te
    #     new_fh2o = answer*np.exp((2.834e6/461.5)*(-1/Te + 1/np.min(Te_grid)))
#         new_fh2o = answer*np.exp((-1/Te + 1/np.min(Te_grid)))
#         print ('extrap',new_fh2o)#np.exp((Te/np.min(Te_grid))-1) * answer)
    #     return new_fh2o#np.exp((Te/np.min(Te_grid))-1.0) * answer

    #if answer <0:
    #    return 0.0
    #elif answer>1:
    #    return 1.0
    #return answer    
    
    return 10**answer



@jit(nopython=True)
def my_fCO2(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):

                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2




    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = fCO2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = fCO2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = fCO2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = fCO2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fCO2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD


    if CO2_index ==0:
        return -9.9

    if Tsurf < np.min(T_surf_grid): ## added fudge to zero out escape
        #print('do nothing')
        ###return np.exp(15*(-1+Tsurf/np.min(T_surf_grid))) * 10**answer 
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer #looks like a decent extrapolation, need to see what it does to false positives

#    print (Te,answer) extrapolating fH2O low Tskin
    #if Te < np.min(Te_grid): #extrapolate down to lower Te
    #     new_fh2o = answer*np.exp((2.834e6/461.5)*(-1/Te + 1/np.min(Te_grid)))
#         new_fh2o = answer*np.exp((-1/Te + 1/np.min(Te_grid)))
#         print ('extrap',new_fh2o)#np.exp((Te/np.min(Te_grid))-1) * answer)
    #     return new_fh2o#np.exp((Te/np.min(Te_grid))-1.0) * answer

    #if answer <0:
    #    return 0.0
    #elif answer>1:
    #    return 1.0
    #return answer    
    
    return 10**answer


@jit(nopython=True)
def my_fN2(Tsurf,Te,PH2O,PCO2,PCO,PH2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):

                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2




    if PH2 <= np.min(P_H2_grid_new):
        Actual_H2 = np.min(P_H2_grid_new) 
        H2_index = 0
    elif PH2 >= np.max(P_H2_grid_new):
        Actual_H2 = np.max(P_H2_grid_new) 
        H2_index = len(P_H2_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2_grid_new)):
            if (P_H2_grid_new[i]>PH2)and(P_H2_grid_new[i-1]<=PH2):
                H2_index = i-1
                Actual_H2 = PH2

    if PCO <= np.min(P_CO_grid_new):
        Actual_CO = np.min(P_CO_grid_new) 
        CO_index = 0
    elif PCO >= np.max(P_CO_grid_new):
        Actual_CO = np.max(P_CO_grid_new) 
        CO_index = len(P_CO_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO_grid_new)):
            if (P_CO_grid_new[i]>PCO)and(P_CO_grid_new[i-1]<=PCO):
                CO_index = i-1
                Actual_CO = PCO


    #strt modify
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]
    intCO = P_CO_grid_new[1+CO_index] - P_CO_grid_new[CO_index]
    intH2 = P_H2_grid_new[1+H2_index] - P_H2_grid_new[H2_index] 

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]
    delH2 =  Actual_H2 -   P_H2_grid_new[H2_index]     
    delCO  =  Actual_CO  -   P_CO_grid_new[CO_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2
    ad = delCO / intCO
    bd = delH2 / intH2

    C0000 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index]
    C0100 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index]
    C1000 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index]
    C1100 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index]
    
    C0010 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index]
    C0110 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index]
    C0001 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index,H2_index+1]
    C0101 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C0011 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C0111 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1001 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index,H2_index+1]
    C1101 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index,H2_index+1]

    C1011 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index+1]
    C1111 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index+1]
    C1010 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index,CO_index+1,H2_index]
    C1110 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index,CO_index+1,H2_index]

    # difference over Te dimension
    C000 = C0000 * (1 - yd) + yd * C1000
    C001 = C0001 * (1 - yd) + yd * C1001
    C010 = C0010 * (1 - yd) + yd * C1010
    C011 = C0011 * (1 - yd) + yd * C1011
    C100 = C0100 * (1 - yd) + yd * C1100
    C101 = C0101 * (1 - yd) + yd * C1101
    C110 = C0110 * (1 - yd) + yd * C1110
    C111 = C0111 * (1 - yd) + yd * C1111
    
    #difference over H2O dimension
    C00 = C000*(1-zd) + C100*zd
    C01 = C001*(1-zd) + C101*zd
    C10 = C010*(1-zd) + C110*zd
    C11 = C011*(1-zd) + C111*zd

    #Difference over CO dimension
    C0 = C00*(1-ad) + C10*ad
    C1 = C01*(1-ad) + C11*ad

    #Difference over H2 dimension
    CC = C0*(1-bd) + C1*bd

    D0000 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index]
    D0100 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index]
    D1000 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index]
    D1100 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index]
    
    D0010 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D0110 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index]
    D0001 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D0101 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D0011 = fN2_new[Ts_index,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D0111 = fN2_new[Ts_index,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1001 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index,H2_index+1]
    D1101 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index,H2_index+1]

    D1011 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index+1]
    D1111 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index+1]
    D1010 = fN2_new[Ts_index,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fN2_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1,CO_index+1,H2_index]
    D1110 = fN2_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1,CO_index+1,H2_index]

    # difference over Te dimension
    D000 = D0000 * (1 - yd) + yd * D1000
    D001 = D0001 * (1 - yd) + yd * D1001
    D010 = D0010 * (1 - yd) + yd * D1010
    D011 = D0011 * (1 - yd) + yd * D1011
    D100 = D0100 * (1 - yd) + yd * D1100
    D101 = D0101 * (1 - yd) + yd * D1101
    D110 = D0110 * (1 - yd) + yd * D1110
    D111 = D0111 * (1 - yd) + yd * D1111
    
    #difference over H2O dimension
    D00 = D000*(1-zd) + D100*zd
    D01 = D001*(1-zd) + D101*zd
    D10 = D010*(1-zd) + D110*zd
    D11 = D011*(1-zd) + D111*zd

    #Difference over CO dimension
    D0 = D00*(1-ad) + D10*ad
    D1 = D01*(1-ad) + D11*ad

    #Difference over H2 dimension
    DD = D0*(1-bd) + D1*bd

    #difference over CO2 dimension
    answer = CC*(1-qd) + qd * DD


    if Tsurf < np.min(T_surf_grid): ## added fudge to zero out escape
        #print('do nothing')
        ###return np.exp(15*(-1+Tsurf/np.min(T_surf_grid))) * 10**answer 
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer #looks like a decent extrapolation, need to see what it does to false positives

#    print (Te,answer) extrapolating fH2O low Tskin
    #if Te < np.min(Te_grid): #extrapolate down to lower Te
    #     new_fh2o = answer*np.exp((2.834e6/461.5)*(-1/Te + 1/np.min(Te_grid)))
#         new_fh2o = answer*np.exp((-1/Te + 1/np.min(Te_grid)))
#         print ('extrap',new_fh2o)#np.exp((Te/np.min(Te_grid))-1) * answer)
    #     return new_fh2o#np.exp((Te/np.min(Te_grid))-1.0) * answer

    #if answer <0:
    #    return 0.0
    #elif answer>1:
    #    return 1.0
    #return answer    
    
    return 10**answer






'''
print (my_water_frac(1300,230,1e21,1e19,1e16,1e16),my_fH2O(1300,230,1e21,1e19,1e16,1e16))
print (my_water_frac(300,230,1e21,1e19,1e16,1e16),my_fH2O(300,230,1e21,1e19,1e16,1e16))


fH2O_4D_new = scipy.interpolate.RegularGridInterpolator((T_surf_grid,Te_grid,P_H2O_grid_new,P_CO2_grid_new,P_CO_grid_new,P_H2_grid_new),fH2O_new,method='linear',bounds_error=False,fill_value = None)
water4D_new = scipy.interpolate.RegularGridInterpolator((T_surf_grid,Te_grid,P_H2O_grid_new,P_CO2_grid_new,P_CO_grid_new,P_H2_grid_new),water_frac_multi_new,method='linear',bounds_error=False,fill_value = None)
print ('Compare 2nd funs')
print (my_water_frac(300,230,1e21,1e19,1e16,1e16),water4D_new((300,230,1e21,1e19,1e16,1e16)))
print (my_fH2O(300,230,1e21,1e19,1e16,1e16),10**fH2O_4D_new((300,230,1e21,1e19,1e16,1e16)))
'''

@jit(nopython=True) 
def correction(Tsurf,Te,PH2O,PCO2,PCO,PH2,rp,g,CO3,PN2,MMW,PO2): ## strictly speaking need O2 as well
    #Temperature held constant for the purposes of calculating equilibrium constants (see manuscript)
    if Tsurf<274:
        T=274
    else:
        T=Tsurf
    pK1=17.788 - .073104 *T - .0051087*35 + 1.1463*10**-4*T**2
    pK2=20.919 - .064209 *T - .011887*35 + 8.7313*10**-5*T**2
    #H_CO2=pylab.exp(9345.17/T - 167.8108 + 23.3585 * pylab.log(T)+(.023517 - 2.3656*10**-4*T+4.7036*10**-7*T**2)*35)  
    H_CO2=1.0/(0.018*10.0*np.exp(-6.8346 + 1.2817e4/T - 3.7668e6/T**2 + 2.997e8/T**3)   )
    # from https://srd.nist.gov/JPCRD/jpcrd427.pdf  with unit conversion

    atmo_fraction = my_water_frac(Tsurf,Te,PH2O,PCO2,PCO,PH2)
    #print(atmo_fraction)

    if (Tsurf>3000.0):
        OLR1 = my_interp(3000.0,Te,PH2O,PCO2,PCO,PH2)
        OLR2 = my_interp(2990,Te,PH2O,PCO2,PCO,PH2)
        logOLR = np.log(OLR1) + (Tsurf-3000.0)*(np.log(OLR1)-np.log(OLR2))/(3000.0-2990.0)
        OLR_final = np.exp(logOLR)
        return [OLR_final,PCO2,0.0,0.0,0.0,0.0,PCO2]

    if (atmo_fraction == 1.0)or(PCO2<0):
        return [my_interp(Tsurf,Te,PH2O,PCO2,PCO,PH2),PCO2,0.0,0.0,0.0,0.0,PCO2]
    else:
        #atmoH2O =  atmo_fraction*PH2O
        Mass_oceans_crude = PH2O*(1.0 - atmo_fraction ) #* 4 *np.pi *rp**2 / g ## (not) doing this properly
        #mtot = atmoH2O + PCO2 + PN2 + PO2 + PCO + PH2 #actually all masses
        #PTOT = g*mtot /( 4*np.pi*rp**2)
        MCO2 = 0.044
        Mave = MMW
        #Mave = MCO2 * (PCO2/PTOT) + 0.018 * (atmoH2O/PTOT) + 0.028 * (PN2/PTOT)
        total_mass_CO2 = PCO2
        cCon = total_mass_CO2/(MCO2*Mass_oceans_crude) # mol CO2/kg ocean
        if cCon < CO3: #quick fix to ensure no more CO3 than in total atmo-ocean system!
            CO3 = 0.9999*cCon
        #cCon = s * pCO2 + DIC, where pCO2 is in bar
        s = 1e5*(4.0 *np.pi * rp**2 / (MCO2*g) )* (MCO2 / Mave) / Mass_oceans_crude ## mol CO2/ kg atm / Pa, so *1e5 Pa/bar
        # a bit of a fudge - might be better to return partial pressures in grid, then use those to find new pCO2
        # e.g. pp
        #aa = ALK/(10**-pK2*10**-pK1)*(1+s/H_CO2)
        #bb = (ALK-cCon)/(10**-pK2)
        #cc = (ALK-2*cCon)
        # eq S24 from Nat. Comm. paper
        aa = CO3 * (1 + s/H_CO2)/(10**-pK2*10**-pK1)
        bb = CO3 / (10**-pK2)
        cc = CO3 - cCon
        rr1 = - (bb + np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        rr2 = - (bb - np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        if rr1 <= rr2:
            EM_H_o  = rr2
        else:
            EM_H_o = rr1
        #roots=np.roots([ALK/(10**-pK2*10**-pK1)*(1+s/H_CO2),(ALK-cCon)/(10**-pK2),(ALK-2*cCon)])
        #EM_H_o=np.max(roots) ## Evolving model ocean hydrogen molality
        ALK = CO3 * (2+EM_H_o/(10**-pK2))
        EM_pH_o=-np.log10(EM_H_o) ## Evolving model ocean pH
        EM_hco3_o=ALK-2*CO3 ## Evolving model ocean bicarbonate molality
        EM_co2aq_o=( EM_hco3_o*EM_H_o/(10**-pK1) ) ## Evolving model ocean aqueous CO2 molality
        EM_ppCO2_o = EM_co2aq_o /H_CO2 ## Evolving model atmospheric pCO2
        DIC_check = EM_hco3_o + CO3 + EM_co2aq_o
        Mass_CO2_atmo = PCO2 - DIC_check*MCO2*Mass_oceans_crude
        true_OLR =  my_interp(Tsurf,Te,PH2O,Mass_CO2_atmo,PCO,PH2)
        #return [my_interp(Tsurf,Te,PH2O,PCO2,PCO,PH2),PCO2,0.0,0.0,Mass_oceans_crude,0.0,PCO2]
        return [true_OLR,EM_ppCO2_o*1e5,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo]


''' # try without messing with it first, if fast then recreate grid with everything you need.
@jit(nopython=True) 
def correctionClima(Tsurf,Te,PH2O,PCO2,PCO,PH2,rp,g,CO3,PN2,MMW,PO2): ## strictly speaking need O2 as well
    #Temperature held constant for the purposes of calculating equilibrium constants (see manuscript)
    if Tsurf<274:
        T=274
    else:
        T=Tsurf
    pK1=17.788 - .073104 *T - .0051087*35 + 1.1463*10**-4*T**2
    pK2=20.919 - .064209 *T - .011887*35 + 8.7313*10**-5*T**2
    #H_CO2=pylab.exp(9345.17/T - 167.8108 + 23.3585 * pylab.log(T)+(.023517 - 2.3656*10**-4*T+4.7036*10**-7*T**2)*35)  
    H_CO2=1.0/(0.018*10.0*np.exp(-6.8346 + 1.2817e4/T - 3.7668e6/T**2 + 2.997e8/T**3)   )
    # from https://srd.nist.gov/JPCRD/jpcrd427.pdf  with unit conversion

    atmo_fraction = my_water_frac(Tsurf,Te,PH2O,PCO2,PCO,PH2)
    #print(atmo_fraction)

    if (Tsurf>3000.0):
        OLR1 = my_interp(3000.0,Te,PH2O,PCO2,PCO,PH2)
        OLR2 = my_interp(2990,Te,PH2O,PCO2,PCO,PH2)
        logOLR = np.log(OLR1) + (Tsurf-3000.0)*(np.log(OLR1)-np.log(OLR2))/(3000.0-2990.0)
        OLR_final = np.exp(logOLR)
        return [OLR_final,PCO2,0.0,0.0,0.0,0.0,PCO2]

    if (atmo_fraction == 1.0)or(PCO2<0):
        return [my_interp(Tsurf,Te,PH2O,PCO2,PCO,PH2),PCO2,0.0,0.0,0.0,0.0,PCO2]
    else:
        #atmoH2O =  atmo_fraction*PH2O
        Mass_oceans_crude = PH2O*(1.0 - atmo_fraction ) #* 4 *np.pi *rp**2 / g ## (not) doing this properly
        #mtot = atmoH2O + PCO2 + PN2 + PO2 + PCO + PH2 #actually all masses
        #PTOT = g*mtot /( 4*np.pi*rp**2)
        MCO2 = 0.044
        Mave = MMW
        #Mave = MCO2 * (PCO2/PTOT) + 0.018 * (atmoH2O/PTOT) + 0.028 * (PN2/PTOT)
        total_mass_CO2 = PCO2
        cCon = total_mass_CO2/(MCO2*Mass_oceans_crude) # mol CO2/kg ocean
        if cCon < CO3: #quick fix to ensure no more CO3 than in total atmo-ocean system!
            CO3 = 0.9999*cCon
        #cCon = s * pCO2 + DIC, where pCO2 is in bar
        s = 1e5*(4.0 *np.pi * rp**2 / (MCO2*g) )* (MCO2 / Mave) / Mass_oceans_crude ## mol CO2/ kg atm / Pa, so *1e5 Pa/bar
        # a bit of a fudge - might be better to return partial pressures in grid, then use those to find new pCO2
        # e.g. pp
        #aa = ALK/(10**-pK2*10**-pK1)*(1+s/H_CO2)
        #bb = (ALK-cCon)/(10**-pK2)
        #cc = (ALK-2*cCon)
        # eq S24 from Nat. Comm. paper
        aa = CO3 * (1 + s/H_CO2)/(10**-pK2*10**-pK1)
        bb = CO3 / (10**-pK2)
        cc = CO3 - cCon
        rr1 = - (bb + np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        rr2 = - (bb - np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        if rr1 <= rr2:
            EM_H_o  = rr2
        else:
            EM_H_o = rr1
        #roots=np.roots([ALK/(10**-pK2*10**-pK1)*(1+s/H_CO2),(ALK-cCon)/(10**-pK2),(ALK-2*cCon)])
        #EM_H_o=np.max(roots) ## Evolving model ocean hydrogen molality
        ALK = CO3 * (2+EM_H_o/(10**-pK2))
        EM_pH_o=-np.log10(EM_H_o) ## Evolving model ocean pH
        EM_hco3_o=ALK-2*CO3 ## Evolving model ocean bicarbonate molality
        EM_co2aq_o=( EM_hco3_o*EM_H_o/(10**-pK1) ) ## Evolving model ocean aqueous CO2 molality
        EM_ppCO2_o = EM_co2aq_o /H_CO2 ## Evolving model atmospheric pCO2
        DIC_check = EM_hco3_o + CO3 + EM_co2aq_o
        Mass_CO2_atmo = PCO2 - DIC_check*MCO2*Mass_oceans_crude
        true_OLR =  my_interp(Tsurf,Te,PH2O,Mass_CO2_atmo,PCO,PH2)
        return [true_OLR,EM_ppCO2_o*1e5,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo]
'''

'''
print('Corection compare')
print(my_interp(285,200,1.4e21,1.5e16,1e16,1e16))
print (correction(285,200,1.4e21,3.8e16,1e16,1e16,6371000,9.8,5e-5,1e18,0.028,1e-6 ))

import pdb
pdb.set_trace()
'''
# Testing Te dep (weird because linear interpolation)
'''
print (correction(290,200,1e6,1e6,6371000,9.8,2.5e-3,1e5,0.028,1e-6 ))

t1 = time.time()
correction(600,210,1e6,1e6,6371000,9.8,2.5e-3,1e5,0.018,1e-6) 
t2 = time.time()
print (t2 - t1)
#water_frac1 = water4D_fun_new(320,250,1e7,1e6)
#water_frac = my_water_frac(320,250,1e7,1e6)
#print (water_frac1,water_frac)

#T = np.linspace(250,4000,1000)
Te_array = np.linspace(150,300,1000)
Te_exam = Te_array
OLR = np.copy(Te_array)
OLR_new = np.copy(Te_array)
CO2_out = np.copy(Te_array)
#BB = np.copy(T)
water_frac_ar = np.copy(Te_array)
#water_frac_old = np.copy(T)
my_fH2O_array = np.copy(Te_array)
# #


H2O_exam=5e7#250e5#1e9

CO2_exam = 1e4#300e5#1e8
T = 260#750


for i in range(0,len(Te_array)):

###    #BB[i] = 5.67e-8 * T[i]**4.0

    [OLR[i],CO2_out[i],ggg,aaa,ccc] = correction(T,float(Te_exam[i]),H2O_exam,CO2_exam,6371000,9.8,1.3e-3,1e5,0.018,1e-6) 

    OLR_new[i] = my_interp(T,float(Te_exam[i]),H2O_exam,CO2_exam)

    water_frac_ar[i] = my_water_frac(T,float(Te_exam[i]),H2O_exam,CO2_exam)

##    water_frac_old[i] = water4D_fun_new(T[i],214,1e7,4.8e7)

    my_fH2O_array[i] = my_fH2O(T,float(Te_exam[i]),H2O_exam,CO2_exam)



pylab.figure()

pylab.subplot(3,1,1)

##pylab.xlabel('Tsurf (K)')
##pylab.ylabel('OLR (W/m2)')
pylab.semilogy(Te_exam,OLR,'r')
pylab.semilogy(Te_exam,OLR_new,'b--')
####pylab.plot(T,BB,'k')
pylab.subplot(3,1,2)
##pylab.plot(T,CO2_out,'r')
##pylab.plot(T,CO2_exam + 0 *T,'b--')
##pylab.xlabel('Tsurf (K)')
pylab.ylabel('Atmo water fraction')
pylab.plot(Te_exam,water_frac_ar)
###pylab.plot(T,water_frac_ar,'r--')
pylab.subplot(3,1,3)
pylab.semilogy(Te_exam,my_fH2O_array)
##pylab.xlabel('Tsurf (K)')
##pylab.ylabel('Stratosphere H2O mixing ratio')
pylab.show()


'''
# Testing T dep
'''
print (correction(290,200,1e6,1e6,6371000,9.8,2.5e-3,1e5,0.018,1e-6 ))

t1 = time.time()
correction(600,210,1e6,1e6,6371000,9.8,2.5e-3,1e5,0.018,1e-6) 
t2 = time.time()
print (t2 - t1)
#water_frac1 = water4D_fun_new(320,250,1e7,1e6)
#water_frac = my_water_frac(320,250,1e7,1e6)
#print (water_frac1,water_frac)

T = np.linspace(250,4000,1000)
OLR = np.copy(T)
OLR_new = np.copy(T)
CO2_out = np.copy(T)
#BB = np.copy(T)
water_frac_ar = np.copy(T)
#water_frac_old = np.copy(T)
my_fH2O_array = np.copy(T)
# #

Te_exam=180
H2O_exam=1e8#250e5#1e9
CO2_exam = .1e7#300e5#1e8

for i in range(0,len(T)):
###    #BB[i] = 5.67e-8 * T[i]**4.0
    [OLR[i],CO2_out[i],ggg,aaa,ccc] = correction(float(T[i]),Te_exam,H2O_exam,CO2_exam,6371000,9.8,1.3e-3,1e5,0.018,1e-6) 
    OLR_new[i] = my_interp(float(T[i]),Te_exam,H2O_exam,CO2_exam)
    water_frac_ar[i] = my_water_frac(T[i],Te_exam,H2O_exam,CO2_exam)
##    water_frac_old[i] = water4D_fun_new(T[i],214,1e7,4.8e7)
    my_fH2O_array[i] = my_fH2O(T[i],Te_exam,H2O_exam,CO2_exam)

pylab.figure()
pylab.subplot(3,1,1)
##pylab.xlabel('Tsurf (K)')
##pylab.ylabel('OLR (W/m2)')
pylab.semilogy(T,OLR,'r')
pylab.semilogy(T,OLR_new,'b--')
####pylab.plot(T,BB,'k')
pylab.subplot(3,1,2)
##pylab.plot(T,CO2_out,'r')
##pylab.plot(T,CO2_exam + 0 *T,'b--')
##pylab.xlabel('Tsurf (K)')
pylab.ylabel('Atmo water fraction')
pylab.plot(T,water_frac_ar)
###pylab.plot(T,water_frac_ar,'r--')
pylab.subplot(3,1,3)
pylab.semilogy(T,my_fH2O_array)
##pylab.xlabel('Tsurf (K)')
##pylab.ylabel('Stratosphere H2O mixing ratio')
pylab.show()


## For supplementary figure
num_ex = 6
T = np.linspace(273,4000,1000)
T = np.linspace(150,4000,1000)
OLR = np.ones(shape=(len(T),num_ex))
CO2 = np.logspace(2,7,num_ex)
water_frac_ar = np.copy(OLR)
strat_H2O = np.copy(OLR)

for i in range(0,len(T)):
    for k in range(0,num_ex):
        [OLR[i,k],EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check] = correction(float(T[i]),210,260*1e5,CO2[k],6371000,9.8,1e-3,1e5,0.018,1e-6) 
        water_frac_ar[i,k] = my_water_frac(T[i],210,260*1e5,CO2[k])
        strat_H2O[i,k] = my_fH2O(T[i],210,260*1e5,CO2[k])

pylab.figure()
pylab.subplot(3,1,1)
for k in range(0,num_ex):
    aye = np.log10(CO2[k])
    ayeaye = str('10$^%d$' %aye)
    labelc = 'pCO$_2$='+ayeaye+' Pa'
    pylab.semilogy(T,OLR[:,k],label=labelc)
    pylab.ylabel('OLR (W/m2)')
    #pylab.xlabel('Surface Temperaure (K)')
    pylab.legend(frameon=False,ncol=2)
pylab.subplot(3,1,2)
for k in range(0,num_ex):
    pylab.plot(T,water_frac_ar[:,k])
    pylab.ylabel('Fraction of water in atmosphere')
    #pylab.xlabel('Surface Temperaure (K)')
pylab.subplot(3,1,3)
for k in range(0,num_ex):
    aye = np.log10(CO2[k])
    ayeaye = str('10$^%d$' %aye)
    labelc = 'pCO$_2$='+ayeaye+' Pa'
    pylab.semilogy(T,strat_H2O[:,k],label= labelc)
    pylab.ylabel('Stratospheric H2O mixing ratio')
    pylab.xlabel('Surface Temperaure (K)')
    pylab.legend(frameon=False,ncol=2)
pylab.show()
'''
