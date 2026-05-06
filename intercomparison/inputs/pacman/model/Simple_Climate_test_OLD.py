import numpy as np
import pylab as plt
from numba import jit
import scipy.optimize
global y_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
from radiative_functions_6dim_CO_as_N2 import *
from numba_nelder_mead import nelder_mead

M_H2O_grid_new = np.logspace(15,22,28)
M_CO2_grid_new = np.logspace(15,21.5,12) 
M_CO_grid_new = np.logspace(15,21.5,6) 
M_H2_grid_new = np.logspace(15,22,7)
T_surf_grid = np.linspace(250,3000,120)#150)
Te_grid = np.linspace(150,350,10)

### MAYB NEED TO LINSPACE TO INTERPOLATE BETWEEN ORDERS OF MAGNITUDE

#Everything_array = np.load('Ev_array.npy')
#Everything_array = np.load('Ev_array_improved_corrected.npy')
                           # Output_slice[m,z,nn,r,ik,0] = olr_single 
                           # Output_slice[m,z,nn,r,ik,1:6] = upper_single
                           # Output_slice[m,z,nn,r,ik,6] = atm_frac

#values = Everything_array[:,:,:,:,:,:,0]

#print (np.shape(values)) #(120, 28, 12, 6, 10, 7)

from scipy import optimize
import scipy.optimize

#points = (T_surf_grid,M_H2O_grid_new,M_CO2_grid_new,M_CO_grid_new,Te_grid,M_H2_grid_new)

'''
def funTs_new(Ts,H2O_input,CO2_input,CO_input,Te_input,H2_input,ASR):
    OLR = interpn(points,values,np.array([Ts,H2O_input,CO2_input,CO_input,Te_input,H2_input]))
    print(Ts,OLR,ASR,(ASR-OLR)**2)
    return (ASR-OLR)**2
'''

@jit(nopython=True)
def funTs_new_fast(Ts,H2O_input,CO2_input,CO_input,Te_input,H2_input,ASR):
    Ts_in= Ts[0]
    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(Ts_in,Te_input,H2O_input,CO2_input,CO_input,H2_input,6371000.0,9.8,5e-5,1e18,0.028,1e-6 )
    #print(Ts,OLR,ASR,(ASR-OLR)**2)
    return -(ASR-OLR)**2


#specific example
#points = (T_surf_grid,M_H2O_grid_new,M_CO2_grid_new,M_CO_grid_new,Te_grid,M_H2_grid_new)
ASR_in = 250.0
MH2O_in = 3e21
MCO2_in = 1e16
MCO_in = 1e16
MH2_in = 1e18
Te_in = 210.0
MN2_in = 1e18

def solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in,Tguess):
    initialize_fast = 251.0#
    initialize_fast = np.array(Tguess)
    ace1 =  nelder_mead(funTs_new_fast, x0=initialize_fast, bounds=np.array([[250.0], [4000.0]]).T, args = (MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
    SurfT = ace1.x[0]
    new_abs = abs(ace1.fun)
    if new_abs > 0.1:
        initialize_fast = 1500#
        initialize_fast = np.array(initialize_fast)
        ace2 =  nelder_mead(funTs_new_fast, x0=initialize_fast, bounds=np.array([[250.0], [4000.0]]).T, args = (MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
        #print(ace2)
        if abs(ace2.fun) <  new_abs:
            SurfT = ace2.x[0]
            new_abs = abs(ace2.fun)
               
        if new_abs > 1.0:#0.1:
            
            ace3= optimize.minimize(funTs_new_fast,x0=251,args = (MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in),method='L-BFGS-B',bounds = ((250,4000),))
            #print(ace3)
            if abs(ace3.fun) <  new_abs:
                SurfT = ace3.x[0]
                new_abs = abs(ace3.fun) 
                if new_abs > 1.0:#0.1:
                    rand_start = 250 + 1850*np.random.uniform()
                    ace4= optimize.minimize(funTs_new_fast,x0=rand_start,args = (MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in),method='L-BFGS-B',bounds = ((150,4000),))
                    #print(ace4)
                    if abs(ace4.fun) <  new_abs:
                        SurfT =ace4.x[0]     
                        new_abs = abs(ace4.fun) 
    #print ('Surface temperature:',SurfT,' Error:',new_abs)
    [H2O_input,CO2_input,CO_input,Te_input,H2_input,ASR] = [MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in]
    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_input,H2O_input, CO2_input,CO_input,H2_input, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
    return [SurfT,new_abs, OLR,EM_ppCO2_o]  
   
'''                                                                                                 

def solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in,Tguess):
    OLR = ASR_in
    new_abs = 1e-12
    #SurfT = (ASR_in/280)**0.2 * (273 + 100*np.log10(MH2O_in/1e21)+10*np.log10(MCO2_in/1e18))
    total_mass = MH2O_in+MCO2_in+MCO_in+MH2_in
    SurfT = (ASR_in/5.67e-8)**0.25 +100* np.max([1,np.log10((total_mass)/1e15) ])
    #print ('SurfT,total_mass',SurfT,total_mass)
    return [SurfT,new_abs, OLR]                                                                                                      
'''




def Plot_compare(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in,SurfT_compare):
    OLR=[]
    #OLR_3 = []
    point_ex = np.linspace(250,4000,1000)
    for i in range(0,len(point_ex)):
    #OLR.append(interpn(points,values,np.array([point_ex[i],0.5*(M_H2O_grid_new[1]+M_H2O_grid_new[2]),M_CO2_grid_new[1],M_CO_grid_new[0],Te_grid[1],M_H2_grid_new[0]])))
    #OLR.append(interpn(points,values,np.array([point_ex[i],0.3e21,1e20,1e16,200.0,1e18])))
        [OLR_out,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(point_ex[i],Te_in,MH2O_in,MCO2_in,MCO_in,MH2_in,6371000.0,9.8,5e-5,MN2_in,0.028,1e-6 )
    #OLR_out = my_interp(point_ex[i],Te_in,MH2O_in,MCO2_in,MCO_in,MH2_in)
        OLR.append(OLR_out)

#ace2 = optimize.minimize(funTs_new_fast,x0=1000,args=(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in),method='COBYLA',bounds = ((250,3000),),options = {'maxiter':3000})
#print(ace2)

    plt.figure()
    plt.semilogy(point_ex,np.array(OLR),'b',label='OLR')
    plt.semilogy(point_ex,0*point_ex+ASR_in,'k',label='ASR')
    plt.semilogy([SurfT_compare,SurfT_compare],[np.min(OLR),np.max(OLR)],'g--',label='T_surface')
    plt.legend()
    plt.show()

'''
[SurfT,new_abs] =solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in)
Plot_compare(MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in,ASR_in,SurfT)


t1 = time.time()
#[SurfT,new_abs] =solve_Tsurf(MH2O_in/10.0,MCO2_in,MCO_in,Te_in,1e21,ASR_in)
[SurfT,new_abs] =solve_Tsurf(MH2O_in*10.0,MCO2_in,MCO_in,210.0,MH2_in,240.0)
[SurfT,new_abs] =solve_Tsurf(MH2O_in,MCO2_in*10.0,MCO_in,Te_in,MH2_in,ASR_in)
[SurfT,new_abs] =solve_Tsurf(MH2O_in,MCO2_in*10.0,MCO_in,Te_in,MH2_in,255.0)
t2 = time.time()


points = (T_surf_grid,M_H2O_grid_new,M_CO2_grid_new,M_CO_grid_new,Te_grid,M_H2_grid_new)
valuesPCO = Everything_array[:,:,:,:,:,:,7]
valuesPN2 = Everything_array[:,:,:,:,:,:,8]
valuesPH2O = Everything_array[:,:,:,:,:,:,9]
valuesPCO2 = Everything_array[:,:,:,:,:,:,10]
valuesPH2 = Everything_array[:,:,:,:,:,:,11]

PCO=interpn(points,valuesPCO,np.array([SurfT,MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in]))
PN2 =interpn(points,valuesPN2,np.array([SurfT,MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in]))
PH2O=interpn(points,valuesPH2O,np.array([SurfT,MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in]))
PCO2 =interpn(points,valuesPCO2,np.array([SurfT,MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in]))
PH2 =interpn(points,valuesPH2,np.array([SurfT,MH2O_in,MCO2_in,MCO_in,Te_in,MH2_in]))

print ('Surface temperature:',SurfT)
print('PCO',PCO,'PN2',PN2,'PH2O',PH2O,'PCO2',PCO2,'PH2',PH2)
#[SurfT,new_abs] =solve_Tsurf(1e17,1e18,MCO_in,Te_in,MH2_in,300.0)
#Plot_compare(1e17,1e18,MCO_in,Te_in,MH2_in,300.0,SurfT)
#pdb.set_trace()
'''
