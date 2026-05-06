## This function combines multiple Monte Carlo outputs into a single figure 

import numpy as np
from other_functions import *
from joblib import Parallel, delayed
from New_PACMAN_v4 import forward_model
from stellar_funs import *
from all_classes_NPM import *
import pylab as plt
import os
import shutil
from numba import jit
global y_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
from radiative_functions_6dim_CO_as_N2 import my_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
global solve_Tsurf,Plot_compare
from Simple_Climate_test import solve_Tsurf,Plot_compare


#redo
[bin_centers_e,bin_low_e,bin_high_e,bin_means_e] = np.load('e_metalsink_REDO.npy',allow_pickle=True)
[bin_centers_b,bin_low_b,bin_high_b,bin_means_b] = np.load('e_nominal_REDO.npy',allow_pickle=True)




plt.figure()
#plt.subplot(3,3,1)
#plt.semilogy(bin_centers,(bin_means[0,:]/(365*24*60*60)-1e7)/1e6,'k-')
#pylab.fill_between(bin_centers,(bin_low[0,:]/(365*24*60*60)-1e7)/1e6,(bin_high[0,:]/(365*24*60*60)-1e7)/1e6, color='grey', alpha=0.4) 
#plt.ylabel('Duration magma ocean (Myr)')
plt.subplot(4,4,9)
plt.semilogy(bin_centers_e,bin_means_e[5,:]*0.012,'b',label='Atmo')
pylab.fill_between(bin_centers_e,bin_low_e[5,:]*0.012,bin_high_e[5,:]*0.012, color='blue', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[6,:]*0.012,'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[6,:]*0.012,bin_high_e[6,:]*0.012, color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[7,:]*0.012,'r',label='Total')
pylab.fill_between(bin_centers_e,bin_low_e[7,:]*0.012,bin_high_e[7,:]*0.012, color='red', alpha=0.4) 
plt.ylabel('Final C Reservoirs (kg)')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e11,8e21]) 
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,10)
plt.semilogy(bin_centers_b,bin_means_b[5,:]*0.012,'b',label='Atmo')
pylab.fill_between(bin_centers_b,bin_low_b[5,:]*0.012,bin_high_b[5,:]*0.012, color='blue', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[6,:]*0.012,'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[6,:]*0.012,bin_high_b[6,:]*0.012, color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[7,:]*0.012,'r',label='Total')
pylab.fill_between(bin_centers_b,bin_low_b[7,:]*0.012,bin_high_b[7,:]*0.012, color='red', alpha=0.4) 
#plt.ylabel('Final C reservoir (mol C)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e11,8e21]) 
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,13)
plt.semilogy(bin_centers_e,bin_means_e[2,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[2,:],bin_high_e[2,:], color='grey', alpha=0.4)  
plt.semilogy(bin_centers_e,bin_means_e[3,:],'b',label='Fluid')
pylab.fill_between(bin_centers_e,bin_low_e[3,:],bin_high_e[3,:], color='blue', alpha=0.4)  
plt.semilogy(bin_centers_e,bin_means_e[4,:],'r',label='Total')
pylab.fill_between(bin_centers_e,bin_low_e[4,:],bin_high_e[4,:], color='red', alpha=0.4)  
plt.ylabel('Final H Reservoirs (kg)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e15,3e23])
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,14)
plt.semilogy(bin_centers_b,bin_means_b[2,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[2,:],bin_high_b[2,:], color='grey', alpha=0.4)  
plt.semilogy(bin_centers_b,bin_means_b[3,:],'b',label='Fluid')
pylab.fill_between(bin_centers_b,bin_low_b[3,:],bin_high_b[3,:], color='blue', alpha=0.4)  
plt.semilogy(bin_centers_b,bin_means_b[4,:],'r',label='Total')
pylab.fill_between(bin_centers_b,bin_low_b[4,:],bin_high_b[4,:], color='red', alpha=0.4)  
#plt.ylabel('Present day H reservoirs (kg)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e15,3e23])
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e2, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e-1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e-1, color='red', alpha=0.1) 

plt.semilogy(bin_centers_e,bin_means_e[26,:],'k-')
pylab.fill_between(bin_centers_e,bin_low_e[26,:],bin_high_e[26,:], color='grey', alpha=0.4) 

#def forw(x):
#    return x*1000.0

#def backn(x):
#    return x/1000.0
#import matplotlib
#matplotlib.axes.SecondaryAxis(parent=pylab.xaxis,orientation='x',location='top',functions=(forw,backn))
#pylab.secondary_xaxis(location='top',functions=(forw,backn))

#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')

plt.title('Trappist-1e; Fe->Core')
#plt.title('LP 890-9 c; Fe->Core')
#plt.title('Prox Cen b; Fe->Core')
plt.ylabel('Total Surface Pressure (bar)')
plt.ylim([1e-1,2e4])
plt.xlim([20.345,22.7])
plt.subplot(4,4,2) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e2, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e-1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e-1, color='red', alpha=0.1) 
plt.semilogy(bin_centers_b,bin_means_b[26,:],'k-')
pylab.fill_between(bin_centers_b,bin_low_b[26,:],bin_high_b[26,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1e; Fe->Mantle')
#plt.title('LP 890-9 c; Fe->MantleCore')
#plt.title('Prox Cen b; Fe->Core')
#plt.ylabel('Total Present Surface (bar)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e-1,2e4])
plt.xlim([20.345,22.7])

#plt.subplot(3,3,5)
#plt.plot(bin_centers,bin_means[8,:],'r')
#pylab.fill_between(bin_centers,bin_low[8,:],bin_high[8,:], color='red', alpha=0.4) 
#plt.ylabel('Final mantle redox (deltaFMQ)')
#plt.subplot(3,3,6)
#plt.semilogy(bin_centers,bin_means[9,:],'k',label='Solid')
#pylab.fill_between(bin_centers,bin_low[9,:],bin_high[9,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[10,:],'r',label='Atmo')
#pylab.fill_between(bin_centers,bin_low[10,:],bin_high[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
#plt.legend()
plt.subplot(4,4,5)
plt.semilogy(bin_centers_e,bin_means_e[11,:],'k-',label='CO$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[11,:],bin_high_e[11,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[12,:],'r-',label='CO')
pylab.fill_between(bin_centers_e,bin_low_e[12,:],bin_high_e[12,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[13,:],'c-',label='H$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[13,:],bin_high_e[13,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[1,:],'b-',label='H$_2$O')
pylab.fill_between(bin_centers_e,bin_low_e[1,:],bin_high_e[1,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[14,:],'m-',label='CH$_4$')
pylab.fill_between(bin_centers_e,bin_low_e[14,:],bin_high_e[14,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[15,:],'y-',label='O$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[15,:],bin_high_e[15,:], color='yellow', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_means_e[16,:],'g-',label='Atmo H2O')
#pylab.fill_between(bin_centers_e,bin_low_e[16,:],bin_high_e[16,:], color='green', alpha=0.4) 
plt.ylabel('Final Pressure (bar)')
plt.ylim([1e-3,2e4])
plt.xlim([20.345,22.7])
plt.legend(ncol=2)
plt.subplot(4,4,6)
plt.semilogy(bin_centers_b,bin_means_b[11,:],'k-',label='CO$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[11,:],bin_high_b[11,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[12,:],'r-',label='CO')
pylab.fill_between(bin_centers_b,bin_low_b[12,:],bin_high_b[12,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[13,:],'c-',label='H$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[13,:],bin_high_b[13,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[1,:],'b-',label='H$_2$O')
pylab.fill_between(bin_centers_b,bin_low_b[1,:],bin_high_b[1,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[14,:],'m-',label='CH$_4$')
pylab.fill_between(bin_centers_b,bin_low_b[14,:],bin_high_b[14,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[15,:],'y-',label='O$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[15,:],bin_high_b[15,:], color='yellow', alpha=0.4) 
#plt.semilogy(bin_centers_b,bin_means_b[16,:],'g-',label='Atmo H2O')
#pylab.fill_between(bin_centers_b,bin_low_b[16,:],bin_high_b[16,:], color='green', alpha=0.4) 
#plt.ylabel('Final Pressure (bar)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e-3,2e4])
plt.xlim([20.345,22.7])
plt.legend(ncol=2)





#redo
[bin_centers_e,bin_low_e,bin_high_e,bin_means_e] = np.load('b_metalsink_REDO.npy',allow_pickle=True)
[bin_centers_b,bin_low_b,bin_high_b,bin_means_b] = np.load('b_nominal_REDO.npy',allow_pickle=True)





#plt.subplot(3,3,1)
#plt.semilogy(bin_centers,(bin_means[0,:]/(365*24*60*60)-1e7)/1e6,'k-')
#pylab.fill_between(bin_centers,(bin_low[0,:]/(365*24*60*60)-1e7)/1e6,(bin_high[0,:]/(365*24*60*60)-1e7)/1e6, color='grey', alpha=0.4) 
#plt.ylabel('Duration magma ocean (Myr)')
plt.subplot(4,4,11)
plt.semilogy(bin_centers_e,bin_means_e[5,:]*0.012,'b',label='Atmo')
pylab.fill_between(bin_centers_e,bin_low_e[5,:]*0.012,bin_high_e[5,:]*0.012, color='blue', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[6,:]*0.012,'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[6,:]*0.012,bin_high_e[6,:]*0.012, color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[7,:]*0.012,'r',label='Total')
pylab.fill_between(bin_centers_e,bin_low_e[7,:]*0.012,bin_high_e[7,:]*0.012, color='red', alpha=0.4) 
#plt.ylabel('Final C reservoir (mol C)')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e11,8e21]) 
plt.xlim([20.345,22.7])

plt.legend()
plt.subplot(4,4,12)
plt.semilogy(bin_centers_b,bin_means_b[5,:]*0.012,'b',label='Atmo')
pylab.fill_between(bin_centers_b,bin_low_b[5,:]*0.012,bin_high_b[5,:]*0.012, color='blue', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[6,:]*0.012,'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[6,:]*0.012,bin_high_b[6,:]*0.012, color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[7,:]*0.012,'r',label='Total')
pylab.fill_between(bin_centers_b,bin_low_b[7,:]*0.012,bin_high_b[7,:]*0.012, color='red', alpha=0.4) 
#plt.ylabel('Present Day C Reservoirs (mol C)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e11,8e21]) 
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,15)
plt.semilogy(bin_centers_e,bin_means_e[2,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[2,:],bin_high_e[2,:], color='grey', alpha=0.4)  
plt.semilogy(bin_centers_e,bin_means_e[3,:],'b',label='Fluid')
pylab.fill_between(bin_centers_e,bin_low_e[3,:],bin_high_e[3,:], color='blue', alpha=0.4)  
plt.semilogy(bin_centers_e,bin_means_e[4,:],'r',label='Total')
pylab.fill_between(bin_centers_e,bin_low_e[4,:],bin_high_e[4,:], color='red', alpha=0.4)  
#plt.ylabel('Present Day H Reservoirs (kg)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e15,3e23])
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,16)
plt.semilogy(bin_centers_b,bin_means_b[2,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[2,:],bin_high_b[2,:], color='grey', alpha=0.4)  
plt.semilogy(bin_centers_b,bin_means_b[3,:],'b',label='Fluid')
pylab.fill_between(bin_centers_b,bin_low_b[3,:],bin_high_b[3,:], color='blue', alpha=0.4)  
plt.semilogy(bin_centers_b,bin_means_b[4,:],'r',label='Total')
pylab.fill_between(bin_centers_b,bin_low_b[4,:],bin_high_b[4,:], color='red', alpha=0.4)  
#plt.ylabel('Present day H reservoirs (kg)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e15,3e23])
plt.xlim([20.345,22.7])
plt.legend()
plt.subplot(4,4,3) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e2, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e-1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e-1, color='red', alpha=0.1)  
plt.semilogy(bin_centers_e,bin_means_e[26,:],'k-')
pylab.fill_between(bin_centers_e,bin_low_e[26,:],bin_high_e[26,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1b; Fe->Core')
#plt.title('LP 890-9 b; Fe->Core')
#plt.title('Trappist-1f; Fe->Core')
#plt.ylabel('Total Present Surface Volatiles (bar)')
plt.ylim([1e-1,2e4])
plt.xlim([20.345,22.7])
plt.subplot(4,4,4)
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e2, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e0, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+3.16e-1, color='red', alpha=0.1) 
pylab.fill_between(bin_centers_e,bin_centers_e*0+1e-2,bin_centers_e*0+1e-1, color='red', alpha=0.1)  
plt.semilogy(bin_centers_b,bin_means_b[26,:],'k-')
pylab.fill_between(bin_centers_b,bin_low_b[26,:],bin_high_b[26,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1b; Fe->Mantle')
#plt.title('LP 890-9 b; Fe->Mantle')
#plt.title('Trappist-1f; Fe->Mantle')
#plt.ylabel('Total Present Surface Volatiles (bar)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e-1,2e4])
plt.xlim([20.345,22.7])

#plt.subplot(3,3,5)
#plt.plot(bin_centers,bin_means[8,:],'r')
#pylab.fill_between(bin_centers,bin_low[8,:],bin_high[8,:], color='red', alpha=0.4) 
#plt.ylabel('Final mantle redox (deltaFMQ)')
#plt.subplot(3,3,6)
#plt.semilogy(bin_centers,bin_means[9,:],'k',label='Solid')
#pylab.fill_between(bin_centers,bin_low[9,:],bin_high[9,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[10,:],'r',label='Atmo')
#pylab.fill_between(bin_centers,bin_low[10,:],bin_high[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
#plt.legend()
plt.subplot(4,4,7)
plt.semilogy(bin_centers_e,bin_means_e[11,:],'k-',label='CO$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[11,:],bin_high_e[11,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[12,:],'r-',label='CO')
pylab.fill_between(bin_centers_e,bin_low_e[12,:],bin_high_e[12,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[13,:],'c-',label='H$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[13,:],bin_high_e[13,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[1,:],'b-',label='H$_2$O')
pylab.fill_between(bin_centers_e,bin_low_e[1,:],bin_high_e[1,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[14,:],'m-',label='CH$_4$')
pylab.fill_between(bin_centers_e,bin_low_e[14,:],bin_high_e[14,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[15,:],'y-',label='O$_2$')
pylab.fill_between(bin_centers_e,bin_low_e[15,:],bin_high_e[15,:], color='yellow', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_means_e[16,:],'g-',label='Atmo H2O')
#pylab.fill_between(bin_centers_e,bin_low_e[16,:],bin_high_e[16,:], color='green', alpha=0.4) 
#plt.ylabel('Final Pressure (bar)')
plt.ylim([1e-3,2e4])
plt.xlim([20.345,22.7])
plt.legend(ncol=2)
plt.subplot(4,4,8)
plt.semilogy(bin_centers_b,bin_means_b[11,:],'k-',label='CO$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[11,:],bin_high_b[11,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[12,:],'r-',label='CO')
pylab.fill_between(bin_centers_b,bin_low_b[12,:],bin_high_b[12,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[13,:],'c-',label='H$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[13,:],bin_high_b[13,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[1,:],'b-',label='H$_2$O')
pylab.fill_between(bin_centers_b,bin_low_b[1,:],bin_high_b[1,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[14,:],'m-',label='CH$_4$')
pylab.fill_between(bin_centers_b,bin_low_b[14,:],bin_high_b[14,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[15,:],'y-',label='O$_2$')
pylab.fill_between(bin_centers_b,bin_low_b[15,:],bin_high_b[15,:], color='yellow', alpha=0.4) 
#plt.semilogy(bin_centers_b,bin_means_b[16,:],'g-',label='Atmo H2O')
#pylab.fill_between(bin_centers_b,bin_low_b[16,:],bin_high_b[16,:], color='green', alpha=0.4) 
#plt.ylabel('Final Pressure (bar)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([1e-3,2e4])
plt.xlim([20.345,22.7])
plt.legend(ncol=2)


#plt.subplot(3,3,9)
#plt.semilogy(bin_centers,bin_means[24,:],'r-',label='Surface')
#pylab.fill_between(bin_centers,bin_low[24,:],bin_high[24,:], color='red', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[25,:],'k-',label='Mantle')
#pylab.fill_between(bin_centers,bin_low[25,:],bin_high[25,:], color='grey', alpha=0.4) 
#plt.ylabel('Final temperature (K)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.legend()
#plt.subplot(3,3,8)
#plt.semilogy(bin_centers,bin_means[17,:],'k-',label='CO$_2$')
#pylab.fill_between(bin_centers,bin_low[17,:],bin_high[17,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[18,:],'r-',label='CO')
#pylab.fill_between(bin_centers,bin_low[18,:],bin_high[18,:], color='red', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[19,:],'c-',label='H$_2$')
#pylab.fill_between(bin_centers,bin_low[19,:],bin_high[19,:], color='cyan', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[20,:],'b-',label='H$_2$O')
#pylab.fill_between(bin_centers,bin_low[20,:],bin_high[20,:], color='blue', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[21,:],'m-',label='CH$_4$')
#pylab.fill_between(bin_centers,bin_low[21,:],bin_high[21,:], color='magenta', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[22,:],'y-',label='O$_2$')
#pylab.fill_between(bin_centers,bin_low[22,:],bin_high[22,:], color='yellow', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[23,:],'g-',label='Atmo H2O')
#pylab.fill_between(bin_centers,bin_low[23,:],bin_high[23,:], color='green', alpha=0.4) 
#plt.ylabel('Pressure at magma ocean solidification (bar)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.legend()


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#


#redo
[bin_centers_e,bin_low_e,bin_high_e,bin_means_e] = np.load('e_metalsink_REDO.npy',allow_pickle=True)
[bin_centers_b,bin_low_b,bin_high_b,bin_means_b] = np.load('e_nominal_REDO.npy',allow_pickle=True)


### Another for interesintg outcomes supplementary materials
plt.figure()
#plt.subplot(3,3,1)
#plt.semilogy(bin_centers,(bin_means[0,:]/(365*24*60*60)-1e7)/1e6,'k-')
#pylab.fill_between(bin_centers,(bin_low[0,:]/(365*24*60*60)-1e7)/1e6,(bin_high[0,:]/(365*24*60*60)-1e7)/1e6, color='grey', alpha=0.4) 
#plt.ylabel('Duration magma ocean (Myr)')
plt.subplot(4,4,9)
plt.semilogy(bin_centers_e,bin_means_e[0,:]/(365*24*60*60),'k')#,label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[0,:]/(365*24*60*60),bin_high_e[0,:]/(365*24*60*60), color='grey', alpha=0.4)  
plt.ylabel('Magma Ocean \nsolidification time (yrs)')
plt.ylim([1e7,1e10]) 
plt.xlim([20.345,22.7])

plt.subplot(4,4,10)
plt.semilogy(bin_centers_b,bin_means_b[0,:]/(365*24*60*60),'k')#,label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[0,:]/(365*24*60*60),bin_high_b[0,:]/(365*24*60*60), color='grey', alpha=0.4)  
#plt.ylabel('Final C reservoir (mol C)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e7,1e10]) 
plt.xlim([20.345,22.7])

plt.subplot(4,4,13)
plt.plot(bin_centers_e,bin_means_e[8,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[8,:],bin_high_e[8,:], color='grey', alpha=0.4)  
plt.ylabel('Final Mantle\nRedox State ($\Delta$FMQ)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([-7,8])
plt.xlim([20.345,22.7])

plt.subplot(4,4,14)
plt.plot(bin_centers_b,bin_means_b[8,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[8,:],bin_high_b[8,:], color='grey', alpha=0.4)  
#plt.ylabel('Final Mantle Redox (deltaFMQ)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([-7,8])
plt.xlim([20.345,22.7])


plt.subplot(4,4,1) 

plt.semilogy(bin_centers_e,bin_means_e[24,:],'k-',label='Surface')
pylab.fill_between(bin_centers_e,bin_low_e[24,:],bin_high_e[24,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[25,:],'r-',label='Mantle')
pylab.fill_between(bin_centers_e,bin_low_e[25,:],bin_high_e[25,:], color='red', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1e; Fe->Core')
#plt.title('LP 890-9 c; Fe->Core')
#plt.title('Prox Cen b; Fe->Core')
plt.ylabel('Final Temperature (K)')
plt.ylim([200,2500])
plt.xlim([20.345,22.7])
plt.legend()

plt.subplot(4,4,2) 
plt.semilogy(bin_centers_b,bin_means_b[24,:],'k-',label='Surface')
pylab.fill_between(bin_centers_b,bin_low_b[24,:],bin_high_b[24,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[25,:],'r-',label='Mantle')
pylab.fill_between(bin_centers_b,bin_low_b[25,:],bin_high_b[25,:], color='red', alpha=0.4) 
#plt.ylabel('Final Temperature (K)')
plt.ylim([200,2500])
plt.xlim([20.345,22.7])
plt.legend()
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1e; Fe->Mantle')
#plt.title('LP 890-9 c; Fe->Mantle')
#plt.title('Prox Cen b; Fe->Core')


#plt.subplot(3,3,5)
#plt.plot(bin_centers,bin_means[8,:],'r')
#pylab.fill_between(bin_centers,bin_low[8,:],bin_high[8,:], color='red', alpha=0.4) 
#plt.ylabel('Final mantle redox (deltaFMQ)')
#plt.subplot(3,3,6)
#plt.semilogy(bin_centers,bin_means[9,:],'k',label='Solid')
#pylab.fill_between(bin_centers,bin_low[9,:],bin_high[9,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[10,:],'r',label='Atmo')
#pylab.fill_between(bin_centers,bin_low[10,:],bin_high[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
#plt.legend()
plt.subplot(4,4,5)
plt.semilogy(bin_centers_e,bin_means_e[9,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[9,:],bin_high_e[9,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[10,:],'r',label='Atmo')
pylab.fill_between(bin_centers_e,bin_low_e[10,:],bin_high_e[10,:], color='red', alpha=0.4) 
plt.ylabel('Final free\nO reservoirs (kg)')
plt.ylim([1e19,1e25])
plt.xlim([20.345,22.7])
plt.legend()

plt.subplot(4,4,6)
plt.semilogy(bin_centers_b,bin_means_b[9,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[9,:],bin_high_b[9,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[10,:],'r',label='Atmo')
pylab.fill_between(bin_centers_b,bin_low_b[10,:],bin_high_b[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
plt.ylim([1e19,1e25])
plt.xlim([20.345,22.7])
plt.legend()


#redo
[bin_centers_e,bin_low_e,bin_high_e,bin_means_e] = np.load('b_metalsink_REDO.npy',allow_pickle=True)
[bin_centers_b,bin_low_b,bin_high_b,bin_means_b] = np.load('b_nominal_REDO.npy',allow_pickle=True)




#plt.subplot(3,3,1)
#plt.semilogy(bin_centers,(bin_means[0,:]/(365*24*60*60)-1e7)/1e6,'k-')
#pylab.fill_between(bin_centers,(bin_low[0,:]/(365*24*60*60)-1e7)/1e6,(bin_high[0,:]/(365*24*60*60)-1e7)/1e6, color='grey', alpha=0.4) 
#plt.ylabel('Duration magma ocean (Myr)')
plt.subplot(4,4,11)
plt.semilogy(bin_centers_e,bin_means_e[0,:]/(365*24*60*60),'k')#,label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[0,:]/(365*24*60*60),bin_high_e[0,:]/(365*24*60*60), color='grey', alpha=0.4)  
#plt.ylabel('Magma Ocean solidification time (yrs)')
plt.ylim([1e7,1e10]) 
plt.xlim([20.345,22.7])

plt.subplot(4,4,12)
plt.semilogy(bin_centers_b,bin_means_b[0,:]/(365*24*60*60),'k')#,label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[0,:]/(365*24*60*60),bin_high_b[0,:]/(365*24*60*60), color='grey', alpha=0.4)  
#plt.ylabel('Final C reservoir (mol C)')
#plt.xlabel('Initial H (log$_{10}$(kg))')
#plt.ylim([1e13,3e23])# mol
plt.ylim([1e7,1e10]) 
plt.xlim([20.345,22.7])

plt.subplot(4,4,15)
plt.plot(bin_centers_e,bin_means_e[8,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[8,:],bin_high_e[8,:], color='grey', alpha=0.4)  
#plt.ylabel('Final Mantle Redox State ($\Delta$FMQ)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([-7,8])
plt.xlim([20.345,22.7])

plt.subplot(4,4,16)
plt.plot(bin_centers_b,bin_means_b[8,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[8,:],bin_high_b[8,:], color='grey', alpha=0.4)  
#plt.ylabel('Final Mantle Redox State ($\Delta$FMQ)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.ylim([-7,8])
plt.xlim([20.345,22.7])


plt.subplot(4,4,3) 

plt.semilogy(bin_centers_e,bin_means_e[24,:],'k-',label='Surface')
pylab.fill_between(bin_centers_e,bin_low_e[24,:],bin_high_e[24,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[25,:],'r-',label='Mantle')
pylab.fill_between(bin_centers_e,bin_low_e[25,:],bin_high_e[25,:], color='red', alpha=0.4) 
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')
plt.title('Trappist-1b; Fe->Core')
#plt.title('LP 890-9 b; Fe->Core')
#plt.title('Trappist-1f; Fe->Core')
#plt.ylabel('Final Temperature (K)')
plt.ylim([200,2500])
plt.xlim([20.345,22.7])
plt.legend()

plt.subplot(4,4,4) 
plt.semilogy(bin_centers_b,bin_means_b[24,:],'k-',label='Surface')
pylab.fill_between(bin_centers_b,bin_low_b[24,:],bin_high_b[24,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[25,:],'r-',label='Mantle')
pylab.fill_between(bin_centers_b,bin_low_b[25,:],bin_high_b[25,:], color='red', alpha=0.4) 
#plt.ylabel('Final Temperature (K)')
plt.ylim([200,2500])
plt.xlim([20.345,22.7])
plt.legend()
#plt.semilogy(bin_centers_e,bin_centers_e*0+20,'r--')

plt.title('Trappist-1b; Fe->Mantle')
#plt.title('LP 890-9 b; Fe->Mantle')
#plt.title('Trappist-1f; Fe->Mantle')




#plt.subplot(3,3,5)
#plt.plot(bin_centers,bin_means[8,:],'r')
#pylab.fill_between(bin_centers,bin_low[8,:],bin_high[8,:], color='red', alpha=0.4) 
#plt.ylabel('Final mantle redox (deltaFMQ)')
#plt.subplot(3,3,6)
#plt.semilogy(bin_centers,bin_means[9,:],'k',label='Solid')
#pylab.fill_between(bin_centers,bin_low[9,:],bin_high[9,:], color='grey', alpha=0.4) 
#plt.semilogy(bin_centers,bin_means[10,:],'r',label='Atmo')
#pylab.fill_between(bin_centers,bin_low[10,:],bin_high[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
#plt.legend()
plt.subplot(4,4,7)
plt.semilogy(bin_centers_e,bin_means_e[9,:],'k',label='Solid')
pylab.fill_between(bin_centers_e,bin_low_e[9,:],bin_high_e[9,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_e,bin_means_e[10,:],'r',label='Atmo')
pylab.fill_between(bin_centers_e,bin_low_e[10,:],bin_high_e[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
plt.ylim([1e19,1e25])
plt.xlim([20.345,22.7])
plt.legend()

plt.subplot(4,4,8)
plt.semilogy(bin_centers_b,bin_means_b[9,:],'k',label='Solid')
pylab.fill_between(bin_centers_b,bin_low_b[9,:],bin_high_b[9,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers_b,bin_means_b[10,:],'r',label='Atmo')
pylab.fill_between(bin_centers_b,bin_low_b[10,:],bin_high_b[10,:], color='red', alpha=0.4) 
#plt.ylabel('Final free O reservoirs (kg)')
plt.ylim([1e19,1e25])
plt.xlim([20.345,22.7])
plt.legend()









plt.show()
