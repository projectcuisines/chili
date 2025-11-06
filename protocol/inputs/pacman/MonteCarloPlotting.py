import numpy as np
import pylab
from all_classes_NPM import *
import pylab as plt
import matplotlib.pyplot as pplt
import matplotlib 
matplotlib.use("TkAgg")
import os
import shutil

## Load Monte Carlo outputs

run_num = 74

#ET_outputs = np.load('TRAPPIST1b_Fe_core_outputs.npy',allow_pickle=True)  
#ET_inputs = np.load('TRAPPIST1b_Fe_core_inputs.npy',allow_pickle=True) 

#ET_outputs = np.load('TRAPPIST1b_Fe_mantle_outputs.npy',allow_pickle=True)   
#ET_inputs = np.load('TRAPPIST1b_Fe_mantle_inputs.npy',allow_pickle=True) 

#ET_outputs = np.load('TRAPPIST1e_Fe_mantle_outputs.npy',allow_pickle=True)    
#ET_inputs = np.load('TRAPPIST1e_Fe_mantle_inputs.npy',allow_pickle=True) 

#ET_outputs = np.load('Model_outputs.npy',allow_pickle=True)   
#ET_inputs = np.load('Model_inputs.npy',allow_pickle=True)

ET_outputs = np.load(f"simulation_results/Model_outputs_run{run_num}.npy",allow_pickle=True)   
ET_inputs = np.load(f"simulation_results/Model_inputs_run{run_num}.npy",allow_pickle=True)

EO = 1.57*10**20 #one earth ocean of hydrogen

plot_on = 1


def Plot_fun(i):
    fmt = ET_outputs[i].fmt
    total_time = ET_outputs[i].total_time
    total_y = ET_outputs[i].total_y
    if 2>1:
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    else:
        y_out = total_y[:,0:fmt]
        time = total_time[0:fmt]

    solid_time = total_time[fmt:]
    solid_y = total_y[:,fmt:]
    total_time_plot = (np.copy(total_time)/(365*24*60*60)-3e7)/1e6
    solid_time = (solid_time/(365*24*60*60)-3e7)/1e6
    time = (time/(365*24*60*60)-3e7)/1e6
    Mantle_carbon = solid_y[18,:]+solid_y[20,:] + solid_y[13,:]/0.012
    Mantle_hydrogen = solid_y[23,:] + solid_y[0,:]/0.001

    redox_state = ET_outputs[i].redox_state
    redox_state_solid = ET_outputs[i].redox_state_solid
    f_O2_mantle = ET_outputs[i].f_O2_mantle
    graph_check_arr = ET_outputs[i].graph_check_arr

    MoltenFe_in_FeO = ET_outputs[i].MoltenFe_in_FeO
    MoltenFe_in_FeO1pt5 = ET_outputs[i].MoltenFe_in_FeO1pt5
    MoltenFe_in_Fe = ET_outputs[i].MoltenFe_in_Fe

    F_FeO_ar = ET_outputs[i].F_FeO_ar
    F_FeO1_5_ar = ET_outputs[i].F_FeO1_5_ar
    F_Fe_ar = ET_outputs[i].F_Fe_ar


    '''
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time,y_out[7,:],label='Mantle')
    plt.plot(time,y_out[8,:],label='Surface',linestyle='--')
    plt.xlabel('Time (Myrs)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.subplot(2,2,2)
    plt.semilogy(time,y_out[9,:],label='OLR')
    plt.semilogy(time,y_out[10,:],label='ASR')
    plt.semilogy(time,y_out[11,:],label='q')
    plt.xlabel('Time (Myrs)')
    plt.legend()
    plt.ylabel('Fluxes')
    plt.subplot(2,2,3)
    plt.plot(time,y_out[2,:])
    plt.ylabel('Solidification radius (m)')
    plt.xlabel('Time (Myrs)')
    plt.subplot(2,2,4)
    plt.semilogy(time,y_out[30,:])
    plt.ylabel('Viscosity')
    plt.xlabel('Time (Myrs)')
    plt.figure()
    plt.plot(time,y_out[0,:])
    plt.plot(time,y_out[1,:])
    plt.ylabel('H')
    plt.figure()
    plt.plot(time,y_out[12,:])
    plt.plot(time,y_out[13,:])
    plt.ylabel('C')
    plt.figure()
    plt.plot(time,y_out[3,:])
    plt.plot(time,y_out[4,:])
    plt.ylabel('O')
    plt.figure()
    plt.semilogy(time,y_out[25,:],label='H2O')
    plt.semilogy(time,y_out[26,:],label='H2')
    plt.semilogy(time,y_out[27,:],label='CO2')
    plt.semilogy(time,y_out[28,:],label='CO')
    plt.semilogy(time,y_out[29,:],label='CH4')
    plt.semilogy(time,y_out[22,:]/1e5,label='O2')
    plt.xlabel('Time (Myrs)')
    plt.legend()
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.semilogy(time,y_out[18,:],label='n_C_diss')
    plt.semilogy(time,y_out[19,:],label='n_C_atm')
    plt.semilogy(time,y_out[20,:],label='n_C_graphite')
    plt.semilogy(time,y_out[18,:]+y_out[19,:]+y_out[20,:],label='n_C_sum',linestyle='--')
    plt.semilogy(time,y_out[12,:]/0.012,label='n_C_fluid',linestyle='-.')
    #plt.semilogy(time,graph_check_arr,label='graph_check_arr',linestyle = ':')
    plt.legend()
    plt.subplot(3,1,2)
    plt.loglog(time,y_out[21,:],label='n_H_atm')
    plt.loglog(time,y_out[23,:],label='n_H_diss')
    plt.loglog(time,y_out[1,:]/0.001,label='n_H_fluid',linestyle='--')
    plt.loglog(time,y_out[0,:]/0.001,label='n_H_solid',linestyle='-')
    plt.loglog(time,y_out[21,:]+y_out[23,:]+y_out[0,:]/0.001,label='n_H_sum',linestyle='-.')
    plt.loglog(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    plt.legend()
    plt.subplot(3,1,3)
    plt.loglog(time,y_out[17,:],label='Fe mol fraction in melt',linestyle=':')
    plt.legend()
    #total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001
    #H_solid = np.array((y_out[0,:]/0.001)/total_H)
    #plt.loglog(time,y_out[21,:]/total_H,label='H Atmo',color='c',linestyle = '-')
    #plt.loglog(solid_time,solid_y[21,:]/total_H[-1],color='c',linestyle = '-')
    #plt.loglog(time,y_out[23,:]/total_H,label='H Melt',color='m',linestyle = '-') #    plt.semilogy(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    #plt.loglog(time,y_out[31,:]/total_H,label='H in H2O in Melt',color='b',linestyle = ':') 
    #plt.loglog(time,H_solid,label='H Solid',color='y',linestyle='-')

    '''

    fO2_bar = y_out[22,:]/1e5
    TotalP = y_out[25,:]+y_out[26,:]+y_out[27,:]+y_out[28,:]+y_out[29,:]+fO2_bar


    #import pdb
    #pdb.set_trace()
    '''
    plt.figure()
    plt.plot(total_time_plot,redox_state,label='log(fO2) wrt FMQ')
    plt.plot(total_time_plot,redox_state_solid,label='solid log(fO2) wrt FMQ')
    plt.ylabel('Oxygen fugacity (delta FMQ)')
    plt.xlabel('Time (Myrs)')
    '''

    time = time*1e6
    total_time_plot = total_time_plot*1e6
    solid_time = solid_time*1e6

    '''
    plt.figure()
    plt.subplot(3,3,1)
    plt.title(round(ET_inputs[i][2].Init_fluid_H2O/EO,3))
    plt.semilogx(time,y_out[7,:],label='Mantle')
    plt.semilogx(time,y_out[8,:],label='Surface',linestyle='--')
    #plt.xlabel('Time (yrs)')
    plt.plot([np.max(time),np.max(time)],[0.9*np.min(y_out[8,:]),1.1*np.max(y_out[7,:])],'k--')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])
    plt.subplot(3,3,2)
    plt.loglog(time,y_out[9,:],label='OLR')
    plt.loglog(time,y_out[10,:],label='ASR')
    plt.loglog(time,y_out[11,:],label='Interior')
    plt.plot([np.max(time),np.max(time)],[0.9*np.min(y_out[11,:]),10.1*np.max(y_out[9,:])],'k--')
    #plt.xlabel('Time (yrs)')
    plt.ylabel('Flux (W/m$^2$)')
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])
    plt.subplot(3,3,3)
    plt.semilogx(time,y_out[2,:]/1000.0)
    plt.ylabel('Solidification radius (km)')
    #plt.xlabel('Time (yrs)')
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.max(time),np.max(time)],[0.9*np.min(y_out[2,:]/1000.0),1.1*np.max(y_out[2,:]/1000.0)],'k--')
    plt.subplot(3,3,4)
    plt.semilogx(total_time_plot,redox_state,label='Melt')
    plt.plot(total_time_plot,redox_state_solid,label='Solid')
    plt.ylabel('Melt oxygen fugacity ($\Delta$FMQ)')
    #plt.xlabel('Time (yrs)')
    plt.plot([np.max(time),np.max(time)],[0.9*np.min([np.min(redox_state),-3]),1.1*np.max([np.max(redox_state),3])],'k--')
    plt.xlim([1.0,np.max(solid_time)])
    plt.legend()
    plt.subplot(3,3,5)
    plt.loglog(time,y_out[25,:],label='H$_2$O')
    plt.loglog(time,y_out[26,:],label='H$_2$')
    plt.loglog(time,y_out[27,:],label='CO$_2$')
    plt.loglog(time,y_out[28,:],label='CO')
    plt.loglog(time,y_out[29,:],label='CH$_4$')
    plt.loglog(time,y_out[22,:]/1e5,label='O$_2$')
    plt.loglog(time,y_out[24,:]/1e5,label='Atmo H$_2$O',linestyle=':')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Pressure (bar)')
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([1e-6,10.1*np.max(y_out[25:29,:])])
    plt.plot([np.max(time),np.max(time)],[0.9*np.min(y_out[25:29,:]),20.1*np.max(y_out[25:29,:])],'k--')
    plt.legend()
    plt.subplot(3,3,6)
    total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001
    H_solid = np.array((y_out[0,:]/0.001)/total_H)
    plt.loglog(time,y_out[21,:]/total_H,label='H Atmo',color='c',linestyle = '-')
    plt.loglog(solid_time,solid_y[21,:]/total_H[-1],color='c',linestyle = '-')
    plt.loglog(time,y_out[23,:]/total_H,label='H Melt',color='m',linestyle = '-') #    plt.semilogy(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    plt.loglog(time,y_out[31,:]/total_H,label='H in H$_2$O in Melt',color='b',linestyle = ':') 
    plt.loglog(time,H_solid,label='H Solid',color='y',linestyle='-')
    plt.loglog(solid_time,Mantle_hydrogen/total_H[-1],color='y',linestyle = '-') #internal reservoirs after solidified
    total_C = y_out[18,:]+y_out[19,:] +y_out[20,:]+ y_out[13,:]/0.012
    C_solid = np.array((y_out[13,:]/0.012)/total_C)
    plt.loglog(time,y_out[19,:]/total_C,label='C Atmo',color='c',linestyle = '--')
    plt.loglog(solid_time,solid_y[19,:]/total_C[-1],color='c',linestyle = '--')
    plt.loglog(time,y_out[18,:]/total_C,label='C Melt',color='m',linestyle = '--')
    plt.loglog(time,y_out[20,:]/total_C,label='C Graphite',color='r',linestyle = '--')
    plt.loglog(time,C_solid,label='C Solid',color='y',linestyle = '--')
    plt.loglog(solid_time,Mantle_carbon/total_C[-1],color='y',linestyle = '--') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([1e-5,10])
    plt.plot([np.max(time),np.max(time)],[1e-10,5.0],'k--')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir fraction')
    plt.legend()

    plt.subplot(3,3,7)
    total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001 
    H_solid = np.array((y_out[0,:]/0.001))
    plt.loglog(time,total_H,label='Total H',color='k',linestyle = '-')
    plt.loglog(time,y_out[0,:]/0.001+ y_out[1,:]/0.001,label='Sum H',color='r',linestyle = '--')
    plt.loglog(solid_time,solid_time*0 + total_H[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[21,:],label='H Atmo',color='c',linestyle = '-')
    plt.loglog(solid_time,solid_y[21,:],color='c',linestyle = '-')
    plt.loglog(time,y_out[23,:],label='H Melt',color='m',linestyle = '-') #    plt.semilogy(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    plt.loglog(time,y_out[31,:],label='H in H$_2$O in Melt',color='b',linestyle = ':') 
    plt.loglog(time,H_solid,label='H Solid',color='y',linestyle='-')
    plt.loglog(solid_time,Mantle_hydrogen,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([1e18,5*np.max(total_H)])
    plt.plot([np.max(time),np.max(time)],[1e10,10*np.max(total_H)],'k--')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol H)')
    plt.legend()

    plt.subplot(3,3,8)
    total_C = y_out[18,:]+y_out[19,:] +y_out[20,:]+ y_out[13,:]/0.012
    C_solid = np.array((y_out[13,:]/0.012))
    plt.loglog(time,total_C,label='Total C',color='k',linestyle = '-')
    plt.loglog(time,y_out[12,:]/0.012+ y_out[13,:]/0.012,label='Sum C',color='r',linestyle = '--')
    plt.loglog(solid_time,solid_time*0 + total_C[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[19,:],label='C Atmo',color='c',linestyle = '--')
    plt.loglog(solid_time,solid_y[19,:],color='c',linestyle = '--')
    plt.loglog(time,y_out[18,:],label='C Melt',color='m',linestyle = '--')
    plt.loglog(time,y_out[20,:],label='C Graphite',color='r',linestyle = '--')
    plt.loglog(time,C_solid,label='C Solid',color='y',linestyle = '--')
    plt.loglog(solid_time,Mantle_carbon,color='y',linestyle = '--') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    #plt.ylim([1e18,5*np.max(total_C)])
    plt.plot([np.max(time),np.max(time)],[1e10,10*np.max(total_C)],'k--')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol C)')
    plt.legend()

    plt.subplot(3,3,9)
    O_solid_res = y_out[3,:]/0.016
    O_solid_res_solid = solid_y[3,:]/0.016
    total_O =  y_out[3,:]/0.016+y_out[4,:]/0.016 
    total_O_solid = solid_y[3,:]/0.016+solid_y[4,:]/0.016 
    Sum = y_out[16,:]+y_out[14,:] + y_out[15,:] # + O_solid_res
    Sum_solid = solid_y[16,:]+solid_y[14,:] + solid_y[15,:]#+O_solid_res_solid
    plt.plot(time,total_O,color='k',label='Total free O')
    plt.plot(solid_time,total_O_solid,color='k')
    plt.plot(time,y_out[15,:],color='c',label='O Molten Fe$_2$O$_3$')
    plt.plot(time,y_out[16,:],color='r',label='O Atmo')
    plt.plot(solid_time,solid_y[16,:],color='r')
    plt.plot(time,y_out[14,:],color='b',label='O Dissolved') #O_solid_res
    plt.plot(time,O_solid_res,color='y',label='O Solid') 
    #plt.loglog(time,y_out[4,:]/0.016,label='O fluid')
    plt.plot(time,Sum,color='m',linestyle = '--',label='sum fluids')
    plt.plot(time,Sum+y_out[3,:]/0.016,linestyle=':',label='sum fluids + y3')
    #plt.plot(solid_time,Sum_solid,color='m',linestyle = '--')
    plt.plot(time,y_out[4,:]/0.016 ,color='g',linestyle = ':',label='y4')
    plt.plot(solid_time,solid_y[4,:]/0.016 ,color='g',linestyle = ':')
    plt.xlim([1.0,np.max(solid_time)])
    #plt.ylim([1e20,5*np.max(total_O)])
    plt.plot([np.max(time),np.max(time)],[-1e18,10*np.max(total_O)],'k--')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol O)')
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e18)
    plt.legend()
    '''
    ###########################################################
    ### SINGLE RUN PLOT WITH ACCURATE SOLIDIFICATION RADIUS ###
    ###########################################################
    initH = ET_inputs[i][2].Init_fluid_H2O/EO
    alb = ET_inputs[i][1].albedoH
    #import pdb
    #pdb.set_trace()
    plt.figure(figsize=(12,12))
    #plt.suptitle('Illustrative time evolution showing transient habitability', fontsize=20, y=0.99)
    plt.suptitle(f"Run: {run_num}.{i}, Initial H: {initH:.2}EO, albedo: {alb}\nIllustrative time evolution showing transient habitability", x=0.5, y=1)
    plt.subplot(3,3,1)
    #plt.title('a) Temperature', fontsize=18, fontweight='bold', loc='left')
    plt.semilogx(time,y_out[7,:],label='Mantle')
    plt.semilogx(time,y_out[8,:],label='Surface',linestyle='--')
    #plt.xlabel('Time (yrs)')
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[8,:]),1.1*np.max(y_out[7,:])],'k--', label='Magma ocean\nsolidification')
    plt.ylabel('Temperature (K)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])

    plt.subplot(3,3,2)
    #plt.title('b) Flux', fontsize=18, fontweight='bold', loc='left')
    plt.loglog(time,y_out[9,:],label='OLR')
    plt.loglog(time,y_out[10,:],label='ASR')
    plt.loglog(time,y_out[11,:],label='Interior')
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[11,:]),10.1*np.max(y_out[9,:])],'k--', label='Magma ocean\nsolidification')
    #plt.xlabel('Time (yrs)')
    plt.ylabel('Flux (W/m$^2$)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])

    plt.subplot(3,3,3)
    #plt.title('c) Solidification radius', fontsize=18, fontweight='bold', loc='left')
    plt.semilogx(time,y_out[2,:]/1000.0)
    plt.ylabel('Solidification radius (km)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xlabel('Time (yrs)')
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[2,:]/1000.0),1.1*np.max(y_out[2,:]/1000.0)],'k--', label='Magma ocean\nsolidification')
    plt.legend()

    plt.subplot(3,3,4)
    #plt.title('d) Redox state', fontsize=18, fontweight='bold', loc='left')
    plt.semilogx(total_time_plot,redox_state,label='Magma ocean')
    plt.plot(total_time_plot,redox_state_solid,label='Solid mantle')
    plt.ylabel('Oxygen fugacity ($\Delta$FMQ)', fontsize=13)
    plt.xlabel('Time (yrs)')
    plt.plot([np.min(solid_time),np.min(solid_time)],[np.min(redox_state_solid[~np.isnan(redox_state_solid)])-3,1+np.max(redox_state_solid[~np.isnan(redox_state_solid)])],'k--')
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([np.min(redox_state_solid[~np.isnan(redox_state_solid)])-2,2+np.max(redox_state_solid[~np.isnan(redox_state_solid)])])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.subplot(3,3,5)
    #plt.title('e) Atmospheric gases', fontsize=18, fontweight='bold', loc='left')
    #plt.loglog(time,y_out[24,:]/1e5,label='H$_2$O (atmo)')
    plt.loglog(time,y_out[25,:],label='H$_2$O (total)')
    plt.loglog(time,y_out[26,:],label='H$_2$')
    plt.loglog(time,y_out[27,:],label='CO$_2$')
    plt.loglog(time,y_out[28,:],label='CO')
    plt.loglog(time,y_out[29,:],label='CH$_4$')
    plt.loglog(time,y_out[22,:]/1e5,label='O$_2$')
    plt.loglog(time,y_out[25,:]-y_out[24,:]/1e5,label='H$_2$O (liquid)',linestyle='--',color='b')
    plt.fill_between(time, y_out[25,:]-y_out[24,:]/1e5, 1e-10, where=(y_out[25,:]-y_out[24,:]/1e5 > 1e-10), interpolate=True, color='blue', alpha=0.15)
    plt.xlabel('Time (yrs)')
    plt.ylabel('Pressure (bar)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([1e-6,5.1*np.max(y_out[25:29,:])])
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[25:29,:]),20.1*np.max(y_out[25:29,:])],'k--')
    plt.legend()

    plt.subplot(3,3,6)
    #plt.title('f) Iron', fontsize=18, fontweight='bold', loc='left')
    summ = total_y[5,:]*1000.0/(56+1.5*16) + total_y[6,:]*1000.0/(56+16.)+total_y[30,:]*1000/56.0 + MoltenFe_in_FeO*1000.0/(56) + MoltenFe_in_FeO1pt5*1000.0/(56) + MoltenFe_in_Fe*1000/56.0
    plt.loglog(time,summ,label='Total Fe',color='k',linestyle='-')
    plt.loglog(total_time_plot,total_y[5,:]*1000.0/(56+1.5*16),label='Solid FeO$_{1.5}$')
    plt.loglog(total_time_plot,total_y[6,:]*1000.0/(56+16.),label='Solid FeO')
    plt.loglog(total_time_plot,total_y[30,:]*1000/56.0,label='Solid Fe')
    plt.loglog(time,MoltenFe_in_FeO*1000.0/(56),label='Molten FeO')
    plt.loglog(time,MoltenFe_in_FeO1pt5*1000.0/(56),label='Molten FeO$_{1.5}$')
    plt.loglog(time,MoltenFe_in_Fe*1000/56.0,label='Molten Fe')
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e17,10*np.max(summ)],'k--')
    plt.ylim([np.max(summ)*1e-5,np.max(summ)*5])
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir Fe (mol)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(ncol=2)
    #'''
    plt.subplot(3,3,7)
    #plt.title('g) Hydrogen', fontsize=18, fontweight='bold', loc='left')
    total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001 
    H_solid = np.array((y_out[0,:]/0.001))
    plt.loglog(time,total_H,label='Total H',color='k',linestyle = '-')
    #plt.loglog(time,y_out[0,:]/0.001+ y_out[1,:]/0.001,label='Sum H',color='r',linestyle = '--')
    plt.loglog(solid_time,solid_time*0 + total_H[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[21,:],label='H Atmo',color='c',linestyle = '-')
    plt.loglog(solid_time,solid_y[21,:],color='c',linestyle = '-')
    plt.loglog(time,y_out[23,:],label='H in Melt',color='m',linestyle = '-') #    plt.semilogy(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    plt.loglog(time,y_out[31,:],label='H in H$_2$O in Melt',color='tab:blue',linestyle = ':') 
    plt.loglog(time,H_solid,label='H Solid',color='y',linestyle='-')
    plt.loglog(solid_time,Mantle_hydrogen,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e10,10*np.max(total_H)],'k--')
    plt.ylim([1e18,5*np.max(total_H)])
    plt.xlabel('Time (yrs)', fontsize=15)
    plt.ylabel('Reservoir size (mol H)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.subplot(3,3,8)
    #plt.title('h) Carbon', fontsize=18, fontweight='bold', loc='left')
    total_C = y_out[18,:]+y_out[19,:] +y_out[20,:]+ y_out[13,:]/0.012
    C_solid = np.array((y_out[13,:]/0.012))
    plt.loglog(time,total_C,label='Total C',color='k',linestyle = '-')
    #plt.loglog(time,y_out[12,:]/0.012+ y_out[13,:]/0.012,label='Sum C',color='r',linestyle = '--')
    #plt.loglog(solid_time,solid_time*0 + total_C[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[19,:],label='C Atmo',color='c',linestyle = '--')
    plt.loglog(solid_time,solid_y[19,:],color='c',linestyle = '--')
    plt.loglog(time,y_out[18,:],label='C Melt',color='m',linestyle = '--')
    plt.loglog(time,y_out[20,:],label='C Graphite',color='r',linestyle = '--')
    plt.loglog(time,C_solid,label='C Solid',color='y',linestyle = '-')
    plt.loglog(solid_time,Mantle_carbon,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e10,10*np.max(total_C)],'k--')
    plt.ylim([1e18,5*np.max(total_C)])
    plt.xlabel('Time (yrs)', fontsize=15)
    plt.ylabel('Reservoir size (mol C)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.subplot(3,3,9)
    #plt.title('i) Oxygen', fontsize=18, fontweight='bold', loc='left')
    O_solid_res = y_out[3,:]/0.016
    O_solid_res_solid = solid_y[3,:]/0.016
    total_O =  y_out[3,:]/0.016+y_out[4,:]/0.016 
    total_O_solid = solid_y[3,:]/0.016+solid_y[4,:]/0.016 
    Sum = y_out[16,:]+y_out[14,:] + y_out[15,:] # + O_solid_res
    Sum_solid = solid_y[16,:]+solid_y[14,:] + solid_y[15,:]#+O_solid_res_solid
    plt.plot(time,total_O,color='k',label='Total free O')
    plt.plot(solid_time,total_O_solid,color='k')
    plt.plot(time,y_out[15,:],color='c',label='O Molten Fe$_2$O$_3$')
    plt.plot(time,y_out[16,:],color='r',label='O Atmo')
    plt.plot(solid_time,solid_y[16,:],color='r')
    plt.plot(time,y_out[14,:],color='tab:blue',label='O Dissolved') #O_solid_res
    plt.plot(time,O_solid_res,color='y',label='O Solid') 
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[-1e18,10*np.max(total_O)],'k--')
    plt.ylim([1e18,5*np.max(total_O)])
    plt.xlabel('Time (yrs)', fontsize=15)
    plt.ylabel('Reservoir size (mol O)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e18)
    plt.legend()

    #'''
    plt.tight_layout()
    #(y_out[15,:]*0.016*0.5)*(.056*2/.016) #kg Fe

    #plt.show()
    #total_y_og = np.copy(total_y)

    #plt.show()


    

    '''
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.)  + Total_Fe_mol_fraction_start * (56.0+16.0)
    #Simple_molten  = ((Total_Fe_mol_fraction_start-y[17])*(56)/m_sil + y[17]*(56+16)/m_sil  )*(4./3. * np.pi * pm * (rp**3 - total_y[2,:]**3))
    Simple_molten = total_y_og[17,:]  *(4./3. * np.pi * pm * (rp**3 - total_y[2,:]**3))
    #MoltenFe_in_FeO = y_out[17,:]
    plt.figure() #y[16] = n_O_atm; 29 is n_O_diss
    plt.subplot(3,2,1) 
    Oxidized_Iron = total_y_og[5,:]*56/(56+1.5*16)
    Reduced_Iron = total_y_og[6,:]*56/(56+16)  
    plt.loglog(total_time_plot,Oxidized_Iron/(Oxidized_Iron + Reduced_Iron))
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Fe oxidized frac')
    plt.legend()
    plt.subplot(3,2,2)
    plt.loglog(total_time_plot,f_O2_mantle,label='f_O2_mantle')
    plt.loglog(time,fO2_bar,label='fO2_bar')
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('fO2 (bar)')
    plt.legend()
    plt.subplot(3,2,3)
    plt.loglog(total_time_plot,Oxidized_Iron,label='y5')
    plt.loglog(total_time_plot,Reduced_Iron,label='y6')
    plt.loglog(total_time_plot,total_y_og[30,:],label='y30')
    plt.loglog(time,MoltenFe_in_FeO_before,label='Molten FeO')
    plt.loglog(time,MoltenFe_in_FeO1pt5_before,label='Molten FeO1.5')
    plt.loglog(time,MoltenFe_in_Fe_before,label='Molten Fe')
    plt.loglog(time,MoltenFe_in_FeO_before2,label='Molten FeO 2',linestyle='--')
    plt.loglog(time,MoltenFe_in_FeO1pt5_before2,label='Molten FeO1.5 2',linestyle='--')
    plt.loglog(time,MoltenFe_in_Fe_before2,label='Molten Fe 2',linestyle='--')
    ALL_Iron = Oxidized_Iron + Reduced_Iron + total_y_og[30,:] + MoltenFe_in_FeO + MoltenFe_in_FeO1pt5 + MoltenFe_in_Fe
    ALL_Iron2 = Oxidized_Iron + Reduced_Iron + total_y_og[30,:] + F_FeO_ar* 56/(56.0 + 16.0) + F_FeO1_5_ar* 56/(56.0 + 1.5*16.0) + F_Fe_ar
    plt.loglog(total_time_plot,ALL_Iron,label='total',color='k',linestyle='--')
    plt.loglog(total_time_plot,ALL_Iron2,label='total2',color='c',linestyle=':')
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Solid FeO and FeO1.5 (kg Fe)')
    plt.legend()
    plt.subplot(3,2,4)
    plt.loglog(total_time_plot,total_y_og[5,:]*1000.0/(56+1.5*16),label='FeO1.5')
    plt.loglog(total_time_plot,total_y_og[6,:]*1000.0/(56+16.),label='FeO')
    plt.loglog(total_time_plot,total_y_og[30,:]*1000/56.0,label='Fe')
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Solid mol Fe')
    plt.legend()
    plt.subplot(3,2,5)
    Total_solid_Iron = Oxidized_Iron + Reduced_Iron  + total_y_og[30,:]
    Total_molten_Iron = MoltenFe_in_FeO + MoltenFe_in_FeO1pt5 + MoltenFe_in_Fe
    Molten_check = F_FeO_ar* 56/(56.0 + 16.0) + F_FeO1_5_ar * 56/(56.0 + 1.5*16.0) + F_Fe_ar
    plt.loglog(total_time_plot,Total_solid_Iron,label='Solid Fe')
    plt.loglog(total_time_plot,Total_molten_Iron,label='Molten Fe')
    plt.loglog(total_time_plot,Molten_check,label='Molten_check',linestyle='--')
    plt.loglog(total_time_plot,Simple_molten,label='Simple_molten',linestyle=':')
    ##ALL_Iron = total_y[5,:] + total_y[6,:] + total_y[30,:] + MoltenFe_in_FeO + total_y[15,:]*0.056 + MoltenFe_in_Fe
    plt.loglog(total_time_plot,ALL_Iron,label='Total Fe',color='k',linestyle = '-')
    #plt.loglog(total_time_plot,Total_solid_Iron+Molten_check,label='Total Fe again',color='r',linestyle = '--')
    plt.loglog(total_time_plot,Total_solid_Iron+Simple_molten,label='Total Fe again again',color='y',linestyle = ':')
    plt.xlim([1.0,np.max(total_time_plot)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Solid and Molten iron (kg Fe)')
    plt.legend()

    O_solid_res = (2*y_out[13,:]/0.012) +  (0.5*y_out[0,:]/0.001) + (y_out[5,:]*0.5*16/(56+1.5*16))/0.016
    total_O =  y_out[3,:]/0.016+y_out[4,:]/0.016 
    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(time,total_O,color='b',label='total_O')
    plt.loglog(time,y_out[15,:],color='c',label='O molten Fe')
    plt.loglog(time,y_out[16,:],color='r',label='O atm')
    plt.loglog(time,y_out[14,:],color='k',label='O diss') #O_solid_res
    plt.loglog(time,O_solid_res,color='y',label='O_solid_res') 
    plt.loglog(time,y_out[16,:]+y_out[14,:]+ O_solid_res + y_out[15,:]  ,color='m',linestyle = '--',label='sum')
    plt.xlim([1.0,np.max(time)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir moles')
    plt.legend()
    plt.subplot(2,1,2)
    #plt.loglog(time,total_O,color='b',label='total_O')
    plt.loglog(time,(y_out[16,:]+y_out[14,:])/total_O,color='c',label='O atm + diss')
    plt.loglog(time,(y_out[3,:]/0.016)/total_O,color='b',label='y3 - solid')
    plt.loglog(time,(y_out[4,:]/0.016)/total_O,color='m',label='y4 - fluid ')
    plt.loglog(time,(y_out[3,:]/0.016+y_out[4,:]/0.016)/total_O,color='r',linestyle = '--',label='y3+y4')
    plt.xlim([1.0,np.max(time)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir fraction')
    plt.legend()

    plt.figure()
    plt.subplot(2,2,1)
    plt.loglog(time,total_H,color='b',label='total_H')
    plt.loglog(time,total_C,color='c',label='total_C')
    plt.xlim([1.0,np.max(time)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir moles')
    plt.subplot(2,2,2)
    plt.semilogx(time,y_out[14,:])
    plt.xlim([1.0,np.max(time)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('H2+H2O frac')
    plt.legend()
    plt.subplot(2,2,3)
    #total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001
    #H_solid = np.array((y_out[0,:]/0.001)/total_H)
    plt.loglog(time,y_out[21,:],label='H Atmo',color='c',linestyle = '-')
    plt.loglog(solid_time,solid_y[21,:],color='c',linestyle = '-')
    plt.loglog(time,y_out[23,:],label='H Melt',color='m',linestyle = '-')
    plt.loglog(time,y_out[0,:]/0.001,label='H Solid',color='y',linestyle='-')
    #plt.loglog(time,y_out[21,:]+y_out[23,:]+y_out[0,:]/0.001,color = 'k',linestyle = '-')
    #plt.loglog(time,total_H,color='b',label='total_H',linestyle = '--')
    plt.loglog(solid_time,Mantle_hydrogen,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.ylabel('H Reservoir moles')
    plt.legend()
    plt.subplot(2,2,4)
    plt.loglog(time,y_out[19,:],label='C Atmo',color='c',linestyle = '--')
    plt.loglog(solid_time,solid_y[19,:],color='c',linestyle = '--')
    plt.loglog(time,y_out[18,:],label='C Melt',color='m',linestyle = '--')
    plt.loglog(time,y_out[20,:],label='C Graphite',color='k',linestyle = '--')
    plt.loglog(time,y_out[13,:]/0.012,label='C Solid',color='y',linestyle = '--')
    #plt.loglog(time,y_out[19,:] + y_out[18,:] + y_out[20,:] + y_out[13,:]/0.012,color = 'k')
    #plt.loglog(time,total_C,color='b',label='total_C',linestyle = '--')
    plt.loglog(solid_time,Mantle_carbon,color='y',linestyle = '--') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.max(time),np.max(time)],[1e-10,1.0],'k-')
    plt.xlabel('Time (yrs)')
    plt.ylabel('C Reservoir moles')
    plt.legend()


    #np.save('output_ex4',[time,y_out,redox_state])
    #np.save('output_ex_test2',[time,y_out,redox_state])
    plt.show()
    #'''

def Plot_phi_vol_frac(i):
    fmt = ET_outputs[i].fmt
    total_time = ET_outputs[i].total_time
    total_y = ET_outputs[i].total_y
    if 2>1:
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    else:
        y_out = total_y[:,0:fmt]
        time = total_time[0:fmt]

    solid_time = total_time[fmt:]
    solid_y = total_y[:,fmt:]
    total_time_plot = (np.copy(total_time)/(365*24*60*60)-3e7)/1e6
    solid_time = (solid_time/(365*24*60*60)-3e7)/1e6
    time = (time/(365*24*60*60)-3e7)/1e6
    fO2_bar = y_out[22,:]/1e5
    TotalP = y_out[25,:]+y_out[26,:]+y_out[27,:]+y_out[28,:]+y_out[29,:]+fO2_bar

    time = time*1e6
    total_time_plot = total_time_plot*1e6
    solid_time = solid_time*1e6

    plt.figure()
    plt.plot(time, y_out[14,:],color='r')
    plt.xlabel('Time (yrs)', fontsize=15)
    plt.ylabel('Phi (volume fraction)', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([0,1])
    plt.xscale('log')
    #plt.legend()

    



'''
    #import pdb
    #pdb.set_trace()

    #Fe, C, and O reservior subplots (leave out for more succinct plot)
    plt.subplot(3,3,7)
    summ = total_y[5,:]*1000.0/(56+1.5*16) + total_y[6,:]*1000.0/(56+16.)+total_y[30,:]*1000/56.0 + MoltenFe_in_FeO*1000.0/(56) + MoltenFe_in_FeO1pt5*1000.0/(56) + MoltenFe_in_Fe*1000/56.0
    plt.loglog(time,summ,label='Total Fe',color='k',linestyle='-')
    plt.loglog(total_time_plot,total_y[5,:]*1000.0/(56+1.5*16),label='Solid FeO$_{1.5}$')
    plt.loglog(total_time_plot,total_y[6,:]*1000.0/(56+16.),label='Solid FeO')
    plt.loglog(total_time_plot,total_y[30,:]*1000/56.0,label='Solid Fe')
    plt.loglog(time,MoltenFe_in_FeO*1000.0/(56),label='Molten FeO')
    plt.loglog(time,MoltenFe_in_FeO1pt5*1000.0/(56),label='Molten FeO$_{1.5}$')
    plt.loglog(time,MoltenFe_in_Fe*1000/56.0,label='Molten Fe')
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e17,10*np.max(summ)],'k--')
    plt.xlim([1.0,np.max(total_time_plot)])
    #plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir Fe (mol)')
    plt.legend(ncol=2)

    plt.subplot(3,3,8)
    total_C = y_out[18,:]+y_out[19,:] +y_out[20,:]+ y_out[13,:]/0.012
    C_solid = np.array((y_out[13,:]/0.012))
    plt.loglog(time,total_C,label='Total C',color='k',linestyle = '-')
    #plt.loglog(time,y_out[12,:]/0.012+ y_out[13,:]/0.012,label='Sum C',color='r',linestyle = '--')
    #plt.loglog(solid_time,solid_time*0 + total_C[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[19,:],label='C Atmo',color='c',linestyle = '--')
    plt.loglog(solid_time,solid_y[19,:],color='c',linestyle = '--')
    plt.loglog(time,y_out[18,:],label='C Melt',color='m',linestyle = '--')
    plt.loglog(time,y_out[20,:],label='C Graphite',color='r',linestyle = '--')
    plt.loglog(time,C_solid,label='C Solid',color='y',linestyle = '-')
    plt.loglog(solid_time,Mantle_carbon,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e10,10*np.max(total_C)],'k--')
    plt.ylim([1e18,5*np.max(total_C)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol C)')
    plt.legend()

    plt.subplot(3,3,9)
    O_solid_res = y_out[3,:]/0.016
    O_solid_res_solid = solid_y[3,:]/0.016
    total_O =  y_out[3,:]/0.016+y_out[4,:]/0.016 
    total_O_solid = solid_y[3,:]/0.016+solid_y[4,:]/0.016 
    Sum = y_out[16,:]+y_out[14,:] + y_out[15,:] # + O_solid_res
    Sum_solid = solid_y[16,:]+solid_y[14,:] + solid_y[15,:]#+O_solid_res_solid
    plt.plot(time,total_O,color='k',label='Total free O')
    plt.plot(solid_time,total_O_solid,color='k')
    plt.plot(time,y_out[15,:],color='c',label='O Molten Fe$_2$O$_3$')
    plt.plot(time,y_out[16,:],color='r',label='O Atmo')
    plt.plot(solid_time,solid_y[16,:],color='r')
    plt.plot(time,y_out[14,:],color='b',label='O Dissolved') #O_solid_res
    plt.plot(time,O_solid_res,color='y',label='O Solid') 
    #plt.loglog(time,y_out[4,:]/0.016,label='O fluid')
    #plt.plot(time,Sum,color='m',linestyle = '--',label='sum fluids')
    #plt.plot(time,Sum+y_out[3,:]/0.016,linestyle=':',label='sum fluids + y3')
    #plt.plot(solid_time,Sum_solid,color='m',linestyle = '--')
    #plt.plot(time,y_out[4,:]/0.016 ,color='g',linestyle = ':',label='y4')
    #plt.plot(solid_time,solid_y[4,:]/0.016 ,color='g',linestyle = ':')
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[-1e18,10*np.max(total_O)],'k--')
    plt.ylim([1e18,5*np.max(total_O)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol O)')
    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e18)
    plt.legend()
    #(y_out[15,:]*0.016*0.5)*(.056*2/.016) #kg Fe
    #'''

### Generate output compilations for Monte Carlo calculations:

init_O_plot = []
init_CO2_plot =[]

init_O_fail_plot = []
init_CO2_fail_plot =[]

init_H_plot = []
Magma_solid_time = []
Final_surface_water = []
Final_fluid_H = []
Final_solid_H = []
Final_atmo_C = []
Final_solid_C = []
redox_state_solid_ar = []
Solid_O = []
Atmo_O = []

Atmo_CO2 = []
Atmo_CO = []
Atmo_H2 = []
Atmo_CH4 = []
Atmo_H2O = []
Oxygen_fug = []

Atmo_CO2_fmt = []
Atmo_CO_fmt = []
Atmo_H2_fmt = []
Atmo_CH4_fmt = []
Atmo_H2O_fmt = []
Oxygen_fug_fmt = []
Final_surface_water_fmt = []

Final_surface_temp = []
Final_mantle_temp = []

Final_FeO = []
Final_Fe = []
Final_FeO1pt5 = []
Final_Total_Fe = []

# add redox state, magma ocean duration etc. here? Already added actually
success=0
fail=0

for j in range(0,np.shape(ET_outputs)[0]):
    #print(j)
    init_O_plot.append(ET_inputs[j][2].Init_fluid_O)
    init_CO2_plot.append(ET_inputs[j][2].Init_fluid_CO2)  
     
    try:

        Atmo_CO2.append(ET_outputs[j].total_y[27,-1])
        Atmo_CO.append(ET_outputs[j].total_y[28,-1])
        Atmo_H2.append(ET_outputs[j].total_y[26,-1])
        Atmo_CH4.append(ET_outputs[j].total_y[29,-1])
        Atmo_H2O.append(ET_outputs[j].total_y[24,-1])
        Oxygen_fug.append(ET_outputs[j].total_y[22,-1])
        Final_surface_water.append(ET_outputs[j].total_y[25,-1])
        init_H_plot.append(ET_inputs[j][2].Init_fluid_H2O) 

        Atmo_CO2_fmt.append(ET_outputs[j].total_y[27,ET_outputs[j].fmt])
        Atmo_CO_fmt.append(ET_outputs[j].total_y[28,ET_outputs[j].fmt])
        Atmo_H2_fmt.append(ET_outputs[j].total_y[26,ET_outputs[j].fmt])
        Atmo_CH4_fmt.append(ET_outputs[j].total_y[29,ET_outputs[j].fmt])
        Atmo_H2O_fmt.append(ET_outputs[j].total_y[24,ET_outputs[j].fmt])
        Oxygen_fug_fmt.append(ET_outputs[j].total_y[22,ET_outputs[j].fmt])
        Final_surface_water_fmt.append(ET_outputs[j].total_y[25,ET_outputs[j].fmt])

        Final_surface_temp.append(ET_outputs[j].total_y[8,-1])
        Final_mantle_temp.append(ET_outputs[j].total_y[7,-1])

        Magma_solid_time.append(ET_outputs[j].total_time[ET_outputs[j].fmt])
        Final_fluid_H.append(ET_outputs[j].total_y[1,-1])
        Final_solid_H.append(ET_outputs[j].total_y[0,-1])
        Final_atmo_C.append(ET_outputs[j].total_y[19,-1])
        Final_solid_C.append(ET_outputs[j].total_y[13,-1]/0.012)
        redox_state_solid_ar.append(ET_outputs[j].redox_state_solid[-1])
        Solid_O.append(ET_outputs[j].total_y[3,-1]/0.016)
        Atmo_O.append(ET_outputs[j].total_y[16,-1])
        
        #import pdb
        #pdb.set_trace()
        MoltenFe_in_FeO = ET_outputs[j].MoltenFe_in_FeO[-1]
        MoltenFe_in_FeO1pt5 = ET_outputs[j].MoltenFe_in_FeO1pt5[-1]
        MoltenFe_in_Fe = ET_outputs[j].MoltenFe_in_Fe[-1]

        Final_FeO.append(ET_outputs[j].total_y[6,-1]*1000.0/(56+16) + MoltenFe_in_FeO*1000.0/(56) )
        Final_Fe.append(ET_outputs[j].total_y[30,-1]*1000/56.0 + MoltenFe_in_Fe*1000/56.0 )
        Final_FeO1pt5.append(ET_outputs[j].total_y[5,-1]*1000.0/(56+1.5*16) + MoltenFe_in_FeO1pt5*1000.0/(56) )
        Final_Total_Fe.append(ET_outputs[j].total_y[6,-1]*1000.0/(56+16) + MoltenFe_in_FeO*1000.0/(56)+ET_outputs[j].total_y[30,-1]*1000/56.0 + MoltenFe_in_Fe*1000/56.0+ET_outputs[j].total_y[5,-1]*1000.0/(56+1.5*16) + MoltenFe_in_FeO1pt5*1000.0/(56))
        success=success+1
    except:
        pass
        '''
        Magma_solid_time.append(np.nan)
        #Final_surface_water.append(np.nan)
        Final_fluid_H.append(np.nan)
        Final_solid_H.append(np.nan)
        Final_atmo_C.append(np.nan)
        Final_solid_C.append(np.nan)
        redox_state_solid_ar.append(np.nan)
        Solid_O.append(np.nan)
        Atmo_O.append(np.nan)

        Atmo_CO2.append(np.nan)
        Atmo_CO.append(np.nan)
        Atmo_H2.append(np.nan)
        Atmo_CH4.append(np.nan)
        Atmo_H2O.append(np.nan)
        Oxygen_fug.append(np.nan)
        Final_surface_water.append(np.nan)

        Final_surface_temp.append(np.nan)
        Final_mantle_temp.append(np.nan)

        Atmo_CO2_fmt.append(np.nan)
        Atmo_CO_fmt.append(np.nan)
        Atmo_H2_fmt.append(np.nan)
        Atmo_CH4_fmt.append(np.nan)
        Atmo_H2O_fmt.append(np.nan)
        Oxygen_fug_fmt.append(np.nan)
        Final_surface_water_fmt.append(np.nan)
        '''
        init_O_fail_plot.append(ET_inputs[j][2].Init_fluid_O)
        init_CO2_fail_plot.append(ET_inputs[j][2].Init_fluid_CO2)
        fail=fail+1
        #print('nope')

'''
init_O_fail_plot = np.array(init_O_fail_plot)
init_CO2_fail_plot = np.array(init_CO2_fail_plot)
init_O_plot = np.array(init_O_plot)
init_CO2_plot = np.array(init_CO2_plot)

pylab.figure()
plt.subplot(2,2,1)
pylab.hist(np.log10(init_O_plot),color='r')
plt.subplot(2,2,2)
pylab.hist(np.log10(init_O_fail_plot),color='g')
plt.subplot(2,2,3)
pylab.hist(np.log10(init_CO2_plot),color='r')
plt.subplot(2,2,4)
pylab.hist(np.log10(init_CO2_fail_plot),color='g')
'''

print ('Success rate:',100*success/(fail+success),'%')
Atmo_CO2  = np.array(Atmo_CO2)
Atmo_CO  = np.array(Atmo_CO)
Atmo_H2  = np.array(Atmo_H2)
Atmo_CH4  = np.array(Atmo_CH4)
Atmo_H2O  = np.array(Atmo_H2O)/1e5
Oxygen_fug  = np.array(Oxygen_fug)/1e5
Final_surface_water= np.array(Final_surface_water)

Atmo_CO2_fmt  = np.array(Atmo_CO2_fmt)
Atmo_CO_fmt  = np.array(Atmo_CO_fmt)
Atmo_H2_fmt  = np.array(Atmo_H2_fmt)
Atmo_CH4_fmt = np.array(Atmo_CH4_fmt)
Atmo_H2O_fmt = np.array(Atmo_H2O_fmt)/1e5
Oxygen_fug_fmt = np.array(Oxygen_fug_fmt)/1e5
Final_surface_water_fmt= np.array(Final_surface_water_fmt)

Final_surface_temp = np.array(Final_surface_temp)
Final_mantle_temp = np.array(Final_mantle_temp)

Magma_solid_time = np.array(Magma_solid_time)
init_H_plot = np.array(init_H_plot)
Final_fluid_H = np.array(Final_fluid_H)
Final_solid_H = np.array(Final_solid_H)
Final_atmo_C = np.array(Final_atmo_C)
Final_solid_C = np.array(Final_solid_C)
redox_state_solid_ar = np.array(redox_state_solid_ar)
Solid_O = np.array(Solid_O)
Atmo_O = np.array(Atmo_O)

Final_FeO = np.array(Final_FeO)
Final_Fe = np.array(Final_Fe)       
Final_FeO1pt5 = np.array(Final_FeO1pt5)
Final_Total_Fe = np.array(Final_Total_Fe)

def low_fun(inpu):
    out_for_fun = scipy.stats.scoreatpercentile(inpu ,[16], interpolation_method='fraction',axis=0)
    return out_for_fun[0]

def high_fun(inpu):
    out_for_fun = scipy.stats.scoreatpercentile(inpu ,[84], interpolation_method='fraction',axis=0)
    return out_for_fun[0]

'''
## this works, just need to figure out a way to combine output files, and rewrite plotting script below
import scipy.stats
num_of_H_bins = 5
values = [Magma_solid_time,Final_surface_water,Final_solid_H,Final_fluid_H,Final_fluid_H+Final_solid_H,Final_atmo_C,Final_solid_C,Final_atmo_C+Final_solid_C,redox_state_solid_ar,Solid_O,Atmo_O, Atmo_CO2,Atmo_CO,Atmo_H2,Atmo_CH4,Oxygen_fug,Atmo_H2O,Atmo_CO2_fmt,Atmo_CO_fmt,Atmo_H2_fmt,Final_surface_water_fmt,Atmo_CH4_fmt,Oxygen_fug_fmt,Atmo_H2O_fmt,Final_surface_temp,Final_mantle_temp, Atmo_CO2+Atmo_CO+Atmo_H2+Atmo_CH4+Oxygen_fug+Final_surface_water,Final_FeO,Final_Fe,Final_FeO1pt5,Final_Total_Fe] #can combine multiple outputs all at once.
#init_water = 10**np.linspace(20,23,10) #binned init H
bin_means,bin_edges,binnumber=scipy.stats.binned_statistic(np.log10(init_H_plot),values,statistic='median',bins=num_of_H_bins)
bin_std,bin_edges,binnumber=scipy.stats.binned_statistic(np.log10(init_H_plot),values,statistic='std',bins=num_of_H_bins)
bin_low,bin_edges,binnumber=scipy.stats.binned_statistic(np.log10(init_H_plot),values,statistic=low_fun,bins=num_of_H_bins)
bin_high,bin_edges,binnumber=scipy.stats.binned_statistic(np.log10(init_H_plot),values,statistic=high_fun,bins=num_of_H_bins)

bin_width=bin_edges[1] - bin_edges[0]
bin_centers = bin_edges[1:] - bin_width/2

#Optional save binned outputs for combining Monte Carlo outputs from different scenarios into single plot 

#np.save('e_metalsink_REDO',np.array([bin_centers,bin_low,bin_high,bin_means]))
#np.save('e_nominal_REDO',np.array([bin_centers,bin_low,bin_high,bin_means]))
#np.save('b_metalsink_REDO',np.array([bin_centers,bin_low,bin_high,bin_means]))
#np.save('b_nominal_REDO',np.array([bin_centers,bin_low,bin_high,bin_means]))
"""
plt.figure()
plt.semilogy(bin_centers,bin_means[27,:],'b',label='FeO')
pylab.fill_between(bin_centers,bin_low[27,:],bin_high[27,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[28,:],'g',label='Fe')
pylab.fill_between(bin_centers,bin_low[28,:],bin_high[28,:], color='green', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[29,:],'r',label='FeO1.5')
pylab.fill_between(bin_centers,bin_low[29,:],bin_high[29,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[30,:],'k',label='Total')
pylab.fill_between(bin_centers,bin_low[30,:],bin_high[30,:], color='grey', alpha=0.4) 
plt.legend()

plt.figure()
plt.plot(bin_centers,bin_means[2,:],'k')
#plt.semilogy(bin_centers,bin_low[2,:],'k--')
#plt.semilogy(bin_centers,bin_high[2,:],'k--')
pylab.fill_between(bin_centers,bin_low[2,:],bin_high[2,:], color='grey', alpha=0.4)  
#plt.show()


plt.figure()
plt.subplot(3,3,1)
plt.semilogy(bin_centers,(bin_means[0,:]/(365*24*60*60)-1e7)/1e6,'k-')
pylab.fill_between(bin_centers,(bin_low[0,:]/(365*24*60*60)-1e7)/1e6,(bin_high[0,:]/(365*24*60*60)-1e7)/1e6, color='grey', alpha=0.4) 
plt.ylabel('Duration magma ocean (Myr)')
plt.subplot(3,3,2)
plt.semilogy(bin_centers,bin_means[5,:],'b',label='Atmo')
pylab.fill_between(bin_centers,bin_low[5,:],bin_high[5,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[6,:],'k',label='Solid')
pylab.fill_between(bin_centers,bin_low[6,:],bin_high[6,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[7,:],'r',label='Total')
pylab.fill_between(bin_centers,bin_low[7,:],bin_high[7,:], color='red', alpha=0.4) 
plt.ylabel('Final C reservoirs (mol C)')
plt.legend()
plt.subplot(3,3,3)
plt.semilogy(bin_centers,bin_means[2,:],'k',label='Solid')
pylab.fill_between(bin_centers,bin_low[2,:],bin_high[2,:], color='grey', alpha=0.4)  
plt.semilogy(bin_centers,bin_means[3,:],'b',label='Fluid')
pylab.fill_between(bin_centers,bin_low[3,:],bin_high[3,:], color='blue', alpha=0.4)  
plt.semilogy(bin_centers,bin_means[4,:],'r',label='Total')
pylab.fill_between(bin_centers,bin_low[4,:],bin_high[4,:], color='red', alpha=0.4)  
plt.ylabel('Final H reservoirs (kg)')
plt.legend()
plt.subplot(3,3,4) 
plt.semilogy(bin_centers,bin_means[26,:],'k-')
pylab.fill_between(bin_centers,bin_low[26,:],bin_high[26,:], color='grey', alpha=0.4) 
plt.ylabel('Total Surface Volatiles (bar)')
plt.subplot(3,3,5)
plt.plot(bin_centers,bin_means[8,:],'r')
pylab.fill_between(bin_centers,bin_low[8,:],bin_high[8,:], color='red', alpha=0.4) 
plt.ylabel('Final mantle redox (deltaFMQ)')
plt.subplot(3,3,6)
plt.semilogy(bin_centers,bin_means[9,:],'k',label='Solid')
pylab.fill_between(bin_centers,bin_low[9,:],bin_high[9,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[10,:],'r',label='Atmo')
pylab.fill_between(bin_centers,bin_low[10,:],bin_high[10,:], color='red', alpha=0.4) 
plt.ylabel('Final free O reservoirs (kg)')
plt.legend()
plt.subplot(3,3,7)
plt.semilogy(bin_centers,bin_means[11,:],'k-',label='CO2')
pylab.fill_between(bin_centers,bin_low[11,:],bin_high[11,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[12,:],'r-',label='CO')
pylab.fill_between(bin_centers,bin_low[12,:],bin_high[12,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[13,:],'c-',label='H2')
pylab.fill_between(bin_centers,bin_low[13,:],bin_high[13,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[1,:],'b-',label='H2O')
pylab.fill_between(bin_centers,bin_low[1,:],bin_high[1,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[14,:],'m-',label='CH4')
pylab.fill_between(bin_centers,bin_low[14,:],bin_high[14,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[15,:],'y-',label='O2')
pylab.fill_between(bin_centers,bin_low[15,:],bin_high[15,:], color='yellow', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[16,:],'g-',label='Atmo H2O')
pylab.fill_between(bin_centers,bin_low[16,:],bin_high[16,:], color='green', alpha=0.4) 
plt.ylabel('Final Pressure (bar)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.legend()
plt.subplot(3,3,9)
plt.semilogy(bin_centers,bin_means[24,:],'r-',label='Surface')
pylab.fill_between(bin_centers,bin_low[24,:],bin_high[24,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[25,:],'k-',label='Mantle')
pylab.fill_between(bin_centers,bin_low[25,:],bin_high[25,:], color='grey', alpha=0.4) 
plt.ylabel('Final temperature (K)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.legend()
plt.subplot(3,3,8)
plt.semilogy(bin_centers,bin_means[17,:],'k-',label='CO2')
pylab.fill_between(bin_centers,bin_low[17,:],bin_high[17,:], color='grey', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[18,:],'r-',label='CO')
pylab.fill_between(bin_centers,bin_low[18,:],bin_high[18,:], color='red', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[19,:],'c-',label='H2')
pylab.fill_between(bin_centers,bin_low[19,:],bin_high[19,:], color='cyan', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[20,:],'b-',label='H2O')
pylab.fill_between(bin_centers,bin_low[20,:],bin_high[20,:], color='blue', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[21,:],'m-',label='CH4')
pylab.fill_between(bin_centers,bin_low[21,:],bin_high[21,:], color='magenta', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[22,:],'y-',label='O2')
pylab.fill_between(bin_centers,bin_low[22,:],bin_high[22,:], color='yellow', alpha=0.4) 
plt.semilogy(bin_centers,bin_means[23,:],'g-',label='Atmo H2O')
pylab.fill_between(bin_centers,bin_low[23,:],bin_high[23,:], color='green', alpha=0.4) 
plt.ylabel('Pressure at magma ocean solidification (bar)')
plt.xlabel('Initial H (log$_{10}$(kg))')
plt.legend()
"""
import pdb
pdb.set_trace
'''

#########################################
### ORIGINAL MONTE CARLO SCATTER PLOT ###
#########################################
'''
plt.figure()
plt.subplot(3,3,1)
plt.loglog((init_H_plot/(1.57*10**20)),(Magma_solid_time/(365*24*60*60)-3e7)/1e6,'x')
plt.ylabel('Duration magma ocean (Myr)')

plt.subplot(3,3,2)
plt.loglog((init_H_plot/(1.57*10**20)),(Final_surface_water/(1.4*10**21)),'x')
plt.ylabel('Final surface water (EO)')
plt.subplot(3,3,3)
plt.loglog((init_H_plot/(1.57*10**20)),(Final_solid_H/(1.57*10**20)),'kx',label='Solid')
plt.loglog((init_H_plot/(1.57*10**20)),(Final_fluid_H/(1.57*10**20)),'bx',label='Fluid')
plt.loglog((init_H_plot/(1.57*10**20)),(Final_fluid_H/(1.57*10**20))+(Final_solid_H/(1.57*10**20)),'r-')
plt.ylabel('Final H (solid, fluid)')
plt.legend()
plt.subplot(3,3,4)
plt.loglog((init_H_plot/(1.57*10**20)),Final_atmo_C,'bx',label='Atmo')
plt.loglog((init_H_plot/(1.57*10**20)),Final_solid_C,'kx',label='Solid')
plt.loglog((init_H_plot/(1.57*10**20)),Final_atmo_C+Final_solid_C,'r-')
plt.ylabel('Final C (solid, fluid) (kg)')
plt.legend()
plt.subplot(3,3,5)
plt.semilogx((init_H_plot/(1.57*10**20)),redox_state_solid_ar)
plt.ylabel('Final mantle redox ($\Delta$FMQ)')
plt.subplot(3,3,6)
plt.loglog((init_H_plot/(1.57*10**20)),(Solid_O/(1.243*10**21)),'kx',label='Solid')
plt.loglog((init_H_plot/(1.57*10**20)),(Atmo_O/(1.243*10**21)),'rx',label='Atmo')
plt.ylabel('Final O (solid, fluid) (EO)')
plt.legend()
plt.subplot(3,3,7)
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO2,'k-',label='CO2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO,'r-',label='CO')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2,'c-',label='H2')
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_water,'b-',label='H2O')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CH4,'m-',label='CH4')
plt.loglog((init_H_plot/(1.57*10**20)),Oxygen_fug,'y-',label='O2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2O,'g--',label='Atmo H2O')
plt.ylabel('Final Pressure (bar)')
plt.xlabel('Initial H (EO)')
plt.legend()
plt.subplot(3,3,8)
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO2_fmt,'k-',label='CO2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO_fmt,'r-',label='CO')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2_fmt,'c-',label='H2')
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_water_fmt,'b-',label='H2O')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CH4_fmt,'m-',label='CH4')
plt.loglog((init_H_plot/(1.57*10**20)),Oxygen_fug_fmt,'y-',label='O2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2O_fmt,'g--',label='Atmo H2O')
plt.ylabel('fmt Pressure (bar)')
plt.xlabel('Initial H (EO)')
plt.legend()
plt.subplot(3,3,9)
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_temp,'rx',label='Surface')
plt.loglog((init_H_plot/(1.57*10**20)),Final_mantle_temp,'kx',label='Mantle')
plt.ylabel('Final temperature (K)')
plt.xlabel('Initial H (EO)')
plt.legend()
#plt.show()
'''

'''
try:
    #Plot_fun(300)
    #Plot_fun(350)
    #Plot_fun(325)
    Plot_fun(330)
    #Plot_fun(340)
except:
    print ('nope')

Plot_fun(0)
Plot_fun(5)
Plot_fun(10)
Plot_fun(15)
Plot_fun(20)
Plot_fun(25)
Plot_fun(55)
'''

###############################################
### MONTE CARLO SCATTER PLOT WITH FILTERING ###
###############################################

#Plot individual atmospheric species final pressure over initial H 

#Define EO of water, so we can plot over initial H inventories wrt Earth Oceans
#One EO = 1.4e21 kg of H2O, mass fraction of H2/H2O = 0.112 -> one EO of hydrogen is 1.57e20
EO = 1.57*10**20 

#Adjust minimum values to reassign near-zero: y<1e-10 => y=1e-10
def adjust_y_vals(y_data):
    return np.maximum(y_data, 1e-10)
'''
fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
# Subplot 1: Magma Ocean Duration
axes[0,0].loglog(init_H_plot/EO, adjust_y_vals((Magma_solid_time/(365*24*60*60)-3e7)/1e6), 'ko', alpha=0.6, label="Magma Ocean Duration")
axes[0,0].set_ylabel("Magma Ocean Time (Myr)")
axes[0,0].legend()
# Subplot 2: Final H2 Pressure
axes[0,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2), 'co', alpha = 0.6, label="Full")
axes[0,1].set_ylabel("Final H2 (bar)")
axes[0,1].legend()
# Subplot 3: Final CO2 Pressure
axes[1,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO2), 'ko', alpha = 0.6, label="Full")
axes[1,0].set_ylabel("Final CO2 (bar)")
axes[1,0].legend()
# Subplot 4: Final O2 Pressure
axes[1,1].loglog(init_H_plot/EO, adjust_y_vals(Oxygen_fug), 'yo', alpha = 0.6, label="Full")
axes[1,1].set_ylabel("Final O2 (bar)")
axes[1,1].legend()
# Subplot 5: Final CO Pressure
axes[2,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO), 'ro', alpha = 0.6, label="Full")
axes[2,0].set_ylabel("Final CO (bar)")
axes[2,0].legend()
# Subplot 6: Final Atmospheric H2O
axes[2,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2O), 'bo', alpha = 0.6, label="Full")
axes[2,1].set_ylabel("Final Atmos H2O (bar)")
axes[2,1].legend()
# Subplot 7: Final CH4 Pressure
axes[3,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CH4), 'mo', alpha = 0.6, label="Full")
axes[3,0].set_xlabel("Initial Hydrogen Inventory")
axes[3,0].set_ylabel("Final CH4 (bar)")
axes[3,0].legend()
# Subplot 8: Final Surface H2O Inventory
axes[3,1].loglog(init_H_plot/EO, adjust_y_vals(Final_surface_water), 'go', alpha = 0.6, label="Full")
axes[3,1].set_xlabel("Initial Hydrogen Inventory")
axes[3,1].set_ylabel("Final Surf H2O (bar)")
axes[3,1].legend()

# Adjust layout and save the figure
plt.tight_layout()
#plt.savefig("final_atmospheric_pressures_vs_H.png", dpi=300)
'''

################################################# DOWN-SELECT
#Down-select from existing data to plot only the cases which produce final results similar to modern Venus
#For this selection, use Atmo_CO<5, Atmo_CO2<200 bar, and oxygen in the atmosphere Oxygen_fug<20 

# Initialize empty lists for filtered data
filtered_H_plot = []
filtered_CO2 = []
filtered_CO = []
filtered_H2O = []
filtered_O2 = []
filtered_CH4 = []
filtered_H2 = []
filtered_Surface_H2O = []
filtered_Magma_solid_time = []

# Initialize empty lists for habitable data
hab_H_plot = []
hab_CO2 = []
hab_CO = []
hab_H2O = []
hab_O2 = []
hab_CH4 = []
hab_H2 = []
hab_Surface_H2O = []
hab_Magma_solid_time = []

# Loop through all data points and apply the filtering conditions

for i in range(len(init_H_plot)):
    if (Oxygen_fug[i] < 30) and (Atmo_H2[i] < 1):
    #if (Oxygen_fug[i] < 30):
    #if (Atmo_H2[i] < 1):
        filtered_H_plot.append(init_H_plot[i])
        filtered_CO2.append(Atmo_CO2[i])
        filtered_CO.append(Atmo_CO[i])
        filtered_H2O.append(Atmo_H2O[i])
        filtered_O2.append(Oxygen_fug[i])
        filtered_CH4.append(Atmo_CH4[i])
        filtered_H2.append(Atmo_H2[i])
        filtered_Surface_H2O.append(Final_surface_water[i])
        filtered_Magma_solid_time.append(Magma_solid_time[i])


# Convert lists to numpy arrays for plotting
filtered_H_plot = np.array(filtered_H_plot)
filtered_CO2 = np.array(filtered_CO2)
filtered_CO = np.array(filtered_CO)
filtered_H2O = np.array(filtered_H2O)
filtered_O2 = np.array(filtered_O2)
filtered_CH4 = np.array(filtered_CH4)
filtered_H2 = np.array(filtered_H2)
filtered_Surface_H2O = np.array(filtered_Surface_H2O)
filtered_Magma_solid_time = np.array(filtered_Magma_solid_time)
'''
# Create figure with 4 rows and 2 columns
fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
# Subplot 1: Magma Ocean Duration (Bottom-right)
axes[0,0].loglog(init_H_plot/EO, adjust_y_vals((Magma_solid_time/(365*24*60*60)-3e7)/1e6), 'ko', alpha= 0.2, label="Full")
axes[0, 0].loglog(filtered_H_plot/EO, adjust_y_vals(filtered_Magma_solid_time/(365*24*60*60)-3e7)/1e6, 'kx', label="Down-selected")
axes[0, 0].set_ylabel("Magma Ocean Time (Myr)")
axes[0, 0].legend()
# Subplot 2: Final H2 Pressure (Row 3, Right)
axes[0,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2), 'co', alpha= 0.5,  label="Full")
axes[0, 1].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_H2), 'kx', label="Down-selected")
axes[0, 1].set_ylabel("Final H2 (bar)")
axes[0, 1].legend()
# Subplot 3: Final CO2 Pressure (Top-left)
axes[1,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO2), 'go', alpha=0.5, label="Full")
axes[1, 0].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_CO2), 'kx', label="Down-selected")
axes[1, 0].set_ylabel("Final CO2 (bar)")
axes[1, 0].legend()
# Subplot 4: Final O2 Pressure (Row 2, Right)
axes[1,1].loglog(init_H_plot/EO, adjust_y_vals(Oxygen_fug), 'yo', alpha=0.5, label="Full")
axes[1, 1].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_O2), 'kx', label="Down-selected")
axes[1, 1].set_ylabel("Final O2 (bar)")
axes[1, 1].legend()
# Subplot 2: Final CO Pressure (Top-right)
axes[2,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO), 'ro', alpha = 0.5, label="Full")
axes[2, 0].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_CO), 'kx', label="Down-selected")
axes[2, 0].set_ylabel("Final CO (bar)")
axes[2, 0].legend()
# Subplot 3: Final Atmospheric H2O (Row 2, Left)
axes[2,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2O), 'bo', alpha = 0.5, label="Full")
axes[2, 1].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_H2O), 'kx', label="Down-selected")
axes[2, 1].set_ylabel("Final Atmos H2O (bar)")
axes[2, 1].legend()
# Subplot 7: Final CH4 Pressure 
axes[3,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CH4), 'mo', alpha = 0.5, label="Full")
axes[3, 0].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_CH4), 'kx', label="Down-selected")
axes[3, 0].set_xlabel("Initial Hydrogen Inventory")
axes[3, 0].set_ylabel("Final CH4 (bar)")
axes[3, 0].legend()
# Subplot 8: Final Surface H2O Inventory 
axes[3,1].loglog(init_H_plot/EO, adjust_y_vals(Final_surface_water), 'go', alpha = 0.5, label="Full")
axes[3, 1].loglog(filtered_H_plot/(1.57*10**20), adjust_y_vals(filtered_Surface_H2O), 'kx', label="Down-selected")
axes[3, 1].set_xlabel("Initial Hydrogen Inventory")
axes[3, 1].set_ylabel("Final Surf H2O (bar)")
axes[3, 1].legend()

# Adjust layout and save the figure
plt.tight_layout()
#plt.savefig("final_atmospheric_pressures_vs_H_filtered.png", dpi=300)
'''

## Attempt to plot single model run
'''try:
    Plot_fun(0)
except:
    print ('Single plot failed')
    
try:
    Plot_fun(1234)
except:
    print ('Single plot failed')

'''

'''
try:
    Plot_fun(656)
except:
    print ('Single plot failed')'''

l = [0]
#[3113, 3210, 3220, 3229, 3263, 3286, 3387, 3454, 3484, 3506, 3564, 3617, 3634, 3655, 3879]
# Sossi 52: [2929, 2989, 3076, 3092, 3155, 3181, 3187, 3233, 3239, 3264, 3292, 3338, 3378, 3421, 3426, 3430, 3513, 3609, 3621, 3635, 3641, 3653, 3664, 3826, 3838, 3849, 3921, 3950, 4081]

# Sossi 51: [306, 1211, 1227, 1322, 1331, 1367, 1371, 1383, 1403, 1475, 1484, 1536]
# Sossi 50: [1175, 1178, 1267, 1295, 1313, 1331, 1347, 1350, 1358, 1368, 1373, 1376, 1390, 1398, 1404, 1405, 1586, 1619]
#Sossi MC run: [11, 29, 30, 61, 67, 76, 84, 88, 89, 98, 100, 107, 165, 172]


# 34 habitable: [3064, 3205, 3209, 3269, 3291, 3329, 3387, 3422, 3453, 3456, 3560, 3679, 3720, 3737, 3767, 3988, 4185]
# 35 habitable: [2964, 2973, 3020, 3041, 3046, 3115, 3141, 3190, 3260, 3261, 3332, 3368, 3620, 3634, 3658, 3674, 3847, 3934, 3979, 4067]
# 36 habitable: [2743, 2887, 2980, 3065, 3177, 3209, 3237, 3260, 3264, 3333, 3362, 3401, 3488, 3581, 3651, 3716, 3740, 3776, 3939, 4167]
# 37 habitable: [3043, 3083, 3157, 3164, 3217, 3244, 3264, 3314, 3339, 3360, 3381, 3437, 3464, 3471, 3484, 3516, 3524, 3546, 3552, 3642, 3714, 3740, 3777, 3846]
# 50 habitable: [110, 150, 179, 182, 200, 205, 212, 215, 218, 225, 230, 237, 245, 251, 270, 280, 287, 403, 416]

# [2453]
#[1881, 1921, 1934, 1937, 1969, 2020, 2032, 2064, 2093, 2106, 2194, 2196, 2281]
 #[677, 705]

Plot_phi_vol_frac(0)

for i in range(0,len(l)):
    g = l[i]
    Plot_fun(g)

plt.show()
