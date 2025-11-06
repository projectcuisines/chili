import numpy as np
from other_functions import *
from joblib import Parallel, delayed

from fe_speciation import set_iron_param
# Choose the scheme once, e.g., from a config/flag:
Fe_speciation = "Kress"   # "Sossi" or "Kress"
set_iron_param(Fe_speciation)

from PACMAN_P import forward_model
from stellar_funs import *
from all_classes_NPM import *
import pylab as plt
import matplotlib 
matplotlib.use("TkAgg")
import os
import shutil
from numba import jit
global y_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
from radiative_functions_6dim_CO_as_N2 import my_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
global solve_Tsurf,Plot_compare
from Simple_Climate_test import solve_Tsurf,Plot_compare
from Input_settings import *
import logging

#logging.basicConfig(filename ='_run_log.txt', level=logging.INFO, format = '%(asctime)s %(message)s')
#log = logging.getLogger()


num_runs = 1 #5000 #1 #720 # Number of forward model runs, only >1 for Monte Carlo calculations
num_cores = 1 #1 #60 # For parallelization, check number of cores with multiprocessing.cpu_count()

os.mkdir('Temporary_outputs') #creates folder for temporarily storing outputs (deleted at successful completion of code)

## Choose planetary system to simulate (note that planet mass and radius must also be changed in Melt_volatile_partitioning_EXP_v4.py)
#which_planet = "Earth" #For Earth calculations
which_planet = "TRAPPIST" # For TRAPPIST-1b, c, d, e, f, or g
#which_planet = "Prox" #For Prox cen b

start_time_Myr = 30 #in Myr #MAKE SURE THIS AGREES WITH Start_time in "if which_plant = "

if which_planet=="Earth":

    Start_time = 3e7 #in yrs #also change in 769-771 #also in 42-44 in MonteCarloPlotting.py
    Max_time = 5e9 #in yrs

    #Time-step parameters
    step0 =100.100
    step1=500000.0 
    step2=50000
    step3=1e3
    tfin1 = 2000e6 

    Earth_inputs = Switch_Inputs(Metal_sink_switch = Metal_sink_switch, do_solid_evo = 1, heating_switch = 0,Start_time=Start_time,Max_time = Max_time)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = step0, step1=step1 , step2=step2, step3=step3, step4=-999, tfin0=Earth_inputs.Start_time+20000.0, tfin1=Earth_inputs.Start_time+tfin1, tfin2=Earth_inputs.Max_time, tfin3=5e9, tfin4 = -999) # Standard Earth parameters, 0 - 4.5 Gyrs

    Total_Fe_mol_fraction_start = Initial_FeO #0.06

    RE = 1 #0.95 for Venus # ALSO CHANGE IN Melt_Volatile_Partitioning
    ME = 1 #0.815 for Venus # ALSO CHANGE IN Melt_Volatile_Partitioning

    ## Planet-star separation (in AU)
    Planet_sep = 1 #0.7 #1.0 #0.7 #1.0 for Earth, 0.7 for Venus

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)

    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k_cond = 4.2 #Thermal conductivity, W/m/K)
    kappa = 1e-6 #Thermal diffusivity of silicates, m2/s)
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01 #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420 - 80 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825 + 0.000 #Solidus coefficient 

    XAl2O3 = 0.022423 
    XCaO = 0.0335
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    XMgO = 0.478144  
    XSiO2 =  0.4034   

    #viable nominal Earth/Venus single runs calculations
    Init_fluid_CO2 = 2.72e20 #4e20 nominal #5*4e20 for high C # Initial mass of C (kg), NOT CO2
    Init_fluid_H2O = 4.7e20 #0.4e21+1e22 #0.4e21+5e22 #4e20 nominal # Initial mass of H (kg), NOT H2O
    Init_fluid_O = 6e21 #6e21 nominal #6e21+9*0.5e22  # Initial mass of free O (kg)

    init_water = 10**np.linspace(20,23,num_runs) # Initial mass of H (kg), NOT H2O
    init_water = init_water*0 + Init_fluid_H2O #4e20 #8e21  # for single model runs
    init_O = init_water*0 + Init_fluid_O
    init_CO2 = init_water*0 + Init_fluid_CO2 

    Albedo_C_range = init_water*0 + 0.1 
    Albedo_H_range = init_water*0 + 0.1 #spare

    Stellar_Mass = 1.0

    tsat_XUV = 0.023 # Tu et al
    tsat_sun_ar = init_water*0 + tsat_XUV

    beta0 = -1.22 # Tu et al.
    beta_sun_ar = init_water*0 + beta0

    fsat = 10**-3.13 #Tu et al. 
    fsat_ar = init_water*0 + fsat

    mix_epsilon = 0.5
    mix_epsilon_ar = init_water*0 + mix_epsilon

    epsilon = 0.2
    Epsilon_ar = init_water*0 + epsilon

    mult = 1.0
    mult_ar= init_water*0 + mult

    Thermo_temp = 1000.0
    Thermosphere_temp = init_water*0 + Thermo_temp

    Te_input_escape_mod = 0.0
    Tstrat_array = init_water*0 + Te_input_escape_mod

    constant_loss_NT = 0.0
    NT_loss_array = init_water*0 + constant_loss_NT

    visc_offset = 10.0
    offset_range = init_water*0 + visc_offset

    heatscale = 1.0
    heatscale_ar = init_water*0 + heatscale

    spare = 0.0

    '''
    ###########
    #MC optional
    
    init_water = 10**np.linspace(18.5,21.5,num_runs) # 10**np.linspace(18.5,21.5,num_runs) # Initial mass of H (kg), NOT H2O
    init_CO2 = 10**np.random.uniform(19.75,21.3,num_runs) # 10**np.random.uniform(19.75,21.3,num_runs) # Initial mass of C (kg), NOT CO2
    init_O = 10**np.random.uniform(21,22,num_runs) # Initial mass of free O (kg)
    

    Albedo_C_range = np.random.uniform(0.1,0.6,num_runs) #Kopparapu paper, Shields paper
    Albedo_H_range = np.random.uniform(0.1,0.6,num_runs) #np.random.uniform(0.1,0.6,num_runs)
    #for k in range(0,len(Albedo_C_range)):
    #    if Albedo_C_range[k] < Albedo_H_range[k]:
    #        Albedo_H_range[k] = Albedo_C_range[k]-1e-5    
       
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)

    Tstrat_array = np.random.uniform(-30.0,30.0,num_runs) ## modification of skin temperature
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs)
    heatscale_ar = 10**np.random.uniform(-0.48,1.477,num_runs)
    Thermosphere_temp = 10**np.random.uniform(2.3,3.699,num_runs) #Johnstone papers, 200 K - 5000 K

    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    mult_ar = 10**np.random.uniform(-6.0,1.0,num_runs)
   

    #Stellar evolution parameters
    #1.8, 6.2, 45.6 Tu et al. 2015 10,50,90% confidence
    Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),num_runs)
    tsat_sun_ar = (2.9*Omega_sun_ar**1.14)/1000 #XUV saturation time
    fsat = 10**-3.13 #Tu et al.
    fsat_ar = init_water*0 + fsat
    beta_sun_ar = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)
    beta_sun_ar = 0.86*beta_sun_ar #conversion after equation 1, Tu et al.
    
    
    #####
    #'''

if which_planet=="TRAPPIST":

    Start_time = 30e6 #in yrs
    Max_time = 7.5e9 #in yrs
    
    #Time step parameters
    step0 =100.100
    step1=500000.0 
    step2=50000
    step3=1e3
    tfin1 = 2000e6 #fine for 1-e

    #adjustments for 1-b specifically (slightly improve model performance?)
    #Max_time = 7.1e9
    #tfin1 = 7000e6 #needed for b?
    #step1 = 5000000.0 # try for b

    Earth_inputs = Switch_Inputs(Metal_sink_switch = Metal_sink_switch, do_solid_evo = 1, heating_switch = 0,Start_time=Start_time,Max_time = Max_time)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = step0, step1=step1 , step2=step2, step3=step3, step4=-999, tfin0=Earth_inputs.Start_time+20000.0, tfin1=Earth_inputs.Start_time+tfin1, tfin2=Earth_inputs.Max_time, tfin3=7e9, tfin4 = -999) # Standard Earth parameters, 0 - 4.5 Gyrs

    Total_Fe_mol_fraction_start = Initial_FeO 

    ## For setting all planets to Earth size to compare effect of different insolations:
    RE = 1.0
    ME = 1.0
    core_radius = 3.5e6 #3.4e6

    ## Planet-star separation (in AU)
    #Planet_sep = 0.029 #1e
    Planet_sep = 0.0115 #1b

    #optional true masses and radii for TRAPPIST-1 planets
    '''
    TRAPPIST_R_array = np.array([1.116,1.097,0.788,.920,1.045,1.129,0.775]) ## Agol et al.
    TRAPPIST_M_array = np.array([1.374,1.308,0.388,.692,1.039,1.321,0.326]) ## Agol et al.
    TRAPPIST_sep_array = np.array([0.01154, 0.0158,0.02227,0.02925,0.03849,0.04683,.06189 ]) # Agol et al.
    TRAPPIST_Rc_array = TRAPPIST_R_array * 3.4e6
    RE = TRAPPIST_R_array[4]
    ME = TRAPPIST_M_array[4]
    core_radius = TRAPPIST_Rc_array[4]
    '''
    #end optional true values

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)

    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k_cond = 4.2 #Thermal conductivity, W/m/K)
    kappa = 1e-6 #Thermal diffusivity of silicates, m2/s)
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01 #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420 - 80 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825 + 0.000 #Solidus coefficient 

    XAl2O3 = 0.022423 
    XCaO = 0.0335
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    XMgO = 0.478144  
    XSiO2 =  0.4034   


    #Initial volatiles
    Init_fluid_CO2 = 5.45e20 #4e20 #+ 5*4e20 #for high C  # Initial mass of C (kg), NOT CO2
    Init_fluid_H2O = 2.22e20#0.4e21 #+5e22 #+1e22  # Initial mass of H (kg), NOT H2O
    Init_fluid_O = 6e21 #+9*0.5e22#  # Initial mass of free O (kg)

    init_water = 10**np.linspace(20,23,num_runs) ## high H # Initial mass of H (kg), NOT H2O
    init_water = init_water*0 + Init_fluid_H2O #4e20  +8e21 #+5e22# + 8e22
    #init_water = 10**np.linspace(22,23,num_runs) ## high H
    init_O = init_water*0 + Init_fluid_O
    init_CO2 = init_water*0 + Init_fluid_CO2 
    #for carbon test
    #init_water = 10**np.array([21.3,22.3])
    #init_CO2 = init_water*0 + 1e21 
    
    Albedo_C_range = init_water*0 + 0.1 #spare
    Albedo_H_range = init_water*0 + 0.1 

    Stellar_Mass = 0.09

    tsat_XUV = 3.14 #Birky et al.
    tsat_sun_ar = init_water*0 + tsat_XUV

    beta0 = -1.17 #Birky et al.
    beta_sun_ar = init_water*0 + beta0

    fsat = 10**-3.03 #Birky et al.
    fsat_ar = init_water*0 + fsat

    mix_epsilon = 0.5
    mix_epsilon_ar = init_water*0 + mix_epsilon

    epsilon = 0.2
    Epsilon_ar = init_water*0 + epsilon

    mult = 1.0
    mult_ar= init_water*0 + mult

    Thermo_temp = 1000.0
    Thermosphere_temp = init_water*0 + Thermo_temp

    Te_input_escape_mod = 0.0
    Tstrat_array = init_water*0 + Te_input_escape_mod

    constant_loss_NT = 0.0
    NT_loss_array = init_water*0 + constant_loss_NT

    visc_offset = 10.0
    offset_range = init_water*0 + visc_offset

    heatscale = 1.0
    heatscale_ar = init_water*0 + heatscale

    spare = 0.0

    #option for full MCMC calculation (comment out for single model run)
    '''
    init_water = 10**np.linspace(20,23,num_runs) # Initial mass of H (kg), NOT H2O
    init_O = 10**np.random.uniform(21,22,num_runs) # Initial mass of free O (kg)
    init_CO2 = 10**np.random.uniform(20,21.5,num_runs) # Initial mass of C (kg), NOT CO2

    Albedo_C_range = np.random.uniform(0.01,0.5,num_runs) #Kopparapu paper, Shields paper
    Albedo_H_range = np.random.uniform(0.0001,0.2,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5    
       
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)

    # actually using Birky et al. posteriors
    ace = np.load('trappist_posterior_samples_updated.npy')
    indices_narrow = np.where((ace[:,3]>7) & (ace[:,3]<9)) #restrict ages
    ace_narrow = ace[indices_narrow]
    stellar_sample_index = np.random.randint(0,len(ace_narrow),num_runs)
    fsat_ar = 10**ace_narrow[stellar_sample_index,1]
    tsat_sun_ar = ace_narrow[stellar_sample_index,2]
    beta_sun_ar = ace_narrow[stellar_sample_index,4]

    Tstrat_array = np.random.uniform(-30.0,30.0,num_runs) ## modification of skin temperature
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs) 
    heatscale_ar = 10**np.random.uniform(-0.48,1.477,num_runs)
    Thermosphere_temp = 10**np.random.uniform(2.3,3.699,num_runs) #Johnstone papers, 200 K - 5000 K

    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    mult_ar = 10**np.random.uniform(-6.0,1.0,num_runs)

    #Albedo_H_range = np.random.uniform(0.5,0.7,num_runs) #high albedo test
    '''

if which_planet=="LP8909":
    k_LP = 1 #0 for LP 890-9b, 1 for LP 890-9c

    LP9809b_R = 1.320
    LP9809b_M = 2.3 #+1.7 - 0.7
    LP9809b_sep = 0.01875
    #LP9809b_Rc = LP9809b_R * 3.4e6 #preserves core volume fraction
    #LP9809b_step1 = 10000

    LP9809c_R = 1.367
    LP9809c_M = 2.5 #+1.8 - 0.8
    LP9809c_sep = 0.03984
    #LP9809c_Rc = LP9809b_R * 3.4e6 #preserves core volume fraction
    #LP9809c_step1 = 10000

    LP9809_R_array = np.array([LP9809b_R,LP9809c_R]) 
    LP9809_M_array = np.array([LP9809b_M,LP9809c_M])

    LP9809_sep_array = np.array([LP9809b_sep, LP9809c_sep]) 
    LP9809_Rc_array = LP9809_R_array * 3.4e6

    LP9809_Rc = LP9809_Rc_array[k_LP]

    Start_time = 10e6 #yrs
    Max_time = 7.2e9 #yrs
    
    #time step parameters
    step0 =100.100
    step1=500000.0 #5000 originally
    step2=50000
    step3=1e3
    tfin1 = 2000e6 #fine for e

    #adjustments for b specifically
    Max_time = 7.1e9
    tfin1 = 7000e6 #needed for b?
    step1 = 5000000.0 # try for b

    Earth_inputs = Switch_Inputs(Metal_sink_switch = Metal_sink_switch, do_solid_evo = 1, heating_switch = 0,Start_time=Start_time,Max_time = Max_time)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = step0, step1=step1 , step2=step2, step3=step3, step4=-999, tfin0=Earth_inputs.Start_time+20000.0, tfin1=Earth_inputs.Start_time+tfin1, tfin2=Earth_inputs.Max_time, tfin3=7e9, tfin4 = -999) # Standard Earth parameters, 0 - 4.5 Gyrs

    Total_Fe_mol_fraction_start = Initial_FeO 

    RE = LP9809_R_array[k_LP]
    ME = LP9809_M_array[k_LP]

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)

    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k_cond = 4.2 #Thermal conductivity, W/m/K)
    kappa = 1e-6 #Thermal diffusivity of silicates, m2/s)
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01 #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420 - 80 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825 + 0.000 #Solidus coefficient 

    XAl2O3 = 0.022423 
    XCaO = 0.0335
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    XMgO = 0.478144  
    XSiO2 =  0.4034   

    #Initial volatiles
    Init_fluid_CO2 = 4e20 #+ 5*4e20 #for high C  # Initial mass of C (kg), NOT CO2
    Init_fluid_H2O = 0.4e21 #+5e22 #+1e22  # Initial mass of H (kg), NOT H2O
    Init_fluid_O = 6e21 #+9*0.5e22#  # Initial mass of free O (kg)

    init_water = 10**np.linspace(20,23,num_runs) #Initial mass of H (kg), NOT H2O
    #init_water = init_water*0 + 4e20  +8e21 + 8e22
    init_O = init_water*0 + Init_fluid_O
    init_CO2 = init_water*0 + Init_fluid_CO2 
    #for carbon test
    #init_water = 10**np.array([21.3,22.3])
    #init_CO2 = init_water*0 + 1e21 

    # for MCMC
    
    init_O = 10**np.random.uniform(21,22,num_runs)
    init_CO2 = 10**np.random.uniform(20,21.5,num_runs)

    Albedo_C_range = init_water*0 + 0.2 
    Albedo_H_range = init_water*0 + 0.0 #spare

    Stellar_Mass = 0.118 #https://en.wikipedia.org/wiki/LP_890-9 

    tsat_XUV = 1.0 
    tsat_sun_ar = init_water*0 + tsat_XUV

    beta0 = -1.23 
    beta_sun_ar = init_water*0 + beta0

    fsat = 10**-3.00 
    fsat_ar = init_water*0 + fsat

    Planet_sep = LP9809_sep_array[k_LP] 

    mix_epsilon = 0.5
    mix_epsilon_ar = init_water*0 + mix_epsilon

    epsilon = 0.2
    Epsilon_ar = init_water*0 + epsilon

    mult = 1.0
    mult_ar= init_water*0 + mult

    Thermo_temp = 1000.0
    Thermosphere_temp = init_water*0 + Thermo_temp

    Te_input_escape_mod = 0.0
    Tstrat_array = init_water*0 + Te_input_escape_mod

    constant_loss_NT = 0.0
    NT_loss_array = init_water*0 + constant_loss_NT

    visc_offset = 10.0
    offset_range = init_water*0 + visc_offset

    heatscale = 1.0
    heatscale_ar = init_water*0 + heatscale

    spare = 0.0

    #full mcmc
    
    Albedo_C_range = np.random.uniform(0.01,0.5,num_runs) #Kopparapu paper, Shields paper
    Albedo_H_range = np.random.uniform(0.0001,0.2,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5    
       
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)

    #To maximize possible atmospheric loss, assume TRAPPIST-1 like XUV evolution (with bolometric luminosity evolution of larger star)
    ace = np.load('trappist_posterior_samples_updated.npy')
    indices_narrow = np.where((ace[:,3]>7) & (ace[:,3]<9)) #restrict ages
    ace_narrow = ace[indices_narrow]
    stellar_sample_index = np.random.randint(0,len(ace_narrow),num_runs)
    fsat_ar = 10**ace_narrow[stellar_sample_index,1]

    tsat_sun_ar = ace_narrow[stellar_sample_index,2]
    beta_sun_ar = ace_narrow[stellar_sample_index,4]

    Tstrat_array = np.random.uniform(-30.0,30.0,num_runs) ## modification of skin temperature
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs)
    heatscale_ar = 10**np.random.uniform(-0.48,1.477,num_runs)

    Thermosphere_temp = 10**np.random.uniform(2.3,3.699,num_runs) #Johnstone papers, 200 K - 5000 K

    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    mult_ar = 10**np.random.uniform(-6.0,1.0,num_runs) 

    #Albedo_H_range = np.random.uniform(0.5,0.7,num_runs) #high albedo test

    
if which_planet=="Prox":

    Proxb_R = 1.20
    Proxb_M = 1.07 
    Proxb_sep = 0.04856
    Planet_sep = Proxb_sep

    Proxb_Rc = Proxb_R * 3.4e6

    Start_time = 10e6
    Max_time = 7.2e9
    
    step0 =100.100
    step1=500000.0 #5000 originally
    step2=50000
    step3=1e3
    tfin1 = 2000e6 #fine for e

    #adjustments for b specifically
    Max_time = 7.1e9
    tfin1 = 7000e6 #needed for b?
    step1 = 5000000.0 # try for b

    Earth_inputs = Switch_Inputs(Metal_sink_switch = Metal_sink_switch, do_solid_evo = 1, heating_switch = 0,Start_time=Start_time,Max_time = Max_time)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = step0, step1=step1 , step2=step2, step3=step3, step4=-999, tfin0=Earth_inputs.Start_time+20000.0, tfin1=Earth_inputs.Start_time+tfin1, tfin2=Earth_inputs.Max_time, tfin3=7e9, tfin4 = -999) # Standard Earth parameters, 0 - 4.5 Gyrs


    Total_Fe_mol_fraction_start = Initial_FeO 

    RE = Proxb_R
    ME = Proxb_M

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)

    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k_cond = 4.2 #Thermal conductivity, W/m/K)
    kappa = 1e-6 #Thermal diffusivity of silicates, m2/s)
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01 #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420 - 80 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825 + 0.000 #Solidus coefficient 

    XAl2O3 = 0.022423 
    XCaO = 0.0335
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    XMgO = 0.478144  
    XSiO2 =  0.4034   

    #Initial volatiles
    Init_fluid_CO2 = 4e20 #+ 5*4e20 #for high C  # Initial mass of C (kg), NOT CO2
    Init_fluid_H2O = 0.4e21 #+5e22 #+1e22  # Initial mass of H (kg), NOT H2O
    Init_fluid_O = 6e21 #+9*0.5e22#  # Initial mass of free O (kg)


    init_water = 10**np.random.uniform(20,22.6,num_runs)# Initial mass of H (kg), NOT H2O
    init_water = 10**np.linspace(20,22.2,num_runs)# Initial mass of H (kg), NOT H2O
    init_water = 10**np.linspace(20,23,num_runs) # Initial mass of H (kg), NOT H2O
    #init_water = init_water*0 + 4e20  +8e21 + 8e22
    init_O = init_water*0 + Init_fluid_O
    init_CO2 = init_water*0 + Init_fluid_CO2 
    #for carbon test
    #init_water = 10**np.array([21.3,22.3])
    #init_CO2 = init_water*0 + 1e21 

    # for MCMC
    
    init_O = 10**np.random.uniform(21,22,num_runs)
    init_CO2 = 10**np.random.uniform(20,21.5,num_runs)

    Albedo_C_range = init_water*0 + 0.2 
    Albedo_H_range = init_water*0 + 0.0 #spare

    Stellar_Mass = 0.122 

    tsat_XUV = 1.0 
    tsat_sun_ar = init_water*0 + tsat_XUV

    beta0 = -1.23 
    beta_sun_ar = init_water*0 + beta0

    fsat = 10**-3.00 
    fsat_ar = init_water*0 + fsat


    mix_epsilon = 0.5
    mix_epsilon_ar = init_water*0 + mix_epsilon

    epsilon = 0.2
    Epsilon_ar = init_water*0 + epsilon

    mult = 1.0
    mult_ar= init_water*0 + mult

    Thermo_temp = 1000.0
    Thermosphere_temp = init_water*0 + Thermo_temp

    Te_input_escape_mod = 0.0
    Tstrat_array = init_water*0 + Te_input_escape_mod

    constant_loss_NT = 0.0
    NT_loss_array = init_water*0 + constant_loss_NT

    visc_offset = 10.0
    offset_range = init_water*0 + visc_offset

    heatscale = 1.0
    heatscale_ar = init_water*0 + heatscale

    spare = 0.0

    #full mcmc
    
    Albedo_C_range = np.random.uniform(0.01,0.5,num_runs) #Kopparapu papre, Shields paper
    Albedo_H_range = np.random.uniform(0.0001,0.2,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5    
       
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)

    #To maximize possible atmospheric loss, assume TRAPPIST-1 like XUV evolution (with bolometric luminosity evolution of larger star)
    ace = np.load('trappist_posterior_samples_updated.npy')
    indices_narrow = np.where((ace[:,3]>7) & (ace[:,3]<9)) #restrict ages
    ace_narrow = ace[indices_narrow]
    stellar_sample_index = np.random.randint(0,len(ace_narrow),num_runs)
    fsat_ar = 10**ace_narrow[stellar_sample_index,1]
    tsat_sun_ar = ace_narrow[stellar_sample_index,2]
    beta_sun_ar = ace_narrow[stellar_sample_index,4]

    Tstrat_array = np.random.uniform(-30.0,30.0,num_runs) ## modification of skin temperature
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs)
    heatscale_ar = 10**np.random.uniform(-0.48,1.477,num_runs)
    Thermosphere_temp = 10**np.random.uniform(2.3,3.699,num_runs) #Johnstone papers, 200 K - 5000 K

    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    mult_ar = 10**np.random.uniform(-6.0,1.0,num_runs)

    #Albedo_H_range = np.random.uniform(0.5,0.7,num_runs) #high albedo test


inputs = range(0,len(init_water))
output = []


for zzz in inputs:
    ii = zzz
    
    if which_planet=="TRAPPIST":
        Earth_Planet_inputs = Planet_inputs(RE = RE, ME = ME, rc=core_radius, pm=4000.0, Total_Fe_mol_fraction = Total_Fe_mol_fraction_start, Planet_sep=Planet_sep, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=Stellar_Mass, fsat=fsat_ar[ii], beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=NT_loss_array[ii], esc_b = spare, esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii], esc_e = spare, esc_f = spare, interiora =offset_range[ii],interiore = heatscale_ar[ii], Tstrat = Tstrat_array[ii], ThermoTemp = Thermosphere_temp[ii]) 
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]

    if which_planet=="LP8909":
        Earth_Planet_inputs = Planet_inputs(RE = RE, ME = ME, rc=LP9809_Rc, pm=4000.0, Total_Fe_mol_fraction = Total_Fe_mol_fraction_start, Planet_sep=Planet_sep, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=Stellar_Mass, fsat=fsat_ar[ii], beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=NT_loss_array[ii], esc_b = spare, esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii], esc_e = spare, esc_f = spare, interiora =offset_range[ii],interiore = heatscale_ar[ii], Tstrat = Tstrat_array[ii], ThermoTemp = Thermosphere_temp[ii]) 
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]

    if which_planet=="Prox":
        Earth_Planet_inputs = Planet_inputs(RE = RE, ME = ME, rc=Proxb_Rc, pm=4000.0, Total_Fe_mol_fraction = Total_Fe_mol_fraction_start, Planet_sep=Planet_sep, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=Stellar_Mass, fsat=fsat_ar[ii], beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=NT_loss_array[ii], esc_b = spare, esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii], esc_e = spare, esc_f = spare, interiora =offset_range[ii],interiore = heatscale_ar[ii], Tstrat = Tstrat_array[ii], ThermoTemp = Thermosphere_temp[ii]) 
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]

    if which_planet=="Earth":
        Earth_Planet_inputs = Planet_inputs(RE = RE, ME = ME, rc=3.5e6, pm=4000.0, Total_Fe_mol_fraction = Total_Fe_mol_fraction_start, Planet_sep=Planet_sep, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=Stellar_Mass, fsat=fsat_ar[ii], beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=NT_loss_array[ii], esc_b = spare, esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii], esc_e = spare, esc_f = spare, interiora =offset_range[ii],interiore = heatscale_ar[ii], Tstrat = Tstrat_array[ii], ThermoTemp = Thermosphere_temp[ii]) 
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]


    sve_name = 'Temporary_outputs/inputs4L%d' %ii
    np.save(sve_name,inputs_for_later)



def processInput(i):
    load_name = 'Temporary_outputs/inputs4L%d.npy' %i
    try:
        if (which_planet=="TRAPPIST")or(which_planet=="Earth")or(which_planet=="LP8909")or(which_planet=="Prox"): 
            print ('starting ',i)
            [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
            print ('seems to have worked first time',i)
    except:
        try: # try again with slightly different numerical options
            if (which_planet=="TRAPPIST")or(which_planet=="Earth")or(which_planet=="LP8909")or(which_planet=="Prox"): 
                #import pdb
                #pdb.set_trace()
                print ('trying ',i,' again')
                [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                Earth_Numerics.step1 = 50000 #
                Earth_Numerics.step1 = 500000 # try for b
                Earth_Numerics.tfin0=Earth_inputs.Start_time+500000.0 ## Again for b
                outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
                print ('seems to have worked second time',i)
            #import pdb
            #pdb.set_trace()
        except:
            try: # try again with slightly different numerical options
                if (which_planet=="TRAPPIST")or(which_planet=="Earth")or(which_planet=="LP8909")or(which_planet=="Prox"): #really only for b
                    print ('trying ',i,' third time')
                    #import pdb
                    #pdb.set_trace()
                    [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                    Earth_Numerics.step1 = 1000000
                    Earth_Numerics.tfin0=Earth_inputs.Start_time+100000.0 ## Again for b
                    Earth_Numerics.step0 = 300.0 #high init H, for b?
                    outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
                    print ('seems to have worked third time',i)
            except:
                try:
                    if (which_planet=="TRAPPIST")or(which_planet=="Earth")or(which_planet=="LP8909")or(which_planet=="Prox"): #really only for b
                        print ('trying ',i,' 4th time')
                        #import pdb
                        #pdb.set_trace()
                        [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                        Earth_Numerics.step1 = 500000
                        Earth_Numerics.tfin0=Earth_inputs.Start_time+4500000.0 ## Again for b
                        Earth_Numerics.step0 = 1000.0 #high init H, for b?
                        outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar)
                        print ('seems to have worked fourth time',i)
                except:
                    print ('didnt work all times ',i)
                    outs = []
    print ('done with ',i)
    return outs


Everything = Parallel(n_jobs=num_cores,backend='multiprocessing')(delayed(processInput)(i) for i in inputs) #Run paralllized code
input_mega=[] # Collect input parameters for saving
for kj in range(0,len(inputs)):
    print ('saving Temporary_outputs',kj)
    load_name = 'Temporary_outputs/inputs4L%d.npy' %kj
    input_mega.append(np.load(load_name,allow_pickle=True))


#If using the saving method above, comment this out
np.save('Model_outputs',Everything) 
np.save('Model_inputs',input_mega) 

shutil.rmtree('Temporary_outputs')

ET_outputs = np.load('Model_outputs.npy',allow_pickle=True) 
ET_inputs = np.load('Model_inputs.npy',allow_pickle=True) 

### Plotting function for individual model runs
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
    
    time = (time/(365*24*60*60)-(start_time_Myr*1e6))/1e6
    time = time*1e6

    solid_time = total_time[fmt:]
    solid_time = (solid_time/(365*24*60*60)-(start_time_Myr*1e6))/1e6
    solid_time = solid_time*1e6  

    solid_y = total_y[:,fmt:]

    total_time_plot = (np.copy(total_time)/(365*24*60*60)-(start_time_Myr*1e6))/1e6
    total_time_plot = total_time_plot*1e6

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

    fO2_bar = y_out[22,:]/1e5
    TotalP = y_out[25,:]+y_out[26,:]+y_out[27,:]+y_out[28,:]+y_out[29,:]+fO2_bar

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

    #import pdb
    #pdb.set_trace()

    '''
    plt.figure()
    plt.plot(total_time_plot,redox_state,label='log(fO2) wrt FMQ')
    plt.plot(total_time_plot,redox_state_solid,label='solid log(fO2) wrt FMQ')
    plt.ylabel('Oxygen fugacity (delta FMQ)')
    plt.xlabel('Time (Myrs)')
    '''
    '''
    plt.figure()
    plt.subplot(3,3,1)
    #plt.title(ET_inputs[i][2].Init_fluid_H2O)
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
    
    #import pdb
    #pdb.set_trace()
    plt.figure()
    plt.subplot(3,3,1)
    #plt.title(ET_inputs[i][2].Init_fluid_H2O)
    plt.semilogx(time,y_out[7,:],label='Mantle')
    plt.semilogx(time,y_out[8,:],label='Surface',linestyle='--')
    #plt.xlabel('Time (yrs)')
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[8,:]),1.1*np.max(y_out[7,:])],'k--')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])
    plt.subplot(3,3,2)
    plt.loglog(time,y_out[9,:],label='OLR')
    plt.loglog(time,y_out[10,:],label='ASR')
    plt.loglog(time,y_out[11,:],label='Interior')
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[11,:]),10.1*np.max(y_out[9,:])],'k--')
    #plt.xlabel('Time (yrs)')
    plt.ylabel('Flux (W/m$^2$)')
    plt.legend()
    plt.xlim([1.0,np.max(solid_time)])
    plt.subplot(3,3,3)
    plt.semilogx(time,y_out[2,:]/1000.0)
    plt.ylabel('Solidification radius (km)')
    #plt.xlabel('Time (yrs)')
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[2,:]/1000.0),1.1*np.max(y_out[2,:]/1000.0)],'k--')
    plt.subplot(3,3,4)
    plt.semilogx(total_time_plot,redox_state,label='Magma ocean')
    plt.plot(total_time_plot,redox_state_solid,label='Solid mantle')
    plt.ylabel('Oxygen fugacity ($\Delta$FMQ)')
    plt.xlabel('Time (yrs)')
    plt.plot([np.min(solid_time),np.min(solid_time)],[np.min(redox_state_solid[~np.isnan(redox_state_solid)])-3,1+np.max(redox_state_solid[~np.isnan(redox_state_solid)])],'k--')
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([np.min(redox_state_solid[~np.isnan(redox_state_solid)])-3,1+np.max(redox_state_solid[~np.isnan(redox_state_solid)])])
    plt.legend()
    plt.subplot(3,3,5)
    #plt.loglog(time,y_out[24,:]/1e5,label='H$_2$O (atmo)')
    plt.loglog(time,y_out[25,:],label='H$_2$O (total)')
    plt.loglog(time,y_out[26,:],label='H$_2$')
    plt.loglog(time,y_out[27,:],label='CO$_2$')
    plt.loglog(time,y_out[28,:],label='CO')
    plt.loglog(time,y_out[29,:],label='CH$_4$')
    plt.loglog(time,y_out[22,:]/1e5,label='O$_2$')
    plt.loglog(time,y_out[25,:]-y_out[24,:]/1e5,label='H$_2$O (liquid)',linestyle='--',color='y')
    plt.xlabel('Time (yrs)')
    plt.ylabel('Pressure (bar)')
    plt.xlim([1.0,np.max(solid_time)])
    plt.ylim([1e-6,5.1*np.max(y_out[25:29,:])])
    plt.plot([np.min(solid_time),np.min(solid_time)],[0.9*np.min(y_out[25:29,:]),20.1*np.max(y_out[25:29,:])],'k--')
    plt.legend()
    
    plt.subplot(3,3,6)
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
    
    plt.subplot(3,3,6)
    total_H = y_out[21,:]+y_out[23,:] + y_out[0,:]/0.001 
    H_solid = np.array((y_out[0,:]/0.001))
    plt.loglog(time,total_H,label='Total H',color='k',linestyle = '-')
    #plt.loglog(time,y_out[0,:]/0.001+ y_out[1,:]/0.001,label='Sum H',color='r',linestyle = '--')
    plt.loglog(solid_time,solid_time*0 + total_H[-1],color='k',linestyle = '-')
    plt.loglog(time,y_out[21,:],label='H Atmo',color='c',linestyle = '-')
    plt.loglog(solid_time,solid_y[21,:],color='c',linestyle = '-')
    plt.loglog(time,y_out[23,:],label='H in Melt',color='m',linestyle = '-') #    plt.semilogy(time,y_out[31,:],label='n_H2O_H_diss',linestyle=':')
    plt.loglog(time,y_out[31,:],label='H in H$_2$O in Melt',color='b',linestyle = ':') 
    plt.loglog(time,H_solid,label='H Solid',color='y',linestyle='-')
    plt.loglog(solid_time,Mantle_hydrogen,color='y',linestyle = '-') #internal reservoirs after solidified
    plt.xlim([1.0,np.max(solid_time)])
    plt.plot([np.min(solid_time),np.min(solid_time)],[1e10,10*np.max(total_H)],'k--')
    plt.ylim([1e18,5*np.max(total_H)])
    plt.xlabel('Time (yrs)')
    plt.ylabel('Reservoir size (mol H)')
    plt.legend()
    '''
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
    '''
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
    '''


### Generate output compilations for Monte Carlo calculations:

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


for j in range(0,np.shape(ET_outputs)[0]):
    print(j)
    init_H_plot.append(ET_inputs[j][2].Init_fluid_H2O)
    try:
        Atmo_CO2.append(ET_outputs[j].total_y[27,-1])
        Atmo_CO.append(ET_outputs[j].total_y[28,-1])
        Atmo_H2.append(ET_outputs[j].total_y[26,-1])
        Atmo_CH4.append(ET_outputs[j].total_y[29,-1])
        Atmo_H2O.append(ET_outputs[j].total_y[24,-1])
        Oxygen_fug.append(ET_outputs[j].total_y[22,-1])
        Final_surface_water.append(ET_outputs[j].total_y[25,-1])

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
    except:
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
init_H_plot_EO = (init_H_plot/(1.57*10**20))

plt.figure()
plt.subplot(3,3,1)
plt.loglog((init_H_plot/(1.57*10**20)),(Magma_solid_time/(365*24*60*60)-(start_time_Myr*1e6))/1e6,'x')
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
plt.semilogx((init_H_plot/(1.57*10**20)),redox_state_solid_ar,'x')
plt.ylabel('Final mantle redox ($\Delta$FMQ)')
plt.subplot(3,3,6)
plt.loglog((init_H_plot/(1.57*10**20)),(Solid_O/(1.243*10**21)),'kx',label='Solid')
plt.loglog((init_H_plot/(1.57*10**20)),(Atmo_O/(1.243*10**21)),'rx',label='Atmo')
plt.ylabel('Final O (solid, fluid) (EO)')
plt.legend()
plt.subplot(3,3,7)
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO2,'kx',label='CO2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO,'rx',label='CO')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2,'cx',label='H2')
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_water,'bx',label='H2O')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CH4,'mx',label='CH4')
plt.loglog((init_H_plot/(1.57*10**20)),Oxygen_fug,'yx',label='O2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2O,'go', markersize=3, label='Atmo H2O')
plt.ylabel('Final Pressure (bar)')
plt.xlabel('Initial H (EO)')
plt.legend()
plt.subplot(3,3,8)
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO2_fmt,'kd',label='CO2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CO_fmt,'rd',label='CO')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2_fmt,'cd',label='H2')
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_water_fmt,'bd',label='H2O')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_CH4_fmt,'md',label='CH4')
plt.loglog((init_H_plot/(1.57*10**20)),Oxygen_fug_fmt,'yd',label='O2')
plt.loglog((init_H_plot/(1.57*10**20)),Atmo_H2O_fmt,'go', markersize=3, label='Atmo H2O')
plt.ylabel('fmt Pressure (bar)')
plt.xlabel('Initial H (EO)')
plt.legend()
plt.subplot(3,3,9)
plt.loglog((init_H_plot/(1.57*10**20)),Final_surface_temp,'rx',label='Surface')
plt.loglog((init_H_plot/(1.57*10**20)),Final_mantle_temp,'kx',label='Mantle')
plt.ylabel('Final temperature (K)')
plt.xlabel('Initial H (EO)')
plt.legend()

##Plot individual atmospheric species final pressure over initial H 
#

#Define EO of water, so we can plot over initial H inventories wrt Earth Oceans
#One EO = 1.4e21 kg of H2O, mass fraction of H2/H2O = 0.112 -> one EO of hydrogen is 1.57e20
EO = 1.57*10**20 
#Adjust minimum values to reassign near-zero: y<1e-10 => y=1e-10
def adjust_y_vals(y_data):
    return np.maximum(y_data, 1e-10)

fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
# Subplot 1: Magma Ocean Duration
axes[0,0].loglog(init_H_plot/EO, adjust_y_vals((Magma_solid_time/(365*24*60*60)-(start_time_Myr*1e6))/1e6), 'bo', label="Magma Ocean Duration")
axes[0,0].set_ylabel("Magma Ocean Time (Myr)")
axes[0,0].legend()
# Subplot 2: Final H2 Pressure
axes[0,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2), 'cx', label="H2")
axes[0,1].set_ylabel("Final H2 (bar)")
axes[0,1].legend()
# Subplot 3: Final CO2 Pressure
axes[1,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO2), 'kx', label="CO2")
axes[1,0].set_ylabel("Final CO2 (bar)")
axes[1,0].legend()
# Subplot 4: Final O2 Pressure
axes[1,1].loglog(init_H_plot/EO, adjust_y_vals(Oxygen_fug), 'yx', label="O2")
axes[1,1].set_ylabel("Final O2 (bar)")
axes[1,1].legend()
# Subplot 5: Final CO Pressure
axes[2,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CO), 'rx', label="CO")
axes[2,0].set_ylabel("Final CO (bar)")
axes[2,0].legend()
# Subplot 6: Final Atmospheric H2O
axes[2,1].loglog(init_H_plot/EO, adjust_y_vals(Atmo_H2O), 'bx', label="Atmo H2O")
axes[2,1].set_ylabel("Final Atmos H2O (bar)")
axes[2,1].legend()
# Subplot 7: Final CH4 Pressure
axes[3,0].loglog(init_H_plot/EO, adjust_y_vals(Atmo_CH4), 'mx', label="CH4")
axes[3,0].set_xlabel("Initial Hydrogen Inventory")
axes[3,0].set_ylabel("Final CH4 (bar)")
axes[3,0].legend()
# Subplot 8: Final Surface H2O Inventory
axes[3,1].loglog(init_H_plot/EO, adjust_y_vals(Final_surface_water), 'go', label="Surface H2O")
axes[3,1].set_xlabel("Initial Hydrogen Inventory")
axes[3,1].set_ylabel("Final Surf H2O (bar)")
axes[3,1].legend()

# Adjust layout and save the figure
plt.tight_layout()
#plt.savefig("final_atmospheric_pressures_vs_H.png", dpi=300)


## Attempt to plot single model run
try:
    Plot_fun(0)
except:
    print ('Single plot failed')


plt.show()
