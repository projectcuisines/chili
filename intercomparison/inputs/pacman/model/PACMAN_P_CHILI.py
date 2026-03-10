import numpy as np
global y_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
global solve_Tsurf,Plot_compare
from other_functions import *
from Simple_Climate_test import solve_Tsurf,Plot_compare
from Melt_volatile_partitoning_EXP_v4 import Call_MO_Atmo_equil,buffer_fO2
from fe_speciation import Iron_speciation_smooth, Iron_speciation_smooth2
from numba import jit
from scipy.integrate import *
#from escape_functions import * #use this to exactly reproduce calculations in paper
from escape_functions_corrected import * #use this for all other calculations
from stellar_funs import *
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
import pylab as plt
from model_options import MODEL_OPTS
import time
from all_classes_NPM import *
from radiative_functions_6dim_CO_as_N2 import my_interp,my_water_frac,my_fH2O,my_fH2,my_fCO2,my_fCO,my_fN2,correction
import os
import time
import logging

#log = logging.getLogger()

#os.environ["OMP_NUM_THREADS"] = "1"

def forward_model(Switch_Inputs,Planet_inputs,Init_conditions,Numerics,Stellar_inputs,MC_inputs):
    import time


    heating_switch = Switch_Inputs.heating_switch # Controls locus of internal heating, keep default values
    Metal_sink_switch = Switch_Inputs.Metal_sink_switch    
    do_solid_evo = Switch_Inputs.do_solid_evo

    RE = Planet_inputs.RE #Planet radius relative Earth
    ME = Planet_inputs.ME #Planet mass relative Earth
    pm = Planet_inputs.pm #Average mantle density
    rc = Planet_inputs.rc #Metallic core radius (m)
    Total_Fe_mol_fraction_start = Planet_inputs.Total_Fe_mol_fraction # iron mol fraction in mantle
    
    Planet_sep = Planet_inputs.Planet_sep #planet-star separation (AU)
    albedoC = Planet_inputs.albedoC    #cold state albedo
    albedoH = Planet_inputs.albedoH    #hot state albedo
   
    #Stellar parameters
    tsat_XUV = Stellar_inputs.tsat_XUV #XUV saturation time
    Stellar_Mass = Stellar_inputs.Stellar_Mass #stellar mass (relative sun)
    fsat = Stellar_inputs.fsat 
    beta0 = Stellar_inputs.beta0
    epsilon = Stellar_inputs.epsilon
    
    #generate random seed for this forward model call
    #np.random.seed(int(time.time()))
    #seed_save = np.random.randint(1,1e9)

    ## Initial volatlie and redox conditions:
    Init_solid_H2O = Init_conditions.Init_solid_H2O #initial H in solid mantle (kg), NOT H2O
    Init_fluid_H2O = Init_conditions.Init_fluid_H2O #initial H in fluid reservoirs (kg), NOT H2O
    Init_solid_O= Init_conditions.Init_solid_O
    Init_fluid_O = Init_conditions.Init_fluid_O
    Init_solid_FeO1_5 = Init_conditions.Init_solid_FeO1_5
    Init_solid_FeO = Init_conditions.Init_solid_FeO
    Init_fluid_CO2 = Init_conditions.Init_fluid_CO2 #initial C in fluid reservoirs (kg), NOT CO2
    Init_solid_CO2= Init_conditions.Init_solid_CO2 #initial C in solid mantle (kg), NOT CO2

    #Escape parameters
    mult = MC_inputs.esc_c ## for when transition from diffusion to XUV
    mix_epsilon = MC_inputs.esc_d # fraction energy goes into escape above O-drag
    Te_input_escape_mod = MC_inputs.Tstrat
    Thermosphere_temp = MC_inputs.ThermoTemp

    #Interior parameters
    visc_offset = MC_inputs.interiora 
    heatscale = MC_inputs.interiore

    #Impact parameters/non thermal escape (not used/zero in this version of the model)
    constant_loss_NT = MC_inputs.esc_a
    tdc = MC_inputs.esc_b  

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)

    Tsolidus = 1473.0 #surface conditions
    Tliquid = Tsolidus + 600.0 #surface conditions
    ppn2 = 1e5 #background N2 is fixed in this model (cannot be changed here because climate grid must be recomputed)

    Start_time = Switch_Inputs.Start_time
    Max_time = Switch_Inputs.Max_time
    
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

    MM_H2O = 18.01528
    MM_CO2 = 44.01
    MM_O2 = 32.0
    MM_CO = 28.01
    MM_H2 = 2.016
    MM_CH4 = 16.04
    MM_O = 16.0

    
    def func2_get_fO2(log10fO2,XFe2O3_over_XFeO,P,T,Total_Fe):
        fO2 = 10**(log10fO2[0])
        #print('fO2',fO2)
        [A,B,C] = [27215 ,6.57 ,0.0552]
        fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
        f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
        fudge = 1-(1/(np.log10(fO2) - np.log10(f_critical)) )
        XAl2O3 = 0.022423 
        XCaO = 0.0335 
        XNa2O = 0.0024 
        XK2O = 0.0001077 
        terms1 =  11492.0/T - 6.675 - 2.243*XAl2O3
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        fO2_new = np.exp( (np.log(XFe2O3_over_XFeO) - fudge + 1.8282 * Total_Fe - (terms1+terms2+terms3+terms4))/0.196 ) 
        #print(np.log10(fO2_new) - np.log10(fO2))
        if fO2_new < f_critical:
            return np.log10(fO2_new)
        else:
            return np.log10(fO2_new) - np.log10(fO2)
        
    def get_fO2_smooth(XFe2O3_over_XFeO,P,T,Total_Fe): ## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
    
        [A,B,C] = [27215 ,6.57 ,0.0552]
        fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

        f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
 
        XAl2O3 = 0.022423 
        XCaO = 0.0335 
        XNa2O = 0.0024 
        XK2O = 0.0001077 
        terms1 =  11492.0/T - 6.675 - 2.243*XAl2O3
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        fO2 =  np.exp( (np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe -(terms1+terms2+terms3+terms4) )/0.196)
    
        #.196*log(fO2) + terms1+terms2+terms3+terms4 - 1.828 * Total_Fe = np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe 
   
        if XFe2O3_over_XFeO <= 0:
            return f_critical 
        elif fO2 >f_critical*10:
            return fO2
        else:     #    modified mix
            quick_sol = fsolve(func2_get_fO2, x0=np.log10(f_critical*5), args=(XFe2O3_over_XFeO,P,T,Total_Fe),xtol=1e-4)
            fO2_new = 10**(quick_sol[0]+0.0)
            return fO2_new

    def func2_get_fO2_Sossi(log10fO2,XFe2O3_over_XFeO,P,T,Total_Fe):
        fO2 = 10**(log10fO2[0])
        #print('fO2',fO2)
        [A,B,C] = [27215 ,6.57 ,0.0552]
        fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
        f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
        fudge = 1-(1/(np.log10(fO2) - np.log10(f_critical)))

        #XFeO1pt5_over_XFeO = XFe2O3_over_XFeO*2
        
        fO2_new = 10**(np.log10(fO2_IW)+((np.log10(2*XFe2O3_over_XFeO)+1.53)/0.252))

        #print(np.log10(fO2_new) - np.log10(fO2))
        if fO2_new < f_critical:
            return np.log10(fO2_new)
        else:
            return np.log10(fO2_new) - np.log10(fO2)    
    

    def get_fO2_smooth_Sossi(XFe2O3_over_XFeO,P,T,Total_Fe): ## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
    
        [A,B,C] = [27215 ,6.57 ,0.0552]
        fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

        f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))

        #XFeO1pt5_over_XFeO = XFe2O3_over_XFeO*2
        
        fO2 = 10**(np.log10(fO2_IW)+((np.log10(2*XFe2O3_over_XFeO)+1.53)/0.252))
       
        if XFe2O3_over_XFeO <= 0:
            return f_critical 
        elif fO2 >f_critical*10:
            return fO2
        else:     #    modified mix
            quick_sol = fsolve(func2_get_fO2_Sossi, x0=np.log10(f_critical*5), args=(XFe2O3_over_XFeO,P,T,Total_Fe),xtol=1e-4)
            fO2_new = 10**(quick_sol[0]+0.0)
            return fO2_new



    global initialize_everything,time0,time1,time2,time3,int_time0,int_time1
    global integration_time_array

    integration_time_array = np.array([0,0,0,0,0,0])
    time0=0.
    time1=0.
    time2=0.
    time3=0.
    int_time0 = 0.0
    int_time1 = 0.0
    initialize_everything = 0.0
    import time
    new_t = np.linspace(Start_time/1e9,Max_time/1e9,100000)
    #print ('Max_time',Max_time)


    #mixing lumunosity evolutions to interpolate between stellar masses:
    if Stellar_Mass == 0.118: #empirical correction to match observed luminosity
        [Relative_total_Luma,Relative_XUV_luma,Absolute_total_Luma,Absolute_XUV_Luma] = main_sun_fun(new_t,0.11,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        [Relative_total_Lumb,Relative_XUV_lumb,Absolute_total_Lumb,Absolute_XUV_Lumb] = main_sun_fun(new_t,0.13,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        Relative_total_Lum = 0.986*((.13-.118)*Relative_total_Luma + (.118-.11)*Relative_total_Lumb)/(.13 - .11)
        Relative_XUV_lum = 0.986*((.13-.118)*Relative_XUV_luma + (.118-.11)*Relative_XUV_lumb)/(.13 - .11)
        Absolute_total_Lum = 0.986*((.13-.118)*Absolute_total_Luma + (.118-.11)*Absolute_total_Lumb)/(.13 - .11)
        Absolute_XUV_Lum = 0.986*((.13-.118)*Absolute_XUV_Luma + (.118-.11)*Absolute_XUV_Lumb)/(.13 - .11)
    elif Stellar_Mass == 0.122:
        [Relative_total_Luma,Relative_XUV_luma,Absolute_total_Luma,Absolute_XUV_Luma] = main_sun_fun(new_t,0.11,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        [Relative_total_Lumb,Relative_XUV_lumb,Absolute_total_Lumb,Absolute_XUV_Lumb] = main_sun_fun(new_t,0.13,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        Relative_total_Lum = 0.986*((.13-0.122)*Relative_total_Luma + (0.122-.11)*Relative_total_Lumb)/(.13 - .11)
        Relative_XUV_lum = 0.986*((.13-0.122)*Relative_XUV_luma + (0.122-.11)*Relative_XUV_lumb)/(.13 - .11)
        Absolute_total_Lum = 0.986*((.13-0.122)*Absolute_total_Luma + (0.122-.11)*Absolute_total_Lumb)/(.13 - .11)
        Absolute_XUV_Lum = 0.986*((.13-0.122)*Absolute_XUV_Luma + (0.122-.11)*Absolute_XUV_Lumb)/(.13 - .11)
    elif Stellar_Mass == 0.2624: #empirical correction to match observed luminosity
        [Relative_total_Luma,Relative_XUV_luma,Absolute_total_Luma,Absolute_XUV_Luma] = main_sun_fun(new_t,0.20,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        [Relative_total_Lumb,Relative_XUV_lumb,Absolute_total_Lumb,Absolute_XUV_Lumb] = main_sun_fun(new_t,0.30,tsat_XUV,beta0,fsat) #Calculate stellar evolution
        Relative_total_Lum =1.0*((.30-0.2624)*Relative_total_Luma + (0.2624-.20)*Relative_total_Lumb)/(.30 - .20)
        Relative_XUV_lum = 1.0*((.30-0.2624)*Relative_XUV_luma + (0.2624-.20)*Relative_XUV_lumb)/(.30 - .20)
        Absolute_total_Lum = 1.0*((0.30-0.2624)*Absolute_total_Luma + (0.2624-.20)*Absolute_total_Lumb)/(.30 - .20)
        Absolute_XUV_Lum = 1.0*((.30-0.2624)*Absolute_XUV_Luma + (0.2624-.20)*Absolute_XUV_Lumb)/(.30 - .20)
    else:
        [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum] = main_sun_fun(new_t,Stellar_Mass,tsat_XUV,beta0,fsat) #Calculate stellar evolution

    ASR_new = (Absolute_total_Lum/(16*3.14159*(Planet_sep*1.496e11)**2) ) #ASR flux through time (not accounting for bond albedo)

    Te_ar = (ASR_new/5.67e-8)**0.25
    Tskin_ar = Te_ar*(0.5**0.25) ## Skin temperature through time
    Te_fun = interp1d(new_t*1e9*365*24*60*60,Tskin_ar) #Skin temperature function, used in OLR calculations
    ASR_new_fun = interp1d(new_t*1e9*365*24*60*60, ASR_new) #ASR function, used to calculate shortwave radiation fluxes through time
    AbsXUV = interp1d(new_t*1e9*365*24*60*60 , Absolute_XUV_Lum/(4*np.pi*(Planet_sep*1.496e11)**2)) #XUV function, used to calculate XUV-driven escape

    @jit(nopython=True) 
    def fff2(logXFe2O3,XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW):
        XFe2O3 = np.exp(logXFe2O3[0])
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + (Total_Fe-2*XFe2O3) * (56.0+16.0)
        if (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2+3*16)/m_sil) <0:
            return -1e8
        ## g per mol of BSE, so on next line mol Xfe / mol BSE * gXfe/molXfe / g/mol BSE = g Xfe / mol BSe / g/mol BSE = gXfe/g BSE
        terms1 = 0.196*np.log( 1e-5*(MMW/0.032) * (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2.0+3.0*16.0)/m_sil) / (4*np.pi*(rp**2)/g)) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        terms = terms1+terms2+terms3+terms4  
        return -(np.log((XFe2O3 /(Total_Fe -2*XFe2O3))) + 1.828 * Total_Fe - terms)**2.0 

    @jit(nopython=True)         
    def solve_fO2_F_redo_always(y4,P,T,Total_Fe,Mliq,rs,rp,MMW,guess_F_FeO1_5): 
        if 2>1:
            XAl2O3 = 0.022423 
            XCaO = 0.0335
            XNa2O = 0.0024 
            XK2O = 0.0001077 
            XMgO = 0.478144  
            XSiO2 =  0.4034   

            initialize_fast = np.array(float(-50.0))
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)


            if (guess_F_FeO1_5>0.0)and(Mliq>0.0):
                #initialize_fast = np.array( float(-50.0) + 0*np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0) )  )
                #initialize_fast = np.array( np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0) )  )
                initialize_fast = np.array( float(0.0 + 1.0*np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0))  ))
            
            logXFe2O3 =  nelder_mead(fff2, x0=initialize_fast, bounds=np.array([[-100.0], [0.0]]).T, args = (XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW), tol_f=1e-10,tol_x=1e-10, max_iter=1000)
            XFe2O3 = np.exp(logXFe2O3.x[0])           

            XFeO =(Total_Fe - 2*XFe2O3)
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0)
            F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
            F_FeO = XFeO * (56.0 + 16.0) / m_sil 
            fO2_out =  (MMW/0.032) *(y4 - (0.5*16/(56+1.5*16)) * Mliq * XFe2O3*(56.0*2.0+3.0*16.0)/m_sil ) / (4*np.pi*(rp**2)/g)
        return [XFeO,XFe2O3,fO2_out,F_FeO1_5,F_FeO]


    @jit(nopython=True)         
    def solve_fO2_F_redo(y4,P,T,Total_Fe,Mliq,rs,rp,MMW,guess_F_FeO1_5): 
        if T > Tsolidus:
            XAl2O3 = 0.022423 
            XCaO = 0.0335
            XNa2O = 0.0024 
            XK2O = 0.0001077 
            XMgO = 0.478144  
            XSiO2 =  0.4034   

            initialize_fast = np.array(float(-50.0))
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)

            if (guess_F_FeO1_5>0.0)and(Mliq>0.0):
                #initialize_fast = np.array( float(-50.0) + 0*np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0) )  )
                #initialize_fast = np.array( np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0) )  )
                initialize_fast = np.array( float(0.0 + 1.0*np.log( guess_F_FeO1_5*m_sil / (56.0*2.0+3.0*16.0))  ))
            
            logXFe2O3 =  nelder_mead(fff2, x0=initialize_fast, bounds=np.array([[-100.0], [0.0]]).T, args = (XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW), tol_f=1e-10,tol_x=1e-10, max_iter=1000)
            XFe2O3 = np.exp(logXFe2O3.x[0])           

            XFeO =(Total_Fe - 2*XFe2O3)
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0)
            F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
            F_FeO = XFeO * (56.0 + 16.0) / m_sil 
            fO2_out =  (MMW/0.032) *(y4 - (0.5*16/(56+1.5*16)) * Mliq * XFe2O3*(56.0*2.0+3.0*16.0)/m_sil ) / (4*np.pi*(rp**2)/g)

        else:
            fO2_out =  (MMW/0.032) *(y4 / (4*np.pi*(rp**2)/g))
            XFeO = 0.0
            XFe2O3 = 0.0
            F_FeO1_5 = 0.0
            F_FeO = 0.0
        return [XFeO,XFe2O3,fO2_out,F_FeO1_5,F_FeO]


    Init_Tsurf = 3434.3434343 #3808.081632653061
    Tsurf_ad=np.array(Init_Tsurf)
    Guess_ar = np.array([1., 1., 1., 1., 1., 1e20, 0.0001, 0.0001, .1,.1,.1,.1,.1,.5,1e-3,1e-1,1e-10,1e-4])
    Guess_ar2 = np.copy(Guess_ar)
    Stored_things = np.array([1e-12,1e6,250.0,1e6,0.028,100e5])

    def initialize_crude(Init_fluid_H2O, rc,Init_fluid_O,Init_fluid_CO2,ASR_input,Mliq,Mantle_Temperature):
        global integration_time_array
        Tsurf_iterate = Init_Tsurf #3430.103#3500.0
        Pressure_surface = (Init_fluid_H2O+Init_fluid_CO2)*g/(4*np.pi*6371000**2)
        #import pdb
        #pdb.set_trace()
        Temp_linspace = np.linspace(3400.0,3450.0,100) #Mantle_Temperature
        if Mantle_Temperature>3500:
            Temp_linspace = np.linspace(3400,Mantle_Temperature-50,500)
        old_abs = 9e12
        Guess_MMW = 0.028
        Guess_best = np.copy(Guess_ar)
        for temp_i in Temp_linspace:            
            #print (temp_i,Tsurf_iterate)
      
            #[XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(Init_fluid_O,Pressure_surface,temp_i,Total_Fe_mol_fraction,Mliq,rc,rp,Guess_MMW,-0.04) 
            mf_C = Init_fluid_CO2/Mliq #mass fraction C
            mf_H = Init_fluid_H2O/Mliq # mass fraction H
            #f_O2_input = fO2/1e5
            #print ('temp_i,Init_fluid_O,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction')
            #print (temp_i,Init_fluid_O,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction)
            [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(temp_i,Init_fluid_O,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction_start)
            f_O2_input = PO2
            integration_time_array = integration_time_array + new_additions
            #print ('atmo',PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check)
            Temp_guess = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2])
            FH2O = m_H2O
            FCO2 = m_CO2
            FH2_melt = m_H2
            P = PH2O+ PH2+ PCO2+ PCO+ PCH4+PO2

            MH2O_in = 0.018 * natm * PH2O/P
            MH2_in = 0.002 * natm * PH2/P
            MCO2_in = 0.044 * natm * PCO2/P
            MCO_in = 0.028 * natm * PCO/P

            ll = rp - rc
            Tsolidus_visc = Tsolidus
            visc =  viscosity_fun(Mantle_Temperature,pm,visc_offset,temp_i,float(Tsolidus_visc))  
            Ra =np.max([0.0,alpha * g * (Mantle_Temperature -temp_i) * ll**3 / (kappa * visc)  ])
            qm = (k_cond/ll) * (Mantle_Temperature - temp_i) * (Ra/Racr)**(1/3.)
            input_flux = ASR_input+qm
            Te_ar = (ASR_input/5.67e-8)**0.25
            Tskin_new = Te_ar*(0.5**0.25)
            Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
            if Te_input_escape > 350:
                Te_input_escape = 350.0
            if Te_input_escape < 150.0:
                Te_input_escape = 150.0
            [SurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Tskin_new,MH2_in,input_flux ,temp_i)
            #Plot_compare(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,input_flux,SurfT)

            H2O_Pressure_surface = PH2O*1e5
            CO2_Pressure_surface = PCO2*1e5
            UpdateP = P*1e5
            fO2 = PO2*1e5
            UpdateMMW = (fO2*0.032 + H2O_Pressure_surface*0.018 + CO2_Pressure_surface*0.044 + PH2*1e5*0.02+ PCO*0.028*1e5 + ppn2*0.028)/UpdateP

            #import pdb
            #pdb.set_trace()
            SurfT_diff = (SurfT-temp_i)**2
            #print(SurfT,temp_i,new_abs,SurfT_diff)
            if (SurfT_diff<old_abs)and(new_abs<10.0):
                #print ("best so far",temp_i,SurfT_diff,input_flux)
                Pressure_surface = UpdateP
                Guess_best = np.copy(Temp_guess)
                Guess_MMW = UpdateMMW
                Tsurf_iterate = SurfT
                old_abs = SurfT_diff
        return [Tsurf_iterate,Pressure_surface,Guess_best,Guess_MMW]

    import time
    model_run_time = time.time()



    def system_of_equations(t0,y): #System of equations for magma ocean evolution
        tic = time.time()
        if (tic - model_run_time)>60*60*5: ## stop iteration after 5 hours
            print ("TIMED OUT")
            return np.nan*[0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0,0.0,0.0,0.0,0.0,
                           0.0]

        global integration_time_array
        global initialize_everything
        #print('t0',t0)
        #print('y',y)
        global solid_switch,liquid_switch,liquid_switch_worked
        global phi_final,ynewIC,switch_counter,solid_counter
        global time0,time1,time2,time3,int_time0,int_time1
        Total_Fe_mol_fraction = Total_Fe_mol_fraction_start 
        if Metal_sink_switch==1:
            Total_Fe_mol_fraction = y[17]
        
        Mantle_mass = (4./3. * np.pi * pm * (y[2]**3 - rc**3))
        Mantle_mass_div = Mantle_mass if (np.isfinite(Mantle_mass) and Mantle_mass>0) else 1e-30

        
        #################################################################################
        #### If in magma ocean phase
        if  (y[8]>Tsolidus):
            t00 = time.time()

            #Tsurf_ad[...] = y[8]
            beta = 1./3. #Convective heatflow exponent

            #Calculate surface melt fraction
            if Tsurf_ad > Tliquid:
                actual_phi_surf = 1.0
            elif Tsurf_ad < Tsolidus:
                actual_phi_surf = 0.0
            else:
                actual_phi_surf =( Tsurf_ad -Tsolidus)/(Tliquid - Tsolidus)
       
            ll = np.max([rp - rc+0*y[2],1.0]) ## length scale is depth of magma ocean pre-solidification (even if melt <0.4)
            Qr = qr(t0,Start_time,heatscale)+np.exp(-(t0/(1e9*365*24*60*60)-4.5)/5.0)*20e12/((4./3.)* np.pi * pm *  (rp**3.0-rc**3.0))    
            Mliq = Mliq_fun(y[1],rp,y[2],pm)
            Mcrystal = (1-actual_phi_surf)*Mliq
            phi_final = actual_phi_surf

            AB = albedoH 

            Tsolidus_Pmod = Tsolidus
            Tsolidus_visc = Tsolidus        
            visc =  viscosity_fun(y[7],pm,visc_offset,Tsurf_ad,float(Tsolidus_visc))     

            if np.isnan(visc):
                #print ('Viscosity error')
                return [0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0] 

            ASR_input = float((1-AB)*ASR_new_fun(t0))
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
            if Te_input_escape > 350:
                Te_input_escape = 350.0
            if Te_input_escape < 150.0:
                Te_input_escape = 150.0
            #log.info(f"t0: {t0}, Te_input_escape: {Te_input_escape}, ASR_input: {ASR_input}, albedo: {AB}")

            '''
            if (ASR_input < min_ASR):
                ASR_input = min_ASR            
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te         
            '''
            safe_Mliq = max(Mliq,1e-12)                
            mf_C = y[12]/safe_Mliq
            mf_H = y[1]/safe_Mliq

            if initialize_everything == 0:
                #print ('initialize_crude(y[1], rc,y[4],y[12],ASR_input,Mliq,y[7])')
                #print (y[1], rc,y[4],y[12],ASR_input,Mliq,y[7])
                [tryT,Psurf,Guess_best,Guess_MMW] = initialize_crude(y[1], rc,y[4],y[12],ASR_input,Mliq,y[7])
                Tsurf_ad[...] = tryT
                Guess_ar[...] = Guess_best
                Stored_things[5] = Psurf
                Stored_things[4] = Guess_MMW
                initialize_everything = 1


            Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_ad) * ll**3 / (kappa * visc)  ])
            qm = (k_cond/ll) * (y[7] - Tsurf_ad) * (Ra/Racr)**beta
            #First guess
            [PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2] = Guess_ar
            P = PH2O+ PH2+ PCO2+ PCO+ PCH4+PO2
            MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
            MH2_in = (MM_H2/1000.0) * natm * PH2/P
            MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
            MCO_in = (MM_CO/1000.0) * natm * PCO/P  + (MM_O2/1000.0)*natm * PO2/P
            Pguess = P*1e5
            GuessMMW = (y[22]*0.032 + PH2O*1e5*0.018 + PCO2*1e5*0.044 + PH2*1e5*0.02+ PCO*0.028*1e5 + ppn2*0.028)/(Pguess)
            [GuessSurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_ad)

            '''   
            Guess_Fe2O3 = 0.0
            if Mliq>0:
                Guess_Fe2O3 = ((56*2.+16*3)*y[4]/(16.0)) / Mliq # because free O is only one of the oxygens in Fe2O3)   
                print ('Guess_Fe2O3',Guess_Fe2O3)
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],Pguess,GuessSurfT,Total_Fe_mol_fraction,Mliq,y[2],rp,GuessMMW,Guess_Fe2O3)
            f_O2_input = np.max([fO2,1e-20])/1e5
            print ('Actual Fe2O3',F_FeO1_5)
            '''


            #[np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012)]
            [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(GuessSurfT,y[4],mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction)

            integration_time_array = integration_time_array + new_additions
            if (solid_graph<1e-30)or(np.isnan(solid_graph)):
                solid_graph = 1e-30
            Guess_ar[...] = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2])           

            def func2(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                global integration_time_array
                if Tsurf_in[0]<=200.0:
                    return 1e9+(Tsurf_in[0])**2
                #print('inside',Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc)
                #print('shouldntchange',SurfT,ll)
                #import pdb
                #pdb.set_trace()
                Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_in[0]) * ll**3 / (kappa * visc)  ])
                qm = (k_cond/ll) * (y[7] - Tsurf_in[0]) * (Ra/Racr)**beta
                [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(Tsurf_in[0],fO2_in,mf_C,mf_H,Mliq,Guess_ar,Guess_ar,Total_Fe_mol_fraction)
                integration_time_array = integration_time_array + new_additions
                P = PH2O+ PH2+ PCO2+ PCO+ PCH4 +PO2
                MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
                MH2_in = (MM_H2/1000.0) * natm * PH2/P
                MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
                MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
                [SurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_in[0])
                return SurfT - Tsurf_in[0]

 
            '''
            def func2_wrap(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                difference_lin , OLR,PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm =  func2(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input) 
                return difference_lin
            '''

            global int_time0,int_time1
            def func(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                global int_time0,int_time1
                global integration_time_array
                if Tsurf_in<=200.0:
                    return 1e9+(Tsurf_in)**2
                Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_in) * ll**3 / (kappa * visc)  ])
                qm = (k_cond/ll) * (y[7] - Tsurf_in) * (Ra/Racr)**beta
                aa1 = time.time()
                #print ('Guess_ar',Guess_ar)

                [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction)
                #print ('Total_Fe_mol_fraction in func',Total_Fe_mol_fraction)
                #print('solid_graph',solid_graph)
                integration_time_array = integration_time_array + new_additions
                if solid_graph<1e-30:
                    solid_graph = 1e-30
                Guess_ar2[...] = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2])
                #print(Guess_ar2)
                aa2 = time.time()
                P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
                MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
                MH2_in = (MM_H2/1000.0) * natm * PH2/P
                MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
                MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
                [SurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_in) 
                aa3 = time.time()
                int_time0 = int_time0+aa2-aa1
                int_time1 = int_time1 + aa3 - aa2
                return (SurfT - Tsurf_in)**2,OLR,PH2O, PH2, PCO2, PCO, PCH4,PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o


            def func_wrap(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                difference_square,OLR,PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o = func(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)
                return difference_square

            t01 = time.time()
            quick_sol = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-10,Tsurf_ad+10],method='bounded',options={'maxiter':1000,'xatol':1e-8})
            SurfT = quick_sol.x+0.0

            failure_marker = 0
            new_abs = abs(quick_sol.fun)
   
            ## Attempt to solve for surface temperature that balances ASR, OLR, interior heatflow, and partitions volatiles between atmosphere and magma ocean - this is numerically very tricky, hence many attempts/approaches/guesses are tried out
            if new_abs > 0.001:
                failure_marker = 1
                quick_sol2 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-100,Tsurf_ad+10],method='bounded',options={'maxiter':1000,'xatol':1e-8})          
                if abs(quick_sol2.fun)<new_abs:  
                    new_abs = abs(quick_sol2.fun)  
                    SurfT = quick_sol2.x+0.0
                if (new_abs > 0.001):
                    failure_marker = 2
                    quick_sol3 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-5,Tsurf_ad+5],method='bounded',options={'maxiter':1000,'xatol':1e-8})           
                    #quick_sol = fsolve(func2, x0=Tsurf_ad+0.0, args=(f_O2_input,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),xtol=1e-6)
                    #SurfT = quick_sol[0]+0.0
                    ##SurfT_in = Tsurf_ad*0 +  SurfT
                    if abs(quick_sol3.fun)<new_abs: 
                        new_abs = abs(quick_sol3.fun)   
                        SurfT = quick_sol3.x+0.0
                    if (new_abs > 0.1):  
                        failure_marker = 3
                        quick_sol4 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-.5,Tsurf_ad+.5],method='bounded',options={'maxiter':1000,'xatol':1e-8})
                        if abs(quick_sol4.fun) < new_abs:
                            new_abs = abs(quick_sol4.fun) 
                            SurfT = quick_sol4.x+0.0
                        if (new_abs>1.0):
                            failure_maker = 4
                            dumb_loop = 0
                            while dumb_loop < 50:
                                #print('dumb_loop',dumb_loop)
                                mod_up = np.random.uniform(0.001,10)
                                mod_down = 10**np.random.uniform(0,2)
                                quick_sol_loop = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-mod_down,Tsurf_ad+mod_up],method='bounded',options={'maxiter':1000,'xatol':1e-8})
                                if abs(quick_sol_loop.fun) < new_abs:
                                    new_abs = abs(quick_sol_loop.fun)
                                    SurfT = quick_sol_loop.x+0.0
                                if new_abs<1.0:
                                    dumb_loop = 100
                                dumb_loop = dumb_loop + 1

                            if (new_abs>1):
                                failure_marker = 5
                                quick_sol5 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[200.0,5000.0],method='bounded',options={'maxiter':1000,'xatol':1e-8})
                                if abs(quick_sol5.fun) < new_abs:
                                    SurfT = quick_sol5.x+0.0
            
                                        #if SurfT<1000.0:
                                        #    import pdb
                                        #    pdb.set_trace()
                    ##difference_lin , OLR,PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm =  func2(SurfT_in,f_O2_input,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)
            t02 = time.time()

            ## use surface temperature solution to obtain atmosphere + magma ocean composition for subsequent calculations
            difference_square,OLR,PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o = func(SurfT,y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)       
            #if (t0>5625171756555359.0)and(qm>100): 
            #    import pdb
            #    pdb.set_trace()   
            #print ('failure_marker',failure_marker)
            #print ('difference_square',difference_square,'Ra',Ra,'y[7] - SurfT',y[7] - SurfT,'OLR',OLR,'SurfT',SurfT)

            '''
            #try nelder-mead here for speed! analgous to PACMAN - doesnt work if you can't jit the multiphase melt equilibration
            initialize_fast = np.array(Tsurf_ad+0.0)
            ace1 =  nelder_mead(funTs_general2, x0=initialize_fast, bounds=np.array([[200.0], [5000.0]]).T, args = (f_O2_input,mf_C,mf_H,Mliq,Guess_ar,visc,float(y[7])), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
            SurfT = ace1.x[0]
            '''

            #print('output',quick_sol)

            '''
            [PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions]  = Call_MO_Atmo_equil(SurfT,f_O2_input,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2)
            integration_time_array = integration_time_array + new_additions
            #print('saved Ps',[PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check])
            if solid_graph<1e-30:
                solid_graph = 1e-30
            Guess_ar[...] = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check])

            Ra =np.max([0.0,alpha * g * (y[7] - SurfT+0.0) * ll**3 / (kappa * visc)  ])
            qm = (k_cond/ll) * (y[7] - SurfT+0.0) * (Ra/Racr)**beta
            '''

            #print ('atmo',PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check)
            FH2O = m_H2O
            FCO2 = m_CO2
            FH2_melt = m_H2
            P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
            n_C_atm = natm *(PCH4/P + PCO2/P + PCO/P)
            n_C_diss =  m_CO2 * Mliq/(MM_CO2/1000.0)
#            n_C_graphite = solid_graph/ (0.012011/Mliq)
            n_C_graphite = Mliq*solid_graph/ (0.012011)
            #print ('n_C_graphite',n_C_graphite)
            n_H_atm = natm * (4*PCH4/P + 2 * PH2O/P + 2 * PH2/P) 
            n_H_diss =  2 * m_H2O*Mliq/(MM_H2O/1000.0) + 2 * m_H2*Mliq/(MM_H2/1000.0)
            n_H2O_H_diss = 2 * m_H2O*Mliq/(MM_H2O/1000.0)

            n_O_diss =  2*m_CO2 * Mliq/(MM_CO2/1000.0) + m_H2O*Mliq/(MM_H2O/1000.0)
            n_O_atm = natm *(PH2O/P + 2*PCO2/P + PCO/P + 2*PO2/P) 

            MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
            MH2_in = (MM_H2/1000.0) * natm * PH2/P
            MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
            MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
            MCO_in_escape = (MM_CO/1000.0) * natm * PCO/P 

            #NEW VERSION
            #[XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth_Sossi(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)  #Sossi
            [XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)   #Kress & Carm
            #print ('PO2,P*1e5,SurfT,Total_Fe_mol_fraction')
            #print(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)
            
            XFeO = new_Total_Fe / (1 + 2 * (XFe2O3_over_XFeO))   #for Kress & Carmichael fO2 parameterization
            XFe2O3 =XFeO * (XFe2O3_over_XFeO)  

            ''' #OLD VERSION delete once new version is in good working shape
            #log_XFe2O3_over_XFeO = Iron_speciation_Sossi(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)
            #[log_XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth_Sossi(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)  #Sossi
            [log_XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)   #Kress & Carm
            #print ('PO2,P*1e5,SurfT,Total_Fe_mol_fraction')
            #print(PO2,P*1e5,SurfT,Total_Fe_mol_fraction)
            
            #XFeO = new_Total_Fe / (1 + 2 * 10**(log_XFe2O3_over_XFeO))      #for Sossi parameterization
            #XFe2O3 =XFeO * 10**(log_XFe2O3_over_XFeO)                       #for Sossi parameterization
            XFeO = new_Total_Fe / (1 + 2 * np.exp(log_XFe2O3_over_XFeO))   #for Kress & Carmichael fO2 parameterization
            XFe2O3 =XFeO * np.exp(log_XFe2O3_over_XFeO)  '''                  #for Kress & Carmichael fO2 parameterization
            
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0) + (Total_Fe_mol_fraction-XFeO-2*XFe2O3)*56.0
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + Total_Fe_mol_fraction_start * (56.0+16.0) 
            F_FeO = XFeO * (56.0 + 16.0) / m_sil 
            F_FeO1_5 = XFe2O3 * (2*56.0 + 3*16.0) / m_sil 
            F_Fe = (Total_Fe_mol_fraction - new_Total_Fe)*56/m_sil
            if Metal_sink_switch==1:
                F_Fe_integrated = (Total_Fe_mol_fraction_start - new_Total_Fe)*56/m_sil
            #print('compared',Total_Fe_mol_fraction,new_Total_Fe,Total_Fe_mol_fraction_start)
            #print ('F_FeO',F_FeO,'F_FeO1_5',F_FeO1_5,'F_Fe',F_Fe,'Total_Fe_mol_fraction',Total_Fe_mol_fraction,'Total_Fe_mol_fraction_start',Total_Fe_mol_fraction_start)

            #'''

            ##Diagnostic plotting (can be useful for finding bugs)
            if (1>2):#(y[8]<=Tsolidus+10):
            #if (qm <3)and(qm>0):
            #if t0/(365*24*60*60)-1e7 > 7.49552e7:#for crazy CO2 in CLima grid, qm ==0:
            #if t0/(365*24*60*60)-1e7 > 1.8e7:
                T_space = np.linspace(300,3500,100)
                OLR_space = np.copy(T_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(T_space[kz], Te_input_escape,MH2O_in, MCO2_in,MCO_in,MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.semilogy(T_space,OLR_space)
                plt.semilogy(T_space,0*T_space + ASR_input)
                plt.semilogy([SurfT,SurfT],[np.min(OLR_space),np.max(OLR_space)])

                Te_space = np.linspace(150,350,10)
                OLR_space = np.copy(Te_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_space[kz],MH2O_in, MCO2_in,MCO_in,MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.semilogy(Te_space,OLR_space)
                plt.semilogy(Te_space,0*OLR_space + ASR_input)
                plt.semilogy([Te_input_escape,Te_input_escape],[np.min(OLR_space),np.max(OLR_space)])

                MH2O_in_space = np.logspace(15,23,32)
                OLR_space = np.copy(MH2O_in_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_input_escape,MH2O_in_space[kz], MCO2_in,MCO_in,MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.loglog(MH2O_in_space,OLR_space)
                plt.loglog(MH2O_in_space,0*OLR_space + ASR_input)
                plt.loglog([MH2O_in,MH2O_in],[np.min(OLR_space),np.max(OLR_space)])

                MCO2_in_space = np.logspace(15,22,14) 
                OLR_space = np.copy(MCO2_in_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_input_escape,MH2O_in, MCO2_in_space[kz],MCO_in,MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.loglog(MCO2_in_space,OLR_space)
                plt.loglog(MCO2_in_space,0*OLR_space + ASR_input)
                plt.loglog([MCO2_in,MCO2_in],[np.min(OLR_space),np.max(OLR_space)])

                MCO_in_space = np.logspace(15,22,8) 
                OLR_space = np.copy(MCO_in_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_input_escape,MH2O_in, MCO2_in, MCO_in_space[kz],MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.loglog(MCO_in_space,OLR_space)
                plt.loglog(MCO_in_space,0*OLR_space + ASR_input)
                plt.loglog([MCO_in,MCO_in],[np.min(OLR_space),np.max(OLR_space)])

                MH2_in_space = np.logspace(15,23,8)
                OLR_space = np.copy(MH2_in_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(SurfT, Te_input_escape,MH2O_in, MCO2_in, MCO_in,MH2_in_space[kz], 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = my_interp(SurfT,Te_input_escape,MH2O_in, MCO2_in, MCO_in,MH2_in_space[kz]),MCO2_in,0.0,0.0,0.0,0.0,MCO2_in
                    OLR_space[kz] = OLR
                plt.figure()
                plt.loglog(MH2_in_space,OLR_space)
                plt.loglog(MH2_in_space,0*OLR_space + ASR_input)
                plt.loglog([MH2_in,MH2_in],[np.min(OLR_space),np.max(OLR_space)])

                #import pdb
                #pdb.set_trace()
                plt.show()
#            '''
            #[SurfT_meh,new_abs,OLR]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,SurfT) #sus


            #if SurfT<2050:
            #    import pdb
            #    pdb.set_trace()
            #    Plot_compare(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,SurfT+0.0)
            #Plot_compare(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_ad)
            #plt.show()

            H2O_Pressure_surface = PH2O*1e5
            CO2_Pressure_surface = PCO2*1e5
            CO_Pressure_surface = PCO*1e5
            H2_Pressure_surface = PH2*1e5
            Pressure_surface = P*1e5

            fO2 = PO2 * 1e5
            #print ('SurfT',SurfT)
            Tsurf_ad[...] = SurfT+0.0
            y[8] = SurfT
            #print ('y[8]',Tsurf_ad)
             

            #Ra =np.max([0.0,alpha * g * (y[7] -SurfT) * ll**3 / (kappa * visc)  ])
            #qm = (k_cond/ll) * (y[7] - SurfT) * (Ra/Racr)**beta
        
            thermal_cond = k_cond # W/m2/K
            qc = thermal_cond * (Tsurf_ad - y[7]) / (rp-rc)
        
            #T_base = adiabat(rc,y[7],alpha,g,cp,rp) 
            #T_solidus = adiabat(float(y[2]),y[7],alpha,g,cp,rp)
            #visc_solid = (80e9 / (2*5.3e15)) * (1e-3 / 0.5e-9)**2.5 * np.exp((240e3+0*100e3)/(8.314*T_solidus)) * np.exp( - 26 * 0.0)/pm  
            #ll_solid = y[2]-rc
                
        
            #Ra_solid = alpha * g * (T_base -T_solidus) * ll_solid**3 / (kappa * visc_solid) 
            #delta_u = k_cond * (y[7] - SurfT) / qm
               

            if adiabat(rc,y[7],alpha,g,cp,rp) > sol_liq(rc,g,pm,rp,0.0,0.0): #ignore pressure overburden when magma ocean solidifying
                rs_term = 0.0
            else: # if at solidus, start increasing it according to mantle temperature cooling
                #import pdb
                #pdb.set_trace()
                rs_term = rs_term_fun(float(y[2]),a1,b1,a2,b2,g,alpha,cp,pm,rp,y[7],0.0) #doesnt seem to affect things
                    
            #try making qm consistent - trying now - didnt work
            # try making all like Tsurf_ad
            # check why y[7] stops cooling

            #try very precise Tsurf initial
            # try starting init mantle ardoun 3500 to allow wiggle room for Tsurf without exceeding upper bound in solver

            # ideas to speed up
            # rework the guess stuff so that dy/dt depends only on y and t, then see if can get LSODA working
        
            if y[7]>Tsurf_ad:
                if heating_switch == 1:
                    numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) -  4.0*np.pi*((rp)**2)*qm 
                elif heating_switch == 0 :
                    numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) -  4*np.pi*((rp)**2)*qm  
                    #print('t1',(4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0),'t2',-  4*np.pi*((rp)**2)*qm  )
                    #print('qm',qm)
                else:
                    print ('ERROR')
                denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                dy7_dt = numerator/denominator #this is Tp
            else:
                if heating_switch == 1:
                    numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) +  4.0*np.pi*((rp)**2)*qc 
                elif heating_switch == 0:
                    numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) +  4*np.pi*((rp)**2)*qc
                    #print('t1',(4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) ,'t2',4*np.pi*((rp)**2)*qc  )
                    #print('qc',qc)
                else:
                    print ('ERROR')
                denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                dy7_dt = numerator/denominator
            #print('numerator',numerator,'denominator',denominator,'Tsurf_ad',Tsurf_ad)
            #if abs(dy7_dt>0)and(qm<100):
            #    import pdb
            #    pdb.set_trace()

            #if SurfT<Tsolidus:
            #    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0]
                #print ('surface below freezing')
                #numerator = - 4*np.pi*(rp**2)*qm  + (4./3.) * np.pi * pm * Qr * (rp**3 - rc**3) 
                #denominator = pm * cp * (4./3.) *np.pi * (rp**3 - rc**3) 
                #dy7_dt = numerator/denominator

#            if SurfT<2000.0:
#                import pdb
#                pdb.set_trace()

            #print ('dy7_dt',dy7_dt)
            
            ASR = ASR_input

            heat_atm = OLR - ASR  
        
            t03 = time.time()
            time0= time0 + t01-t00 #the first part of function
            time1= time1 + t02-t01 #solving Tsurf and magma ocean?
            time2 = time2 + t03-t02 #last part of function including final magma ocean partitioning

            #if SurfT > y[7]:
            #    true_balance = - heat_atm - qc
            #else:
            #    true_balance = - heat_atm + qm
        
        ###################################################################################  
        ####################################################################################  
        #y[9] = OLR
        #y[10] = ASR
        #y[11] = qm
        #if y[8]<=y[7]:
        #    y[11] = qm
        #else:
        #    y[11] = -qc
        
        elif (y[8]<=Tsolidus):  #terminate evolution once surface temperature drops below solidus
            return [0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0] 

        '''
        if SurfT<Tsolidus:
            fudge = 0.0
            rs_term = 0.0     
            numerator = - 4*np.pi*(rp**2)*qm  + (4./3.) * np.pi * pm * Qr * (rp**3 - rc**3) 
            denominator = pm * cp * (4./3.) *np.pi * (rp**3 - rc**3) 
            dy7_dt = numerator/denominator
            ## isnt capturing transition, makes sense since still to melt equilibration etc. - will need to fix before can really test contigencies
        '''
        #print ('integration_time_array',integration_time_array)

        t04 = time.time()
        #print ('rs_term',rs_term,'dy7_dt',dy7_dt)
        dy2_dt = rs_term * dy7_dt
        drs_dt = dy2_dt
        rs = np.min([y[2],rp])     


        ## updated stratospheric temperature
        Te_ar = (ASR_input/5.67e-8)**0.25
        Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
        if Te_input_escape > 350:
            Te_input_escape = 350.0
        if Te_input_escape < 150.0:
            Te_input_escape = 150.0

        actual_phi_vol, act_visc, Va = temp_meltfrac(rc,rp,alpha,pm,y[7],cp,g,0.0,0.0)
        #act_phi_surf and act_visc not used
        #melt_vol_frac will be exported as y[32]

        # Boundary layer thickness (from Lebrun 2013 and Schaefer 2016)
        safe_qm = max(qm, 1e-12)
        BLT = k_cond * ( ( y[7] - SurfT ) / safe_qm )
        # BLT will be stored as y[34]

        water_frac = my_water_frac(SurfT,Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)

        #fCO = f_i[0,:] #1 in Ev_array
        #fN2 = f_i[1,:]
        #fH2O = f_i[2,:]
        #fCO2 = f_i[3,:]
        #fH2 = f_i[4,:] #5 in Ev_array

        atmo_H2O = np.max([H2O_Pressure_surface*water_frac,0.0])
        #print(EM_ppCO2_o,CO2_Pressure_surface)
        
        #[XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,Tsurf_ad,Total_Fe_mol_fraction,Mliq,rs,rp,Stored_things[4]) #
        
        fO2_pos = np.max([0,fO2])
    
        
        
        #frac_h2o_upper = my_fH2O(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        frac_h2o_upper = np.min([H2O_Pressure_surface/(fO2 +H2O_Pressure_surface +ppn2+CO_Pressure_surface+H2_Pressure_surface+CO2_Pressure_surface), my_fH2O(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)]) ### FIX WHEN DIFFUSION LIMITED FIXED
        if frac_h2o_upper < 0:
            #frac_h2o_upper =  0.0#H2_Pressure_surface / Pressure_surface
            frac_h2o_upper =  atmo_H2O/(fO2+atmo_H2O+ppn2+CO_Pressure_surface+H2_Pressure_surface+atmo_CO2)#0.0#H2_Pressure_surface / Pressure_surface
            if frac_h2o_upper < 1e-6:
                frac_h2o_upper = 0.0 #numerical cutoff
        frac_h2_upper = my_fH2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if frac_h2_upper < 0:
            frac_h2_upper =  H2_Pressure_surface / Pressure_surface
        frac_co_upper = my_fCO(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in_escape,MH2_in)*MCO_in_escape/MCO_in
        if frac_co_upper < 0:
            frac_co_upper =  CO_Pressure_surface / Pressure_surface
        frac_co2_upper = my_fCO2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if frac_co2_upper < 0:
            frac_co2_upper =  CO2_Pressure_surface / Pressure_surface
        frac_n2_upper = my_fN2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        #print ('fracs',frac_h2o_upper,frac_h2_upper)
        #print('fracs CO2',frac_co2_upper,my_fCO2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in))
        #print('fracs CO',frac_co_upper,my_fCO(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in))
        #frac_ch4_upper = (1e5*PCH4)/Pressure_surface #fudge to include CH4
        #import pdb
        #pdb.set_trace()
        frac_ch4_upper = (1e5*PCH4)/(atmo_H2O  + CO2_Pressure_surface + CO_Pressure_surface+  H2_Pressure_surface +1e5*PCH4 + ppn2 + fO2_pos)
        #if frac_ch4_upper_2<frac_ch4_upper:
        #    frac_ch4_upper = np.copy(frac_ch4_upper_2)
     
        if (H2O_Pressure_surface<1e-5)and(frac_h2o_upper<1e-9)and(frac_h2_upper<1e-9): #for numerical efficiency, low threshold for escape cutoff
            frac_h2o_upper = 0.0
            atmo_H2O = 0.0
            H2O_Pressure_surface = 0.0
        if frac_co_upper<1e-9:
            frac_co_upper = 0
        if frac_co2_upper<1e-9:
            frac_co2_upper = 0

        if y[12]<1e11: #throttle C escape for numerical efficiency
            frac_co2_upper = 0.0
            frac_co_upper = 0.0

        #print('frac_co2_upper',frac_co2_upper,'frac_co_upper',frac_co_upper)
        #######################
        ## Atmosphsic escape calculations
        ## diffusion limited escape: ## this is wrong, need to go back and treat H2 and H2O separately in diffusion function
        fCO2_p = (1- frac_h2o_upper-frac_h2_upper)*CO2_Pressure_surface / (CO2_Pressure_surface+fO2_pos+ppn2 + CO_Pressure_surface  )
        fO2_p = (1- frac_h2o_upper-frac_h2_upper)*fO2_pos / (CO2_Pressure_surface+fO2_pos+ppn2 + CO_Pressure_surface  )
        fN2_p = (1- frac_h2o_upper-frac_h2_upper)*ppn2 / (CO2_Pressure_surface+fO2_pos+ppn2 + CO_Pressure_surface  )
        fCO_p = (1- frac_h2o_upper-frac_h2_upper)*CO_Pressure_surface / (CO2_Pressure_surface+fO2_pos+ppn2 + CO_Pressure_surface  )
        
        #XUV-driven escape
        XH_upper = (2*frac_h2o_upper+2*frac_h2_upper+4*frac_ch4_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper) ## assumes CO2->CO + O
        XH = XH_upper #2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p +fCO_p)
        XO = (2*fO2_p+frac_h2o_upper+frac_co2_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper)
        XCO = (frac_co2_upper + frac_co_upper + frac_ch4_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper)

        mol_diff_H2O_flux = 0.5*better_H_diffusion(XH,Te_input_escape,g,XCO,XO,frac_n2_upper) #mol H2/m2/s

        true_epsilon = find_epsilon(Thermosphere_temp,RE,ME,float(AbsXUV(t0)), XO, XH, XCO,epsilon,mix_epsilon)
 
        [mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Thermosphere_temp,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XCO) #kg/m2/s
        numH = ( mH_Odert3 / 0.001 ) # mol H / m2/ s
        numO = ( mO_Odert3 / 0.016 ) # mol O / m2/ s
        numC = ( mC_Odert3 / 0.028 ) # mol CO / m2/ s
        #print ('nums',numH,numO,numC)
        if numH < 2*mol_diff_H2O_flux: ## if diffusion limit exceeds XUV-driven, shift diffusion-limit downward / for E & V opposite: if diff limit exceeds XUV, escape limit = diff limit
            numH = 2*mol_diff_H2O_flux # Earth/Venus
            #mol_diff_H2O_flux = 0.5*np.copy(numH)   #for TRAPPIST
    
        #if XH < 1e-10:
        #    mol_diff_H2O_flux = 0.0
        #    numH = 0.0
        #    numC =0.0
        #    numO = 0.0

        if (XH < 1e-12)and(XH>1e-13):
            mol_diff_H2O_flux = mol_diff_H2O_flux*(XH-1e-13)/9e-13
            numH = numH*(XH-1e-13)/9e-13
            numC = numC*(XH-1e-13)/9e-13
            numO = numO*(XH-1e-13)/9e-13
        if XH<=1e-13:
            mol_diff_H2O_flux = 0.0
            numH = 0.0
            numC =0.0
            numO = 0.0

        ## Combined escape flux, weighted by H abundance:
        #w1 = mult*(2./3 -  XH_upper)**4
        #w2 = XH_upper**4
        w1 = mult*np.max([0.0,(2./3 -  XH_upper)**3])
        w2 = XH_upper**3

        Mixer_H = (w1*2*mol_diff_H2O_flux + w2 * numH ) / (w1+w2) # mol H / m2 /s
        Mixer_O = (w1*0.0 + w2 * numO ) / (w1+w2)
        Mixer_CO = (w1*0.0 + w2 * numC ) / (w1+w2)
        #Mixer_Build = 0.5* Mixer_H - Mixer_O  ## CO2 drag doesn't affect redox (CO2 drag typically negligible anyway)

        #escape = 4*np.pi*rp**2 * Mixer_H*0.018/2 ## kg H2O /s
        #net_escape = 4*np.pi*rp**2 * Mixer_Build*0.016 ## kg O2/s
        #CO2_loss =  4*np.pi*rp**2 * Mixer_C*0.044 ## kg CO2/s

        frac_C_as_CO = (frac_co2_upper + frac_co_upper)/(frac_co2_upper + frac_co_upper + frac_ch4_upper) ##this is a fudge because dont properly to CH4 loss
        '''escape_H = 4*np.pi*rp**2 * Mixer_H*0.001 # kg H / s
        escape_O = 4*np.pi*rp**2 * (Mixer_O+Mixer_CO*frac_C_as_CO)*0.016 #kg O / s 
        escape_C = 4*np.pi*rp**2 * Mixer_CO*0.012 #Kg C /s  
        #print ('escape',escape_H,escape_O,escape_C)'''

        if not MODEL_OPTS.escape_on:
            #Set escape to not operate
            escape_H = 0
            escape_O = 0
            escape_C = 0
        
        else:
            escape_H = 4*np.pi*rp**2 * Mixer_H*0.001 # kg H / s
            escape_O = 4*np.pi*rp**2 * (Mixer_O+Mixer_CO*frac_C_as_CO)*0.016 #kg O / s 
            escape_C = 4*np.pi*rp**2 * Mixer_CO*0.012 #Kg C /s  
            #print ('escape',escape_H,escape_O,escape_C)

        ## add some nonthermal escape
        # H2O_Pressure_surface,CO2_Pressure_surface, fO2_pos, 1e5 for N2
        NT_H2O = (constant_loss_NT * atmo_H2O / Pressure_surface)*0.018 ## Previously was massive because H2O total being used!
        NT_CO2 = (constant_loss_NT * CO2_Pressure_surface / Pressure_surface)*0.044 # shouldn't be any different to before
        NT_O2 = (constant_loss_NT * fO2_pos / Pressure_surface)*0.016
        #escape = escape + NT_H2O
        #CO2_loss = CO2_loss + NT_CO2
        #net_escape = net_escape - NT_O2
        # End non-thermal escape

        # done with escape calculations
        #######################
        #######################

        ## Find ocean depth and land fraction: (not relevant here)
        Ocean_depth = (0.018/Stored_things[4]) * (1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
        if Ocean_depth > Max_depth:
            Linear_Ocean_fraction = 1.0
            Ocean_fraction = 1.0
        else:
            Linear_Ocean_fraction = (Ocean_depth/Max_depth) 
            Ocean_fraction = (Ocean_depth/Max_depth)**0.25 ## Crude approximation to Earth hypsometric curve
    
        ## Melt and crustal oxidation variables:
        actual_phi_surf_melt = 0.0
        F_CO2 = 0.0
        F_H2O = 0.0
        O2_consumption = 0.0
        OG_O2_consumption = 0.0
        Plate_velocity = 0.0
        crustal_depth = 0.0
        Melt_volume = 0.0
        Poverburd = Pressure_surface #fO2_pos +H2O_Pressure_surface + CO2_Pressure_surface + ppn2
        Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(Poverburd),float(0.0*y[0]/Mantle_mass_div)) # Need to change for hydrous melting sensitivity test 
        

        if rs<rp:
            fudge = 1.0-(rs/rp)**1000.0 ## helps with numerical issues
        else:
            fudge = 0.0

        if SurfT<Tsolidus:
            fudge = 0.0
            rs_term = 0.0     

        mu_O = 16.0
        mu_FeO_1_5 = 56.0 + 1.5*16.0 
        mu_FeO = 56.0 + 16.0   
        

        wet_oxidation = 0.0
        water_crust = 0.0
        [total_water_loss_surf,total_water_gain_interior,new_wet_oxidation,total_wet_FeO_lost,total_wet_FeO1_5,total_wet_H2_gained] = [0,0,0,0,0,0]
       
        O_imp_sink = 0.0 ## Zero out impacts for nominal model
        
        O2_dry_magma_oxidation  = 0.0
        Fe_dry_magma_oxidation = 0.0
        O2_magma_oxidation = 0.0
        Fe_magma_oxidation = 0.0
        O2_consumption = 0.0
        new_wet_oxidation = 0.0 

        y[22] = fO2
        MMW = (fO2_pos*0.032 + atmo_H2O*0.018 + CO2_Pressure_surface*0.044 + PH2*1e5*0.02+ PCO*0.028*1e5+ ppn2*0.028)/Pressure_surface
        y[9] = OLR
        y[10] = ASR

        if Metal_sink_switch == 0:
            y[17] = new_Total_Fe
        elif Metal_sink_switch ==1:
            if new_Total_Fe <= y[17]:
                y[17] = new_Total_Fe
        
        y[11] = qm
        Stored_things[...] = np.array([fO2,OLR,ASR,qm,MMW,Pressure_surface])
        #print ('Stored_things',Stored_things)

        y[18] = n_C_diss 
        y[19] = n_C_atm
        y[20] = n_C_graphite 
        y[21] = n_H_atm
        y[23] = n_H_diss 



        O2_magma_oxidation_solid = np.copy(O2_magma_oxidation)+ O2_consumption*0.032
        O2_magma_oxidation_volatile = np.copy(O2_magma_oxidation) + O2_consumption*0.032 + O_imp_sink
        Fe_magma_oxidation_solid = np.copy(Fe_magma_oxidation) + 4*O2_consumption*(0.056+0.016) # kg FeO/s
        Fe_magma_oxidation_solid_1_5 = (mu_FeO_1_5/mu_FeO) * np.copy(Fe_magma_oxidation) +  2*O2_consumption*(0.016*3 + 0.056*2) # kg FeO1.5/s
        
        if (y[12]<1e15)and(y[12]>1e13): #taper off to avoid -ve C (added bottom right run)
            escape_C = escape_C * (np.log10(y[12]) - np.log10(1e13))/ (np.log10(1e15) - np.log10(1e13))
        elif y[12] <= 1e13 :
            escape_C = 0
        else:
            escape_C = np.copy(escape_C)

        ###########################################################################################
        #newFTL and oldFTL refer to models with and without melt trapping during magma ocean soldification, respectively
        F_TL = np.max([0.0,np.min([0.3,-0.3*(1e6*365*24*60*60)*dy7_dt/600.0])]) #newFTL
        #dy0_dt = fudge * 4*np.pi * pm * kH2O * (FH2O*MM_H2/MM_H2O) * rs**2 * drs_dt  + total_water_gain_interior - F_H2O*0.018 #oldFTL

        H_melt_gains = (FH2O*MM_H2/MM_H2O)  *( (1 - F_TL)*kH2O + F_TL) + FH2_melt*F_TL
        dy0_dt = fudge * 4*np.pi * pm * rs**2 * drs_dt* H_melt_gains + total_water_gain_interior - F_H2O*0.018 #newFTL
        #print ('drs_dt',drs_dt)
        if drs_dt < 0.0:
            dy0_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) 
            dy1_dt = - dy0_dt - escape_H
        else:
            #dy1_dt = - fudge * 4*np.pi * pm * kH2O * (FH2O*MM_H2/MM_H2O) * rs**2 * drs_dt  - escape  - total_water_loss_surf + F_H2O*0.018     #oldFTL
            dy1_dt = - fudge * 4*np.pi * pm *rs**2 * drs_dt * H_melt_gains - escape_H  - total_water_loss_surf + F_H2O*0.018    #newFTL
        
        H2O_solid_gain =  fudge * 4*np.pi * pm * (FH2O*MM_O/MM_H2O) * rs**2 * drs_dt *( (1 - F_TL)*kH2O + F_TL) #oxygen in H2O transferred to solid interior
        CO2_solid_gain = fudge * 4*np.pi * pm * (FCO2*2*MM_O/MM_CO2)  * rs**2 * drs_dt *( (1 - F_TL)*kCO2 + F_TL) #oxygen in CO2 transferred to solid interior

        O_transfer_from_Fe_metal = fudge * 4 *np.pi * pm * F_Fe * rs**2 * drs_dt * mu_O / 56.0   
        metal_scalar = 1
        if Metal_sink_switch==1:
            O_transfer_from_Fe_metal = fudge * 4 *np.pi * pm * F_Fe_integrated * rs**2 * drs_dt * mu_O / 56.0 
            metal_scalar = 0 #silicate interior doesnt gain metal (lose free H) if it goes straight to core, but fluid reservoirs do gain oxygen

        dy3_dt = fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) + O2_magma_oxidation_solid + H2O_solid_gain + CO2_solid_gain - metal_scalar*O_transfer_from_Fe_metal ## free O in solid , factor of half because only half is free oxygen
    ### this is correct, only 8 molecular mass free O2 for 56+1.5*16 molecular mass solidified

        if drs_dt < 0.0:
            dy3_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) - metal_scalar*O_transfer_from_Fe_metal# need to put the metal back to conserve O I think
            dy4_dt = -escape_O - fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))+ O_transfer_from_Fe_metal #as above
        else:
            dy4_dt = -escape_O - fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) - O2_magma_oxidation_volatile - H2O_solid_gain - CO2_solid_gain + O_transfer_from_Fe_metal ## magma ocean and atmo, free O   

        dy5_dt = fudge * 4 *np.pi * pm * F_FeO1_5 * rs**2 * drs_dt +  Fe_magma_oxidation_solid_1_5 ## mass FeO1_5 flux, FeO + O2 = 2Fe2O3
        dy6_dt = fudge * 4 * np.pi * pm * F_FeO * rs**2 * drs_dt - Fe_magma_oxidation_solid #F_FeO flux
        dy30_dt = fudge * 4 *np.pi * pm * F_Fe * rs**2 * drs_dt

        if drs_dt < 0.0:
            dy5_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 
            dy6_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 
            dy30_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[30] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 

        #dy13_dt = fudge * 4*np.pi * pm * kCO2 * (FCO2*12.011/MM_CO2) * rs**2 * drs_dt ## mass solid CO2  #oldFTL ## 
        dy13_dt = fudge * 4*np.pi * pm * (FCO2*12.011/MM_CO2)  * rs**2 * drs_dt *( (1 - F_TL)*kCO2 + F_TL) #newFTL
        if drs_dt < 0.0:
            dy13_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) ##mass solid CO2
    
        dy12_dt = -dy13_dt - escape_C ##  escape CO assumed
        
        Weather = 0.0
        Outgas = 0.0

        ###########################################################################################
        # Various convenient tracer variables:
        y[25] = PH2O
        y[26] = PH2
        y[27] = PCO2
        y[28] = PCO
        y[29] = PCH4 
        y[31] = n_H2O_H_diss
        y[32] = actual_phi_vol
        y[34] = BLT
        y[35] = MMW


        n_O_Fe = (F_FeO1_5 * Mliq * 0.5*16/(1.5*16+56))/0.016
        y[16] = n_O_atm
        y[15] = n_O_Fe
        y[14] = n_O_diss
        y[24] = atmo_H2O
        
        t05 = time.time()
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + Total_Fe_mol_fraction_start * (56.0+16.0)

        time3 = time3 + t05-t04 # escape and fluxes and whatnot

        #log.info(f"Inside system_of_equations(), t0: {t0}, Te_input_escape: {Te_input_escape}, OLR: {OLR}, ASR_input: {ASR_input}, albedo: {AB}")

        return [dy0_dt,dy1_dt,dy2_dt,dy3_dt,dy4_dt,
                dy5_dt,dy6_dt,dy7_dt,0.0,0.0,
                0.0, 0.0,dy12_dt,dy13_dt,0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                dy30_dt, 0.0, 0.0, 0.0, 0.0,
                0.0]


    def system_of_equations2(t0,y): #System of equations for solid mantle evolution
        global integration_time_array
        global initialize_everything
        #print('t0',t0)
        #print('y',y)

        global solid_switch,liquid_switch,liquid_switch_worked
        global phi_final,ynewIC,switch_counter,solid_counter
        global time0,time1,time2,time3,int_time0,int_time1
        Total_Fe_mol_fraction = Total_Fe_mol_fraction_start 

        if Metal_sink_switch==1:
            Total_Fe_mol_fraction = y[17]

        Mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))
        Mantle_mass_div = Mantle_mass if (np.isfinite(Mantle_mass) and Mantle_mass>0) else 1e-30

        #################################################################################
        if  (2>1):
 
            beta = 1./3. #Convective heatflow exponent

            #Calculate surface melt fraction
            if Tsurf_ad > Tliquid:
                actual_phi_surf = 1.0
            elif Tsurf_ad < Tsolidus:
                actual_phi_surf = 0.0
            else:
                actual_phi_surf =( Tsurf_ad -Tsolidus)/(Tliquid - Tsolidus)
       
            ll = np.max([rp - y[2],1.0]) ## length scale is depth of magma ocean pre-solidification (even if melt <0.4)
            Qr = qr(t0,Start_time,heatscale)+np.exp(-(t0/(1e9*365*24*60*60)-4.5)/5.0)*20e12/((4./3.)* np.pi * pm *  (rp**3.0-rc**3.0))    
            Mliq = Mliq_fun(y[1],rp,y[2],pm)
            Mcrystal = (1-actual_phi_surf)*Mliq
            phi_final = actual_phi_surf

            AB = albedoH


            Tsolidus_Pmod = Tsolidus
            Tsolidus_visc = Tsolidus   
            visc =  viscosity_fun(y[7],pm,visc_offset,Tsurf_ad,float(Tsolidus_visc))     

            if np.isnan(visc):
                #print ('Viscosity error')
                return [0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0,0.0,0.0,0.0,0.0,
                        0.0] 

            ASR_input = float((1-AB)*ASR_new_fun(t0))
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
            if Te_input_escape > 350:
                Te_input_escape = 350.0
            if Te_input_escape < 150.0:
                Te_input_escape = 150.0

            #log.info(f"t0: {t0}, Te_input_escape: {Te_input_escape}, ASR_input: {ASR_input}, albedo: {AB}")
            #print("ASR_input in sys_of_eq2 = ", ASR_input)
            #print("Te_input_escape in sys_of_eq2 = ", Te_input_escape)

            '''
            if (ASR_input < min_ASR):
                ASR_input = min_ASR            
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te         
            '''
                             
            if Mliq == 0.0:                
                mf_C = y[12]#/Mliq
                mf_H = y[1]#/Mliq

            Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_ad) * ll**3 / (kappa * visc)  ])
            qm = (k_cond/ll) * (y[7] - Tsurf_ad) * (Ra/Racr)**beta
            #First guess
            [PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2] = Guess_ar
            P = PH2O+ PH2+ PCO2+ PCO+ PCH4+PO2
            MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
            MH2_in = (MM_H2/1000.0) * natm * PH2/P
            MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
            MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
            Pguess = P*1e5
            GuessMMW = (y[22]*0.032 + PH2O*1e5*0.018 + PCO2*1e5*0.044 + PH2*1e5*0.02+ PCO*0.028*1e5 + ppn2*0.028)/(Pguess)
            [GuessSurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_ad)

            '''   
            Guess_Fe2O3 = 0.0
            if Mliq>0:
                Guess_Fe2O3 = ((56*2.+16*3)*y[4]/(16.0)) / Mliq # because free O is only one of the oxygens in Fe2O3)   
                print ('Guess_Fe2O3',Guess_Fe2O3)
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],Pguess,GuessSurfT,Total_Fe_mol_fraction,Mliq,y[2],rp,GuessMMW,Guess_Fe2O3)
            f_O2_input = np.max([fO2,1e-20])/1e5
            print ('Actual Fe2O3',F_FeO1_5)
            '''

            #[np.log(m_H2O_g),np.log(m_CO2_g),np.log(PH2O_g),np.log(PCO2_g),np.log(natm_g),np.log(PH2_g),np.log(PCH4_g),np.log(PCO_g),np.log(solid_graph_g*M_melt/0.012)]
            [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(GuessSurfT,y[4],mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction)

            integration_time_array = integration_time_array + new_additions
            Guess_ar[...] = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2])


            def func2(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                global integration_time_array

                if Tsurf_in[0]<=200.0:
                    return 1e9+(Tsurf_in[0])**2
                Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_in[0]) * ll**3 / (kappa * visc)  ])
                qm = (k_cond/ll) * (y[7] - Tsurf_in[0]) * (Ra/Racr)**beta
                [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(Tsurf_in[0],fO2_in,mf_C,mf_H,Mliq,Guess_ar,Guess_ar,Total_Fe_mol_fraction)
                integration_time_array = integration_time_array + new_additions
                P = PH2O+ PH2+ PCO2+ PCO+ PCH4 +PO2
                MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
                MH2_in = (MM_H2/1000.0) * natm * PH2/P
                MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
                MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
                [SurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_in[0])
                return SurfT - Tsurf_in[0]#,OLR,PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm

            '''
            def func2_wrap(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                difference_lin , OLR,PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm =  func2(Tsurf_in,fO2_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input) 
                return difference_lin
            '''
            #print("checkpoint 1: you survived all the initializing stuff")
            global int_time0,int_time1
            def func(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                global int_time0,int_time1
                global integration_time_array
                if Tsurf_in<=200.0:
                    return 1e9+(Tsurf_in)**2
                Ra =np.max([0.0,alpha * g * (y[7] - Tsurf_in) * ll**3 / (kappa * visc)  ])
                qm = (k_cond/ll) * (y[7] - Tsurf_in) * (Ra/Racr)**beta


                [PH2O, PH2, PCO2, PCO, PCH4, PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions,m_H2,new_Total_Fe]  = Call_MO_Atmo_equil(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,Guess_ar2,Total_Fe_mol_fraction)

                integration_time_array = integration_time_array + new_additions
                if solid_graph<1e-30:
                    solid_graph = 1e-30
                Guess_ar2[...] = np.array([PH2O, PH2, PCO2, PCO, PCH4, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,PO2,m_H2])

                P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
                MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
                MH2_in = (MM_H2/1000.0) * natm * PH2/P
                MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
                MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P
                [SurfT,new_abs,OLR,EM_ppCO2_o]  = solve_Tsurf(MH2O_in,MCO2_in,MCO_in,Te_input_escape,MH2_in,ASR_input+qm,Tsurf_in) 

                return (SurfT - Tsurf_in)**2,OLR,PH2O, PH2, PCO2, PCO, PCH4,PO2, natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o


            def func_wrap(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input):
                #print ('ins',Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)
                difference_square,OLR,PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o = func(Tsurf_in,y4_in,mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)
                #print ('outs',difference_square,OLR,PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2)
                return difference_square


            ## Attempt to solve for surface temperature that balances OLR, ASR, and interior heatflow (no magma ocean partitioning this time)
            quick_sol = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[Tsurf_ad-10,Tsurf_ad+10],method='bounded',options={'maxiter':1000,'xatol':1e-8})
            SurfT = quick_sol.x+0.0

            failure_marker = 0
            if abs(quick_sol.fun) > 0.001:
                quick_sol2 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[200.0,5000.0],method='bounded',options={'maxiter':1000,'xatol':1e-8})
                SurfT = quick_sol2.x+0.0
                if abs(quick_sol2.fun) > 0.001:
                    min_bound = np.max([Tsurf_ad-100,200.0])
                    quick_sol3 = minimize_scalar(func_wrap,args=(y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input),bounds=[min_bound,Tsurf_ad+50],method='bounded',options={'maxiter':1000,'xatol':1e-8})
                    SurfT = quick_sol3.x+0.0
                    if abs(quick_sol3.fun) >0.1:
                        failure_marker = 1

            ## use surface temperature solution to obtain variables for subsequent calculations
            difference_square,OLR,PH2O, PH2, PCO2, PCO, PCH4, PO2,natm, m_CO2, m_H2O, mH2O,mH2,mCO2,mCO,mCH4,mO2,solid_graph,graph_check,new_additions, Ra, qm,m_H2,EM_ppCO2_o = func(SurfT,y[4],mf_C,mf_H,Mliq,Guess_ar,visc,ASR_input)  
            #print("checkpoint 2: you survived Tsurf calculations")

  
            FH2O = m_H2O
            FCO2 = m_CO2
            FH2_melt = m_H2
            P = PH2O+ PH2+ PCO2+ PCO+ PCH4 + PO2
            n_C_atm = natm *(PCH4/P + PCO2/P + PCO/P)
            n_C_diss =  m_CO2 * Mliq/(MM_CO2/1000.0)
            #n_C_graphite = solid_graph/ (0.012011/Mliq)
            n_C_graphite = Mliq*solid_graph/ (0.012011)
            #print ('n_C_graphite',n_C_graphite)
            n_H_atm = natm * (4*PCH4/P + 2 * PH2O/P + 2 * PH2/P) 
            n_H_diss =  2 * m_H2O*Mliq/(MM_H2O/1000.0) + 2 * m_H2*Mliq/(MM_H2/1000.0)
            n_H2O_H_diss = 2 * m_H2O*Mliq/(MM_H2O/1000.0)

            n_O_diss =  2*m_CO2 * Mliq/(MM_CO2/1000.0) + m_H2O*Mliq/(MM_H2O/1000.0)
            n_O_atm = natm *(PH2O/P + 2*PCO2/P + PCO/P + 2*PO2/P) 

            MH2O_in = (MM_H2O/1000.0) * natm * PH2O/P
            MH2_in = (MM_H2/1000.0) * natm * PH2/P
            MCO2_in = (MM_CO2/1000.0) * natm * PCO2/P
            MCO_in = (MM_CO/1000.0) * natm * PCO/P + (MM_O2/1000.0)*natm * PO2/P


            H2O_Pressure_surface = PH2O*1e5
            CO2_Pressure_surface = PCO2*1e5
            CO_Pressure_surface = PCO*1e5
            H2_Pressure_surface = PH2*1e5
            Pressure_surface = P*1e5+ppn2

            fO2 = PO2 * 1e5
            Tsurf_ad[...] = SurfT+0.0
            y[8] = SurfT
             

            thermal_cond = k_cond # W/m2/K
            qc = thermal_cond * (Tsurf_ad - y[7]) / (rp-rc)
        

            rs_term = 0.0

            numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) -  4*np.pi*((rp)**2)*qm  
            denominator = (4./3.) * np.pi * pm * cp *(rp**3 - rc**3) 
            dy7_dt = numerator/denominator #0.0 #WHICH ?!?!

            if Tsurf_ad>y[7]: ## addition for conduction
                numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) +  4*np.pi*((rp)**2)*qc
                denominator = (4./3.) * np.pi * pm * cp *(rp**3 - rc**3) 
                dy7_dt = 0.0 #numerator/denominator

            ASR = ASR_input
            

            heat_atm = OLR - ASR

            """  
            if t0/(365*24*60*60)-3e7 > 6e7:
                T_space = np.linspace(300,3500,100)
                OLR_space = np.copy(T_space)
                for kz in range(0,len(OLR_space)):
                    [OLR,EM_ppCO2_o,EM_pH_o,ALK,Mass_oceans_crude,DIC_check,Mass_CO2_atmo] = correction(T_space[kz], Te_input_escape,MH2O_in, MCO2_in,MCO_in,MH2_in, 6371000.0, 9.8, 5e-5, 1e18, 0.028, 1e-6 )
                    OLR_space[kz] = OLR
                plt.figure()
                plt.semilogy(T_space,OLR_space)
                plt.semilogy(T_space,0*T_space + ASR_input)
                plt.semilogy([SurfT,SurfT],[np.min(OLR_space),np.max(OLR_space)])
                plt.show()
            """
         

       
        dy2_dt = rs_term * dy7_dt  #0.0 #WHICH ?!?!
        drs_dt = dy2_dt #0.0 #WHICH ?!?!
        rs = np.min([y[2],rp])     


        ## updated stratospheric temperature
        Te_ar = (ASR_input/5.67e-8)**0.25
        Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
        if Te_input_escape > 350:
            Te_input_escape = 350.0
        if Te_input_escape < 150.0:
            Te_input_escape = 150.0

        
        actual_phi_vol, act_visc, Va = temp_meltfrac(rc,rp,alpha,pm,y[7],cp,g,0.0,0.0)

        # Boundary layer thickness (from Lebrun 2013 and Schaefer 2016)
        safe_qm = max(qm, 1e-12)
        BLT = k_cond * ( ( y[7] - SurfT ) / safe_qm )
        # BLT will be stored as y[34]

        #print("checkpoint 3: you survived BLT calculation")

        water_frac = my_water_frac(SurfT,Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)

        #fCO = f_i[0,:] #1 in Ev_array
        #fN2 = f_i[1,:]
        #fH2O = f_i[2,:]
        #fCO2 = f_i[3,:]
        #fH2 = f_i[4,:] #5 in Ev_array

        atmo_H2O = np.max([H2O_Pressure_surface*water_frac,0.0])
        atmo_CO2 = CO2_Pressure_surface
        atmo_pressure_surf = (fO2+atmo_H2O+ppn2+CO_Pressure_surface+H2_Pressure_surface+atmo_CO2)
        
        #[XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,Tsurf_ad,Total_Fe_mol_fraction,Mliq,rs,rp,Stored_things[4]) #
        
        fO2_pos = np.max([0,fO2])
        frac_h2o_upper = np.min([atmo_H2O/(fO2+atmo_H2O+ppn2+CO_Pressure_surface+H2_Pressure_surface+atmo_CO2), my_fH2O(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)]) 
        if frac_h2o_upper < 0:
            frac_h2o_upper =  atmo_H2O/(fO2+atmo_H2O+ppn2+CO_Pressure_surface+H2_Pressure_surface+atmo_CO2)
            if frac_h2o_upper < 1e-6:
                frac_h2o_upper = 0.0 #numerical cutoff
        frac_h2_upper = my_fH2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if frac_h2_upper < 0:
            frac_h2_upper =  H2_Pressure_surface / atmo_pressure_surf#Pressure_surface
        frac_co_upper = my_fCO(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if frac_co_upper < 0:
            frac_co_upper =  CO_Pressure_surface / atmo_pressure_surf#Pressure_surface
        frac_co2_upper = my_fCO2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if frac_co2_upper < 0:
            frac_co2_upper =  atmo_CO2 / atmo_pressure_surf#Pressure_surface

        frac_ch4_upper = (1e5*PCH4)/(atmo_H2O  + atmo_CO2 + CO_Pressure_surface+  H2_Pressure_surface +1e5*PCH4 + ppn2 + fO2_pos)

        frac_n2_upper = my_fN2(float(SurfT),Te_input_escape,MH2O_in,MCO2_in,MCO_in,MH2_in)
        if (H2O_Pressure_surface<1e-5)and(frac_h2o_upper<1e-9)and(frac_h2_upper<1e-9): #for numerical efficiency, low threshold for escape cutoff
            frac_h2o_upper = 0.0
            atmo_H2O = 0.0
            H2O_Pressure_surface = 0.0

        #######################
        ## Atmosphsic escape calculations
        ## diffusion limited escape: ## this is wrong, need to go back and treat H2 and H2O separately in diffusion function
        fCO2_p = (1- frac_h2o_upper-frac_h2_upper)*atmo_CO2 / (atmo_CO2+fO2_pos+ppn2 + CO_Pressure_surface  )
        fO2_p = (1- frac_h2o_upper-frac_h2_upper)*fO2_pos / (atmo_CO2+fO2_pos+ppn2 + CO_Pressure_surface  )
        fN2_p = (1- frac_h2o_upper-frac_h2_upper)*ppn2 / (atmo_CO2+fO2_pos+ppn2 + CO_Pressure_surface  )
        fCO_p = (1- frac_h2o_upper-frac_h2_upper)*CO_Pressure_surface / (atmo_CO2+fO2_pos+ppn2 + CO_Pressure_surface  )
        mol_diff_H2O_flux = better_diffusion(frac_h2o_upper+frac_h2_upper,Te_input_escape,g,fCO2_p+fCO_p,fO2_p,fN2_p) #mol H2O/m2/s
      
        #XUV-driven escape
        XH_upper = (2*frac_h2o_upper+2*frac_h2_upper+4*frac_ch4_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper) ## assumes CO2->CO + O
        XH = XH_upper #2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p +fCO_p)
        XO = (2*fO2_p+frac_h2o_upper+frac_co2_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper)
        XCO = (frac_co2_upper + frac_co_upper + frac_ch4_upper) / (3*frac_h2o_upper + 2*frac_h2_upper + 4*frac_ch4_upper + 2*fO2_p + 2*frac_co2_upper + frac_co_upper + frac_n2_upper)

        #mol_diff_H2O_flux = 0.5*better_H_diffusion(XH,Te_input_escape,g,XCO,XO,frac_n2_upper) #mol H2/m2/s  
        
        true_epsilon = find_epsilon(Thermosphere_temp,RE,ME,float(AbsXUV(t0)), XO, XH, XCO,epsilon,mix_epsilon)
 
        [mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Thermosphere_temp,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XCO) #kg/m2/s
        numH = ( mH_Odert3 / 0.001 ) # mol H / m2/ s
        numO = ( mO_Odert3 / 0.016 ) # mol O / m2/ s
        numC = ( mC_Odert3 / 0.028 ) # mol CO / m2/ s
        
        if numH < 2*mol_diff_H2O_flux: ## for E & V if XUV-driven exceeds diffusion limit, shift diffusion-limit up
            numH = 2*mol_diff_H2O_flux  #for Earth and Venus
            #mol_diff_H2O_flux = 0.5*np.copy(numH) #for TRAPPIST

        ## Combined escape flux, weighted by H abundance:
        #w1 = mult*(2./3 -  XH_upper)**4
        #w2 = XH_upper**4
        w1 = mult*np.max([0.0,(2./3 -  XH_upper)**3])
        w2 = XH_upper**3

        #if XH < 1e-10:
        #    mol_diff_H2O_flux = 0.0
        #    numH = 0.0
        #    numC =0.0
        #    numO = 0.0
        if (XH < 1e-12)and(XH>1e-13):
            mol_diff_H2O_flux = mol_diff_H2O_flux*(XH-1e-13)/9e-13
            numH = numH*(XH-1e-13)/9e-13
            numC = numC*(XH-1e-13)/9e-13
            numO = numO*(XH-1e-13)/9e-13
        if XH<=1e-13:
            mol_diff_H2O_flux = 0.0
            numH = 0.0
            numC =0.0
            numO = 0.0

        Mixer_H = (w1*2*mol_diff_H2O_flux + w2 * numH ) / (w1+w2) # mol H / m2 /s
        Mixer_O = (w1*0.0 + w2 * numO ) / (w1+w2)
        Mixer_CO = (w1*0.0 + w2 * numC ) / (w1+w2)
        #Mixer_Build = 0.5* Mixer_H - Mixer_O  ## CO2 drag doesn't affect redox (CO2 drag typically negligible anyway)

        #escape_H = 4*np.pi*rp**2 * Mixer_H*0.001 # kg H / s
        #escape_O = 4*np.pi*rp**2 * (Mixer_O+Mixer_CO)*0.016 #kg O / s 
        #escape_C = 4*np.pi*rp**2 * Mixer_CO*0.012 #Kg C /s  
        #print ('escape',escape_H,escape_O,escape_C)

        frac_C_as_CO = (frac_co2_upper + frac_co_upper)/(frac_co2_upper + frac_co_upper + frac_ch4_upper) ##this is a fudge because dont properly to CH4 loss
        '''escape_H = 4*np.pi*rp**2 * Mixer_H*0.001 # kg H / s
        escape_O = 4*np.pi*rp**2 * (Mixer_O+Mixer_CO*frac_C_as_CO)*0.016 #kg O / s 
        escape_C = 4*np.pi*rp**2 * Mixer_CO*0.012 #Kg C /s  '''
        
        if not MODEL_OPTS.escape_on:
            #Set escape to not operate
            escape_H = 0
            escape_O = 0
            escape_C = 0
        
        else:
            escape_H = 4*np.pi*rp**2 * Mixer_H*0.001 # kg H / s
            escape_O = 4*np.pi*rp**2 * (Mixer_O+Mixer_CO*frac_C_as_CO)*0.016 #kg O / s 
            escape_C = 4*np.pi*rp**2 * Mixer_CO*0.012 #Kg C /s 

        ## add some nonthermal escape (zero here)
        # H2O_Pressure_surface,CO2_Pressure_surface, fO2_pos, 1e5 for N2
        NT_H2O = (constant_loss_NT * atmo_H2O / Pressure_surface)*0.018 ## Previously was massive because H2O total being used!
        NT_CO2 = (constant_loss_NT * CO2_Pressure_surface / Pressure_surface)*0.044 # shouldn't be any different to before
        NT_O2 = (constant_loss_NT * fO2_pos / Pressure_surface)*0.016
        #escape = escape + NT_H2O
        #CO2_loss = CO2_loss + NT_CO2
        #net_escape = net_escape - NT_O2
        # End non-thermal escape

        # done with escape calculations
        #######################
        #######################
        #print("checkpoint 4: you survived upper atmos and escape stuff")
        ## Find ocean depth and land fraction:
        Ocean_depth = (0.018/Stored_things[4]) * (1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
        if Ocean_depth > Max_depth:
            Linear_Ocean_fraction = 1.0
            Ocean_fraction = 1.0
        else:
            Linear_Ocean_fraction = (Ocean_depth/Max_depth) 
            Ocean_fraction = (Ocean_depth/Max_depth)**0.25 ## Crude approximation to Earth hypsometric curve
    
        ## Melt and crustal oxidation variables:
        actual_phi_surf_melt = 0.0
        F_CO2 = 0.0
        F_H2O = 0.0
        O2_consumption = 0.0
        OG_O2_consumption = 0.0
        Plate_velocity = 0.0
        crustal_depth = 0.0
        Melt_volume = 0.0
        Poverburd = Pressure_surface
        Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(Poverburd),float(0.0*y[0]/Mantle_mass_div)) # Need to change for hydrous melting sensitivity test 
        
        if rs<rp:
            fudge = 1.0-(rs/rp)**1000.0 ## helps with numerical issues
        else:
            fudge = 0.0

        if SurfT<Tsolidus:
            fudge = 0.0
            rs_term = 0.0     

        mu_O = 16.0
        mu_FeO_1_5 = 56.0 + 1.5*16.0 
        mu_FeO = 56.0 + 16.0   


        wet_oxidation = 0.0
        water_crust = 0.0
        [total_water_loss_surf,total_water_gain_interior,new_wet_oxidation,total_wet_FeO_lost,total_wet_FeO1_5,total_wet_H2_gained] = [0,0,0,0,0,0]
        
        O_imp_sink = 0.0 ## Zero out impacts for nominal model
        
        O2_dry_magma_oxidation  = 0.0
        Fe_dry_magma_oxidation = 0.0
        O2_magma_oxidation = 0.0
        Fe_magma_oxidation = 0.0
        O2_consumption = 0.0
        new_wet_oxidation = 0.0 

        #print("checkpoint 5: you survived ocean stuff")

        y[22] = fO2
        MMW = (fO2_pos*0.032 + atmo_H2O*0.018 + atmo_CO2*0.044 + PH2*1e5*0.02+ PCO*0.028*1e5+ ppn2*0.028)/Pressure_surface
        y[9] = OLR
        y[10] = ASR

        if Metal_sink_switch == 0:
            y[17] = new_Total_Fe
        elif Metal_sink_switch ==1:
            if new_Total_Fe <= y[17]:
                y[17] = new_Total_Fe
        
        y[11] = qm
        Stored_things[...] = np.array([fO2,OLR,ASR,qm,MMW,Pressure_surface])

        y[18] = n_C_diss 
        y[19] = n_C_atm
        y[20] = n_C_graphite 
        y[21] = n_H_atm
        y[23] = n_H_diss 

        F_Fe = 0.0
        if Metal_sink_switch==1:
            F_Fe_integrated = 0.0
        F_FeO1_5 = 0.0
        F_FeO = 0.0

        O2_magma_oxidation_solid = np.copy(O2_magma_oxidation)+ O2_consumption*0.032
        O2_magma_oxidation_volatile = np.copy(O2_magma_oxidation) + O2_consumption*0.032 + O_imp_sink
        Fe_magma_oxidation_solid = np.copy(Fe_magma_oxidation) + 4*O2_consumption*(0.056+0.016) # kg FeO/s
        Fe_magma_oxidation_solid_1_5 = (mu_FeO_1_5/mu_FeO) * np.copy(Fe_magma_oxidation) +  2*O2_consumption*(0.016*3 + 0.056*2) # kg FeO1.5/s
        
        ###########################################################################################
        #newFTL and oldFTL refer to models with and without melt trapping during magma ocean soldification, respectively
        F_TL = np.max([0.0,np.min([0.3,-0.3*(1e6*365*24*60*60)*dy7_dt/600.0])]) #newFTL
        #dy0_dt = fudge * 4*np.pi * pm * kH2O * (FH2O*MM_H2/MM_H2O) * rs**2 * drs_dt  + total_water_gain_interior - F_H2O*0.018 #oldFTL

        H_melt_gains = (FH2O*MM_H2/MM_H2O)  *( (1 - F_TL)*kH2O + F_TL) + FH2_melt*F_TL
        dy0_dt = 0.0 #fudge * 4*np.pi * pm * rs**2 * drs_dt* H_melt_gains + total_water_gain_interior - F_H2O*0.018 #newFTL
        #print ('drs_dt',drs_dt)
        if drs_dt < 0.0:
            dy0_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) 
            dy1_dt = 0.0 #- dy0_dt - escape_H
        else:
            #dy1_dt = - fudge * 4*np.pi * pm * kH2O * (FH2O*MM_H2/MM_H2O) * rs**2 * drs_dt  - escape  - total_water_loss_surf + F_H2O*0.018     #oldFTL
            dy1_dt = 0.0 #- fudge * 4*np.pi * pm *rs**2 * drs_dt * H_melt_gains - escape_H  - total_water_loss_surf + F_H2O*0.018    #newFTL
        
        H2O_solid_gain =  fudge * 4*np.pi * pm * (FH2O*MM_O/MM_H2O) * rs**2 * drs_dt *( (1 - F_TL)*kH2O + F_TL) #oxygen in H2O transferred to solid interior
        CO2_solid_gain = fudge * 4*np.pi * pm * (FCO2*2*MM_O/MM_CO2)  * rs**2 * drs_dt *( (1 - F_TL)*kCO2 + F_TL) #oxygen in CO2 transferred to solid interior

        O_transfer_from_Fe_metal = fudge * 4 *np.pi * pm * F_Fe * rs**2 * drs_dt * mu_O / 56.0   ### CHANGE NEEDED HERE
        metal_scalar = 1
        if Metal_sink_switch==1:
            O_transfer_from_Fe_metal = fudge * 4 *np.pi * pm * F_Fe_integrated * rs**2 * drs_dt * mu_O / 56.0 
            metal_scalar = 0 #silicate interior doesnt gain metal (lose free H) if it goes straight to core, but fluid reservoirs do gain oxygen

        dy3_dt = 0.0 #fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) + O2_magma_oxidation_solid + H2O_solid_gain + CO2_solid_gain - metal_scalar*O_transfer_from_Fe_metal ## free O in solid , factor of half because only half is free oxygen
    ### this is correct, only 8 molecular mass free O2 for 56+1.5*16 molecular mass solidified

        if drs_dt < 0.0:
            dy3_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))# * 0.5*mu_O / (mu_FeO_1_5) 
            dy4_dt = 0.0 #-escape_O - dy3_dt 
        else:
            dy4_dt = 0.0 #-escape_O - fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) - O2_magma_oxidation_volatile - H2O_solid_gain - CO2_solid_gain + O_transfer_from_Fe_metal ## magma ocean and atmo, free O   


        dy5_dt = 0.0 #fudge * 4 *np.pi * pm * F_FeO1_5 * rs**2 * drs_dt +  Fe_magma_oxidation_solid_1_5 ## mass FeO1_5 flux, FeO + O2 = 2Fe2O3
        dy6_dt = 0.0 #fudge * 4 * np.pi * pm * F_FeO * rs**2 * drs_dt - Fe_magma_oxidation_solid #F_FeO flux
        dy30_dt = 0.0 #fudge * 4 *np.pi * pm * F_Fe * rs**2 * drs_dt

        # O2_consumption is mol of free O2
        # 4FeO + O2 -> 2Fe2O3, so 1 mol O2_consumption -> 2 mol Fe2O3 = 2*O2_consumption * M(Fe2O3) for kg/s Fe2O3 = kgFeO1.5
        # similarly 1 mol O2_consumption > 4 mol FeO = 4 * O2_consumption * M(FeO) for kg/s FeO 

        if drs_dt < 0.0:
            dy5_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 
            dy6_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 
            dy30_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[30] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 

        #dy13_dt = fudge * 4*np.pi * pm * kCO2 * (FCO2*12.011/MM_CO2) * rs**2 * drs_dt ## mass solid CO2  #oldFTL ## 
        dy13_dt = 0.0 #fudge * 4*np.pi * pm * (FCO2*12.011/MM_CO2)  * rs**2 * drs_dt *( (1 - F_TL)*kCO2 + F_TL) #newFTL
        if drs_dt < 0.0:
            dy13_dt = 0.0 #fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) ##mass solid CO2
    
        dy12_dt = 0.0 #-dy13_dt - escape_C ##  escape CO assumed
        
        Weather = 0.0
        Outgas = 0.0
        #print("checkpoint 6: YAY you survived all the systems of equations")
        ###########################################################################################
        # Various convenient tracer variables:
        y[25] = PH2O
        y[26] = PH2
        y[27] = PCO2
        y[28] = PCO
        y[29] = PCH4
        y[31] = n_H2O_H_diss
        y[32] = actual_phi_vol
        y[34] = BLT
        y[35] = MMW

        n_O_Fe = (F_FeO1_5 * Mliq * 0.5*16/(1.5*16+56))/0.016
        y[16] = n_O_atm
        y[15] = n_O_Fe
        y[14] = n_O_diss
        y[24] = atmo_H2O

        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + Total_Fe_mol_fraction_start * (56.0+16.0)
        
        #log.info(f"Inside system_of_equations(), t0: {t0}, Te_input_escape: {Te_input_escape}, OLR: {OLR}, ASR_input: {ASR_input}, albedo: {AB}")

        return [dy0_dt,dy1_dt,dy2_dt,dy3_dt,dy4_dt,
                dy5_dt,dy6_dt,dy7_dt,0.0,0.0,
                0.0,0.0,dy12_dt,dy13_dt,0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0,
                dy30_dt, 0.0, 0.0, 0.0, 0.0,
                0.0]
        



    Initial_mantle_temp = 3500.0
    while sol_liq(rc,g,pm,rp,0.0,0.0)>adiabat(rc,Initial_mantle_temp,alpha,g,cp,rp):
        Initial_mantle_temp = Initial_mantle_temp + 200.0
    print ('Initial_mantle_temp',Initial_mantle_temp)
    ## Define initial conditions for integration
    ICs = [Init_solid_H2O, Init_fluid_H2O, rc, Init_solid_O, Init_fluid_O, 
           Init_solid_FeO1_5, Init_solid_FeO, Initial_mantle_temp, Init_Tsurf, 0.0, 
           0.0, 1.5e6, Init_fluid_CO2, Init_solid_CO2, 0.0, 
           0.0, 3999.5, Total_Fe_mol_fraction_start, 0.0, 0.0, 
           0.0, 0.0, 1e-12, 0.0, 0.044, 
           0.0, 0.0, 0.0, 0.0, 0.0, 
           0.0, 0.0, 0.0, 0.0, 0.0, 
           0.0] #init mantle formerly 3500



    ### Various numerical input options (most unused)
    step0 =Numerics.step0 #100.100
    step1=Numerics.step1 #500000.0 #5000 originally
    step2=Numerics.step2 #50000
    step3=Numerics.step3 #1e3
    step4=-999
    tfin0=Numerics.tfin0 #1e7+20000.0 #4530000.0#770#23100#478.2#470#720#20000 #+740 If code breaks before rs increases, then increase this!!
    tfin1=Numerics.tfin1 #1e7+2000e6#7.1769e6#10e6#3.82e6#3e6
    #tfin2=1e7+0.5e6
    tfin3=Numerics.tfin3 #7e9 #2.5890317279765016e+16/(365*24*60*60)#4.5e9#7.96e9
    tfin4 = -999
    if 2>1:
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=step0*365*24*60*60) 
        model_run_time = time.time()
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=step1*365*24*60*60,rtol=1e-5)
        #import pdb
        #pdb.set_trace()
        if (sol2.t[-1]+1e6 < tfin1*365*24*60*60):
            print ('finished way too soon')
            sol2 = 0
            sol = 0
        else:
            print ('seems to have worked')

        total_time = np.concatenate((sol.t,sol2.t))
        total_y = np.concatenate((sol.y,sol2.y),axis=1)
        time = np.copy(total_time)
        y_out = np.copy(total_y)

    fmt = 0 #find magma termination
    i_loop = 1
    while (fmt==0)and(i_loop<len(time)):
        tcheck = (time[i_loop]/(365*24*60*60)-5e7)/1e6

        if (np.max(y_out[0,i_loop:]) == y_out[0,i_loop-1])and(np.max(y_out[1,i_loop:]) == y_out[1,i_loop-1])and(np.max(y_out[7,i_loop:]) == y_out[7,i_loop-1])and(np.max(y_out[8,i_loop:]) == y_out[8,i_loop-1])and(np.max(y_out[2,i_loop:]) == y_out[2,i_loop-1])and(tcheck>0.0001):
            fmt = i_loop
        i_loop = i_loop+1

    if fmt == 0: #never freezes
        fmt = len(total_time) - 2

    solid_time = total_time[fmt:]
    solid_y = total_y[:,fmt:]
    
    #good to here
    Pressure_surface = total_y[22,:]/1e5 + total_y[25,:]+ total_y[26,:]+ total_y[27,:]+ total_y[28,:]+ total_y[29,:]
    f_O2_mantle = np.copy(total_time) #solid mantle
    graph_check_arr = np.copy(total_time)

    print("fmt", fmt, "len(total_time)", len(total_time), "total_y", total_y.shape)

    Mliq = Mliq_fun(total_y[1,-1],rp,total_y[2,-1],pm) 
    if Metal_sink_switch == 1:
        use_this_FeTot = y_out[17,-1]
    else:
        use_this_FeTot = Total_Fe_mol_fraction_start

    #[XFe2O3_over_XFeO,New_Total_Fe] = Iron_speciation_smooth_Sossi(y_out[22,-1]/1e5,Pressure_surface[-1]*1e5,y_out[8,-1],use_this_FeTot)    #Sossi
    [XFe2O3_over_XFeO,New_Total_Fe] = Iron_speciation_smooth(y_out[22,-1]/1e5,Pressure_surface[-1]*1e5,y_out[8,-1],use_this_FeTot)          #Kress & Carm
    XFeO = New_Total_Fe / (1 + 2 * (XFe2O3_over_XFeO))    
    XFe2O3 =XFeO * (XFe2O3_over_XFeO)                     
    
    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0) + (Total_Fe_mol_fraction_start-XFeO-2*XFe2O3)*56.0
    F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
    F_FeO = XFeO * (56.0 + 16.0) / m_sil 
    F_Fe = (use_this_FeTot - New_Total_Fe)*56/m_sil

    total_y[2,fmt:] = total_y[2,fmt:]*0 + rp

    total_y[5,fmt:] = total_y[5,fmt:] + (4./3.) *np.pi * pm * F_FeO1_5  *(rp**3 - y_out[2,-1]**3)  
    total_y[6,fmt:] = total_y[6,fmt:] + (4./3.) *np.pi * pm *  F_FeO * (rp**3 - y_out[2,-1]**3) 
    total_y[30,fmt:] = total_y[30,fmt:] + (4./3.) *np.pi * pm *  F_Fe * (rp**3 - y_out[2,-1]**3) 

    MoltenFe_in_FeO = np.copy(total_y[4,:])
    MoltenFe_in_Fe = np.copy(total_y[4,:])
    MoltenFe_in_FeO1pt5 = np.copy(total_y[4,:])

    F_FeO_ar = np.copy(total_y[4,:])
    F_FeO1_5_ar = np.copy(total_y[4,:])
    F_Fe_ar = np.copy(total_y[4,:])

    total_y_og = np.copy(total_y)
    final_extra_oxygen = 0
    for kk in range(0,len(total_time)):

        iron0 = total_y[30,kk]
        iron3 = total_y[5,kk]*56/(56.0+1.5*16.0)
        iron2 = total_y[6,kk]*56/(56.0+16.0)
        Solid_Mantle_mass = (4./3. * np.pi * pm * (total_y[2,kk]**3 - rc**3))
        Solid_Mantle_mass_div = Solid_Mantle_mass if (np.isfinite(Solid_Mantle_mass) and Solid_Mantle_mass>0) else 1e-30

        moles_Fe = (total_y[30,kk]/0.056)
        moles_FeO = (total_y[6,kk]/(0.056+0.016))
        moles_FeO1pt5 = (total_y[5,kk]/(0.056+1.5*0.016))
        if moles_FeO1pt5>2*moles_Fe: #more oxidized than metal
            iron3 = iron3 - iron0
            iron0 = 0
            new_molesFeO1pt5 = moles_FeO1pt5 - 2*moles_Fe
            new_molesFeO = moles_FeO + 3*moles_Fe
            new_molesFe = 0
            total_y[5,kk] =  new_molesFeO1pt5*(0.056+1.5*0.016)
            total_y[6,kk] = new_molesFeO * (0.056 + 0.016)
            total_y[30,kk] = 0
            #total_y[3,kk] = total_y[3,kk] - 2*moles_Fe*(0.5*0.016)

            if kk>=fmt:
                total_y[15,kk] = total_y[15,kk] - moles_Fe

        else:
            iron0 = iron0 - iron3
            iron3 = 0
            new_molesFeO1pt5 = 0#moles_FeO1pt5 - 2*moles_Fe
            new_molesFeO = moles_FeO + (3/2)*moles_FeO1pt5
            new_molesFe = moles_Fe - (1/2)*moles_FeO1pt5
            total_y[5,kk] =  0
            total_y[6,kk] = new_molesFeO * (0.056 + 0.016)
            total_y[30,kk] = new_molesFe * 0.056
            #total_y[3,kk] = total_y[3,kk] - moles_FeO1pt5*(0.5*0.016)

            if kk>=fmt:
                total_y[15,kk] = total_y[15,kk] - moles_FeO1pt5/2

        try:
            XFe2O3_current = m_sil*(total_y[5,kk]/Solid_Mantle_mass_div)/(56.0*2.0+3.0*16.0)
            XFeO_current = m_sil*(total_y[6,kk]/Solid_Mantle_mass_div)/(56.0+16.0)
            #iron_ratio[k] = iron3/iron2
            #iron_ratio_norm[k][i] = iron3/(iron2+iron3)
            Current_Fe_mol_fraction = XFeO_current+2*XFe2O3_current       
            #f_O2_mantle[kk] = get_fO2_Sossi(0.5*iron3/iron2,Pressure_surface[kk]*1e5,total_y[7,kk],Total_Fe_mol_fraction)
            if Solid_Mantle_mass>0:
                #f_O2_mantle[kk] = get_fO2_smooth_Sossi(XFe2O3_current/XFeO_current,Pressure_surface[kk]*1e5,total_y[7,kk],Current_Fe_mol_fraction)  #Sossi
                f_O2_mantle[kk] = get_fO2_smooth(XFe2O3_current/XFeO_current,Pressure_surface[kk]*1e5,total_y[7,kk],Current_Fe_mol_fraction)  #Kress & Carmich

            else:
                f_O2_mantle[kk] = np.nan
        except:
            f_O2_mantle[kk] = np.nan


        M_melt = Mliq_fun(total_y[1,kk],rp,total_y[2,kk],pm) 
        log10_K1 = 40.07639 - 2.53932e-2 * total_y[8,kk] + 5.27096e-6*total_y[8,kk]**2 + 0.0267 * (Pressure_surface[kk] - 1 )/total_y[8,kk] 
        log10_K2 = - 6.24763 - 282.56/total_y[8,kk] - 0.119242 * (Pressure_surface[kk] - 1000)/total_y[8,kk]
        gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*total_y[22,kk]/1e5)/(1+(10**log10_K1)*(10**log10_K2)*total_y[22,kk]/1e5) 
        gXCO2_melt = 1.0*(44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt) #mass fraction
        graph_check_arr[kk] = (gXCO2_melt*M_melt/(44./1000.0))

    #O_solid_res = (2*y_out[13,:]/0.012) +  (0.5*y_out[0,:]/0.001) + (y_out[5,:]*0.5*16/(56+1.5*16))/0.016
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + Total_Fe_mol_fraction_start * (56.0+16.0)
        free_O_from_metal = (Total_Fe_mol_fraction_start - y_out[17,kk])*56/m_sil
        M_melt = Mliq_fun(total_y[1,kk],rp,total_y[2,kk],pm) 
        final_extra_oxygen = Mliq*((Total_Fe_mol_fraction_start - y_out[17,fmt])*56/m_sil)*16/56.0
        if kk<fmt:
            total_y[4,kk] = total_y[4,kk] + M_melt*free_O_from_metal*16/56.0 ## basically adding back oxygen from metal
                                     # = M_melt*total_y[17,kk]*(56)/m_sil 
            MoltenFe_in_FeO1pt5[kk] = (2*total_y[15,kk])*0.056
            if (free_O_from_metal>0)and(MoltenFe_in_FeO1pt5[kk]==0):
                MoltenFe_in_FeO[kk] =   M_melt*total_y[17,kk]*(56)/m_sil #c.f. (M_melt*XFeO *56 / m_sil )
                MoltenFe_in_Fe[kk] = M_melt*free_O_from_metal
            else:
                Mass_Fe_in_FeO = M_melt*total_y[17,kk]*(56)/m_sil - (2*total_y[15,kk])*0.056
                MoltenFe_in_FeO[kk] =   Mass_Fe_in_FeO #c.f. (M_melt*XFeO *56 / m_sil )
                MoltenFe_in_Fe[kk] = M_melt*free_O_from_metal
                                         
        else:
            #free_O_from_metal = (Total_Fe_mol_fraction_start - y_out[17,fmt])*56/m_sil
            #M_melt = Mliq_fun(total_y[1,fmt],rp,total_y[2,fmt],pm) 
            #total_y[3,kk] = total_y[3,kk] + M_melt*free_O_from_metal*16/56.0
            MoltenFe_in_FeO[kk] = 0.0
            MoltenFe_in_Fe[kk] = 0.0
            MoltenFe_in_FeO1pt5[kk] = 0.0
            M_melt = 0.0

        # NEW VERSION
        if Metal_sink_switch==0:
            #[XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth_Sossi(total_y[22,kk]/1e5,Pressure_surface[kk]*1e5,total_y[8,kk],Total_Fe_mol_fraction_start) #Sossi
            [XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth(total_y[22,kk]/1e5,Pressure_surface[kk]*1e5,total_y[8,kk],Total_Fe_mol_fraction_start) # Kress & Carmich
            XFeO = new_Total_Fe / (1 + 2 * (XFe2O3_over_XFeO))    
            XFe2O3 =XFeO * (XFe2O3_over_XFeO)                     
            F_FeO_ar[kk] = M_melt*XFeO * (56.0 + 16.0) / m_sil 
            F_FeO1_5_ar[kk] = M_melt*XFe2O3 * (2*56.0 + 3*16.0) / m_sil 
            F_Fe_ar[kk] = M_melt*(Total_Fe_mol_fraction_start - new_Total_Fe)*56/m_sil
        else:
            #[XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth2_Sossi(total_y[22,kk]/1e5,Pressure_surface[kk]*1e5,total_y[8,kk],total_y[17,kk])   #Sossi
            [XFe2O3_over_XFeO,new_Total_Fe] = Iron_speciation_smooth2(total_y[22,kk]/1e5,Pressure_surface[kk]*1e5,total_y[8,kk],total_y[17,kk])     #Kress & Carmichael
            XFeO = new_Total_Fe / (1 + 2 * (XFe2O3_over_XFeO))      
            XFe2O3 =XFeO * (XFe2O3_over_XFeO)                       
            F_FeO_ar[kk] = M_melt*XFeO * (56.0 + 16.0) / m_sil 
            F_FeO1_5_ar[kk] = M_melt*XFe2O3 * (2*56.0 + 3*16.0) / m_sil 
            F_Fe_ar[kk] = M_melt*(total_y[17,kk] - new_Total_Fe)*56/m_sil
            MoltenFe_in_Fe[kk] = F_Fe_ar[kk]

    ## add optional solid state evolution      
    if do_solid_evo ==1:
        total_y[1,fmt] = total_y[21,fmt]*0.001
        total_y[0,fmt] = total_y[23,fmt]*0.001 + total_y[0,fmt]
        total_y[12,fmt] = total_y[19,fmt]*0.012
        total_y[13,fmt] =  total_y[18,fmt]*0.012+total_y[20,fmt]*0.012 + total_y[13,fmt]
        total_y[3,fmt] = total_y[3,fmt]+total_y[4,fmt]+final_extra_oxygen - total_y[16,fmt]*0.016 
        total_y[4,fmt] = total_y[16,fmt]*0.016

        if tfin3*365*24*60*60>total_time[fmt]:
            sol3 = solve_ivp(system_of_equations2, [total_time[fmt], tfin3*365*24*60*60], total_y[:,fmt], method = 'RK23', vectorized=False, max_step=10*step1*365*24*60*60,rtol=1e-4)
            print("sol3 shapes:", sol3.t.shape, sol3.y.shape, "total_y pre", total_y.shape)
            #print ('final_extra_oxygen',final_extra_oxygen)
            total_time = np.concatenate((total_time[0:fmt],sol3.t))
            total_y = np.concatenate((total_y[:,0:fmt],sol3.y),axis=1)
        total_y_og = np.copy(total_y)
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    #end optional solid state evolutions

    print(" if you are seeing this you survived the solid evolution stuff ")

    solid_time = total_time[fmt:]
    solid_y = total_y[:,fmt:]
    Mantle_carbon = solid_y[18,:]+solid_y[20,:] + solid_y[13,:]/0.012
    Mantle_hydrogen = solid_y[23,:] + solid_y[0,:]/0.001

    graph_check_arr = np.concatenate((graph_check_arr[0:fmt],graph_check_arr[fmt]+total_time[fmt:]*0))
    #graph_check_arr = graph_check_arr[0:fmt]

    time = np.copy(total_time) 
    y_out = np.copy(total_y) #re load since messed with total_y

    if do_solid_evo ==0:
        time = time[0:fmt]
        y_out = y_out[:,0:fmt]
        graph_check_arr = graph_check_arr[0:fmt]

    #START TIME ERROR IN THIS PLOT (NOT IN USE RN BUT WORTH NOTING)
    time = (time/(365*24*60*60)-5e7)/1e6
    total_time_plot = (np.copy(total_time)/(365*24*60*60)-5e7)/1e6
    solid_time = (solid_time/(365*24*60*60)-5e7)/1e6

    fO2_bar = y_out[22,:]/1e5
    TotalP = y_out[25,:]+y_out[26,:]+y_out[27,:]+y_out[28,:]+y_out[29,:]+fO2_bar
    redox_state = np.copy(fO2_bar)

    print("TotalP min/max", np.nanmin(TotalP), np.nanmax(TotalP), "finite?", np.all(np.isfinite(TotalP)))

    for ii in range(0,len(fO2_bar)):
        if total_y[2,ii]<rp:
            redox_state[ii] = np.log10(fO2_bar[ii]) - np.log10(buffer_fO2(y_out[8,ii],TotalP[ii],'FMQ'))
        else:
            redox_state[ii] = np.nan
    print("checkpoint 1")

    f_O2_mantle_temp = np.copy(f_O2_mantle) #solid mantle
    f_O2_mantle = np.concatenate((f_O2_mantle_temp[0:fmt],f_O2_mantle_temp[fmt]+total_time[fmt:]*0))
    print("checkpoint 2")
    print("total_y shape:", total_y.shape)
    print("len(f_O2_mantle):", len(f_O2_mantle))
    print("len(TotalP):", len(TotalP))
    print("f_O2_mantle min:", np.nanmin(f_O2_mantle), "max:", np.nanmax(f_O2_mantle))
    print("TotalP min:", np.nanmin(TotalP), "max:", np.nanmax(TotalP))
    redox_state_solid = np.copy(f_O2_mantle)
    for ii in range(0,len(f_O2_mantle)):
        Solid_Mantle_mass = (4./3. * np.pi * pm * (total_y[2,ii]**3 - rc**3))
        try:
            bf = buffer_fO2(total_y[7,ii], TotalP[ii], 'FMQ')
            if (f_O2_mantle[ii] <= 0) or (bf <= 0) or (not np.isfinite(f_O2_mantle[ii])) or (not np.isfinite(bf)):
                print("bad at", ii, "fO2", f_O2_mantle[ii], "bf", bf, "T", total_y[7,ii], "P", TotalP[ii])
            redox_state_solid[ii] = np.log10(f_O2_mantle[ii]) - np.log10(bf)
        except Exception as e:
            print("crash at ii", ii, "T", total_y[7,ii], "P", TotalP[ii], "fO2", f_O2_mantle[ii])
            raise
        #if Solid_Mantle_mass>0:
            #redox_state_solid[ii] = np.log10(f_O2_mantle[ii]) - np.log10(buffer_fO2(total_y[7,ii],TotalP[ii],'FMQ'))
        #else:
            #redox_state_solid[ii] = np.nan
    print("checkpoint 3")

    time = time*1e6
    total_time_plot = total_time_plot*1e6
    solid_time = solid_time*1e6

    MoltenFe_in_FeO = np.concatenate((MoltenFe_in_FeO[0:fmt],MoltenFe_in_FeO[fmt]+0*total_time[fmt:]))
    MoltenFe_in_FeO1pt5 = np.concatenate((MoltenFe_in_FeO1pt5[0:fmt],MoltenFe_in_FeO1pt5[fmt]+0*total_time[fmt:]))
    MoltenFe_in_Fe = np.concatenate((MoltenFe_in_Fe[0:fmt],MoltenFe_in_Fe[fmt]+0*total_time[fmt:]))
    print("checkpoint 4")
    F_FeO_ar = np.concatenate((F_FeO_ar[0:fmt],F_FeO_ar[fmt]+0*total_time[fmt:]))
    F_FeO1_5_ar = np.concatenate((F_FeO1_5_ar[0:fmt],F_FeO1_5_ar[fmt]+0*total_time[fmt:]))
    F_Fe_ar = np.concatenate((F_Fe_ar[0:fmt],F_Fe_ar[fmt]+0*total_time[fmt:]))
    print("take 2: fmt", fmt, "len(total_time)", len(total_time), "total_y", total_y.shape)
    

    '''if do_solid_evo ==0:
        MoltenFe_in_FeO_before = MoltenFe_in_FeO[0:fmt]
        MoltenFe_in_FeO1pt5_before = MoltenFe_in_FeO1pt5[0:fmt] #/1.5  # n_O_Fe = (F_FeO1_5 * Mliq * 0.5*16/(1.5*16+56))/0.016.... y_out[15,:]*3*0.016*(.056*2/(.016*3) = y_out[15,:].056*2
        MoltenFe_in_Fe_before = MoltenFe_in_Fe[0:fmt]

        MoltenFe_in_FeO_before2 = F_FeO_ar[0:fmt] * 56/(56.0 + 16.0) ## (M_melt*XFeO * (56.0 + 16.0) / m_sil )*56/(56.0 + 16.0)
        MoltenFe_in_FeO1pt5_before2 = F_FeO1_5_ar[0:fmt] * 56/(56.0 + 1.5*16.0)
        MoltenFe_in_Fe_before2 = F_Fe_ar[0:fmt]
    else:
        MoltenFe_in_FeO_before = MoltenFe_in_FeO
        MoltenFe_in_FeO1pt5_before = MoltenFe_in_FeO1pt5
        MoltenFe_in_Fe_before = MoltenFe_in_Fe

        MoltenFe_in_FeO_before2 = F_FeO_ar * 56/(56.0 + 16.0) ## (M_melt*XFeO * (56.0 + 16.0) / m_sil )*56/(56.0 + 16.0)
        MoltenFe_in_FeO1pt5_before2 = F_FeO1_5_ar * 56/(56.0 + 1.5*16.0)
        MoltenFe_in_Fe_before2 = F_Fe_ar'''

    m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.)  + Total_Fe_mol_fraction_start * (56.0+16.0)
    #Simple_molten  = ((Total_Fe_mol_fraction_start-y[17])*(56)/m_sil + y[17]*(56+16)/m_sil  )*(4./3. * np.pi * pm * (rp**3 - total_y[2,:]**3))
    Simple_molten = total_y_og[17,:]  *(4./3. * np.pi * pm * (rp**3 - total_y[2,:]**3))
    #MoltenFe_in_FeO = y_out[17,:]
    print("packing outputs:", len(total_time), total_y.shape, len(redox_state), len(f_O2_mantle), len(graph_check_arr))
    output_class = Model_outputs(total_time,total_y,redox_state,redox_state_solid,f_O2_mantle,graph_check_arr,fmt,MoltenFe_in_FeO, MoltenFe_in_FeO1pt5,MoltenFe_in_Fe, F_FeO_ar, F_FeO1_5_ar, F_Fe_ar) #return outputs of forward model  
    return output_class



