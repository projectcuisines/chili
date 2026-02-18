import numpy as np
import pylab
from all_classes_NPM import *
import pylab as plt
import matplotlib.pyplot as pplt
import plot_function
import os
import shutil
import sys
import csv

def find_data(run_number,ind):
    run_num = run_number
    index = ind
    run_label = f"Data_run_{run_num}_{index}"
    ET_outputs = np.load(f"simulation_results/Model_outputs_run{run_num}.npy",allow_pickle=True)   
    ET_inputs = np.load(f"simulation_results/Model_inputs_run{run_num}.npy",allow_pickle=True)
    EO = (1.57*10**20)  #one Earth-Ocean mass of hydrogen

    # I am going to just look at ALL of the initial parameters
    Max_time = ET_inputs[index][0].Max_time
    Metal_sink_switch = ET_inputs[index][0].Metal_sink_switch
    Start_time = ET_inputs[index][0].Start_time
    do_solid_evo = ET_inputs[index][0].do_solid_evo
    heating_switch = ET_inputs[index][0].heating_switch
    ME = ET_inputs[index][1].ME
    Planet_sep = ET_inputs[index][1].Planet_sep
    RE = ET_inputs[index][1].RE
    Total_Fe_mol_fraction = ET_inputs[index][1].Total_Fe_mol_fraction
    albedoC = ET_inputs[index][1].albedoC
    albedoH = ET_inputs[index][1].albedoH
    pm = ET_inputs[index][1].pm
    rc = ET_inputs[index][1].rc
    Init_fluid_CO2 = ET_inputs[index][2].Init_fluid_CO2
    Init_fluid_H2O = ET_inputs[index][2].Init_fluid_H2O
    Init_fluid_O = ET_inputs[index][2].Init_fluid_O
    Init_solid_CO2 = ET_inputs[index][2].Init_solid_CO2
    Init_solid_FeO = ET_inputs[index][2].Init_solid_FeO
    Init_solid_FeO1_5 = ET_inputs[index][2].Init_solid_FeO1_5
    Init_solid_H2O = ET_inputs[index][2].Init_solid_H2O
    Init_solid_O = ET_inputs[index][2].Init_solid_O
    step0 = ET_inputs[index][3].step0
    step1 = ET_inputs[index][3].step1
    step2 = ET_inputs[index][3].step2
    step3 = ET_inputs[index][3].step3
    step4 = ET_inputs[index][3].step4
    tfin0 = ET_inputs[index][3].tfin0
    tfin1 = ET_inputs[index][3].tfin1
    tfin2 = ET_inputs[index][3].tfin2
    tfin3 = ET_inputs[index][3].tfin3
    tfin4 = ET_inputs[index][3].tfin4
    total_steps = ET_inputs[index][3].total_steps
    Stellar_Mass = ET_inputs[index][4].Stellar_Mass
    beta0 = ET_inputs[index][4].beta0
    epsilon = ET_inputs[index][4].epsilon
    fsat = ET_inputs[index][4].fsat
    tsat_XUV = ET_inputs[index][4].tsat_XUV
    ThermoTemp = ET_inputs[index][5].ThermoTemp
    Tstrat = ET_inputs[index][5].Tstrat
    esc_a = ET_inputs[index][5].esc_a
    esc_b = ET_inputs[index][5].esc_b
    esc_c = ET_inputs[index][5].esc_c
    esc_d = ET_inputs[index][5].esc_d
    esc_e = ET_inputs[index][5].esc_e
    esc_f = ET_inputs[index][5].esc_f
    interiora = ET_inputs[index][5].interiora
    interiore = ET_inputs[index][5].interiore


    ### Check original number of runs
    #print("number of runs in ET_outputs: ", np.shape(ET_outputs)[0])
    
    time_ar = []
    tsurf_ar = []
    OLR_ar = []
    ASR_ar = []
    water_ar = []
    CO2_ar = []
    CO_ar = []
    H2O_ar = []
    H2_ar = []
    O2_ar = []
    CH4_ar = []
    water_yn = []

    index = ind
    fmt = ET_outputs[index].fmt
    total_time = ET_outputs[index].total_time
    total_y = ET_outputs[index].total_y
    if 2>1:
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    else:
        y_out = total_y[:,0:fmt]
        time = total_time[0:fmt]

    time = (time/(365*24*60*60)-3e7)
    MO_solid = ((ET_outputs[index].total_time[ET_outputs[index].fmt])/(365*24*60*60))-3e7
    #print("y_out shape:", y_out.shape)
    #print("time shape:", time.shape)

    water = y_out[25,:] - y_out[24,:]/1e5 #bar

    for i in range(0, np.shape(ET_outputs)[0]):
         
        try:    

            if (time[i] > 0 and time[i] < 5e9):
                time_ar.append(time[i]/1e6)
                tsurf_ar.append(y_out[8,i])
                OLR_ar.append(y_out[9,i])
                ASR_ar.append(y_out[10,i])
                water_ar.append(water[i])
                CO2_ar.append(y_out[27,i])
                CO_ar.append(y_out[28,i])
                H2O_ar.append(y_out[25,i])
                H2_ar.append(y_out[26,i])
                O2_ar.append(y_out[22,i]/1e5)
                CH4_ar.append(y_out[29,i])

                if(water[i]>0.1):
                    water_tf = 1
                else: 
                    water_tf = 0
                
                water_yn.append(water_tf)
            
        except:
            pass

    if (max(water_yn)==1):
        hab = 1
    else:
        hab = 0

    # === Metadata block ===
    metadata = [
        ["# Monte Carlo Run", "Index", "AlbedoH", "Habitable (T/F)", "MO Solidification (yr)", "AlbedoC", "Init_fluid_H2O", "Init_fluid_CO2", "Init_fluid_O", 
         "beta0", "epsilon", "fsat", "tsat_XUV", "ThermoTemp", "Tstrat", "Init_solid_H2O", "Init_solid_CO2", 
         "Init_solid O", "Init_solid_FeO", "Init_solid_FeO1_5", "Total_Fe_mol_fraction", "ME", "RE", "Planet_sep", 
         "Max_time", "Metal_sink_switch", "Start_time", "do_solid_evo", "heating_switch", "pm", "rc", "Stellar_Mass", 
         "step0", "step1", "step2", "step3", "step4", "tfin0", "tfin1", "tfin2", "tfin3", "tfin4", "total_steps", 
         "esc_a", "esc_b", "esc_c", "esc_d", "esc_e", "esc_f", "interiora", "interiore"],
        [run_num, index, albedoH, hab, MO_solid, albedoC, Init_fluid_H2O, Init_fluid_CO2, Init_fluid_O, 
         beta0, epsilon, fsat, tsat_XUV, ThermoTemp, Tstrat, Init_solid_H2O, Init_solid_CO2, Init_solid_O, 
         Init_solid_FeO, Init_solid_FeO1_5, Total_Fe_mol_fraction, ME, RE, Planet_sep, 
         Max_time, Metal_sink_switch, Start_time, do_solid_evo, heating_switch, pm, rc, Stellar_Mass, 
         step0, step1, step2, step3, step4, tfin0, tfin1, tfin2, tfin3, tfin4, total_steps, 
         esc_a, esc_b, esc_c, esc_d, esc_e, esc_f, interiora, interiore],
    ]

    # === Create output directory and filename ===
    output_folder = "Data_CSVs"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{run_label}.csv"
    filepath = os.path.join(output_folder, filename)

    # === Save to CSV with metadata ===
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Write metadata rows
        writer.writerows(metadata)

        # Optional blank line between metadata and data
        writer.writerow([])

        '''writer.writerow(["Time (yr)", "Tsurf (K)", "OLR (W/m^2)", "ASR (W/m^2)", "Water (bar)", "CO2 (bar)", "H2O (bar)", "O2 (bar)", "CO (bar)", "H2 (bar)", "CH4 (bar)","Habitability T/F"])
        writer.writerows(zip(time_ar, tsurf_ar, OLR_ar, ASR_ar, water_ar, CO2_ar, H2O_ar, O2_ar, CO_ar, H2_ar, CH4_ar,water_yn))'''
    print(f"\n✅ Saved data to:\n{filepath}")

def find_CHILI_protocol_data(run_number,ind):
    run_num = run_number
    index = ind
    run_label = f"Data_run_{run_num}_{index}"
    ET_outputs = np.load(f"simulation_results/Model_outputs_run{run_num}.npy",allow_pickle=True)   
    ET_inputs = np.load(f"simulation_results/Model_inputs_run{run_num}.npy",allow_pickle=True)
    EO = (1.57*10**20)  #one Earth-Ocean mass of hydrogen

    time_ar = []
    tsurf_ar = []
    pH2O_ar = []
    melt_vol_frac_ar = []

    index = ind
    fmt = ET_outputs[index].fmt
    total_time = ET_outputs[index].total_time
    total_y = ET_outputs[index].total_y
    if 2>1:
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    else:
        y_out = total_y[:,0:fmt]
        time = total_time[0:fmt]
    start_time = ET_inputs[index][0].Start_time #in yrs i am pretty sure
    time = ((time/(365*24*60*60))-start_time)
    MO_solid = ((ET_outputs[index].total_time[ET_outputs[index].fmt])/(365*24*60*60))-start_time
    #print("y_out shape:", y_out.shape)
    #print("time shape:", time.shape)

    for i in range(min(y_out.shape[1],len(time))):

        try:    
            if (time[i] > 0 and time[i] <= MO_solid):
                time_ar.append(time[i])
                tsurf_ar.append(y_out[8,i])
                pH2O_ar.append(y_out[25,i])
                melt_vol_frac_ar.append(y_out[14,i])
            
        except:
            pass

    # === Create output directory and filename ===
    output_folder = "CHILI_CSVs"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{run_label}.csv"
    filepath = os.path.join(output_folder, filename)

    # === Save to CSV with metadata ===
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t(yr)", "T_surf(K)", "phi(vol_frac)", "p_H2O(bar)" ])
        writer.writerows(zip(time_ar, tsurf_ar, melt_vol_frac_ar, pH2O_ar))
    print(f"\n✅ Saved data to:\n{filepath}")

def find_CHILI_intercomparison_data(run_number,ind): # function to export data to csv for CHILI intercomparison
    run_num = run_number
    index = ind
    run_label = f"CHILI_data_run_{run_num}_{index}"
    ET_outputs = np.load(f"simulation_results/Model_outputs_run{run_num}.npy",allow_pickle=True)   
    ET_inputs = np.load(f"simulation_results/Model_inputs_run{run_num}.npy",allow_pickle=True)

    time_ar = []
    tsurf_ar = []
    tpot_ar = []
    flux_surf_ar = []
    flux_OLR_ar = []
    flux_ASR_ar = []
    melt_vol_frac_ar = []
    fO2_solid_ar = []
    fO2_melt_ar = []
    thick_surf_bl_ar = []
    massC_solid_ar = []
    massC_melt_ar = []
    massC_atm_ar = []
    massH_solid_ar = []
    massH_melt_ar = []
    massH_atm_ar = []
    massO_atm_ar = []
    psurf_ar = []
    pH2O_ar = []
    pCO2_ar = []
    pCO_ar = []
    pH2_ar = []
    pCH4_ar = []
    pO2_ar = []
    mmw_ar = []
    R_trans_ar = []
    R_solid_ar = []

    index = ind
    fmt = ET_outputs[index].fmt
    total_time = ET_outputs[index].total_time[0:fmt] # forcing max time to be MO solidification or max evo time
    total_y = ET_outputs[index].total_y[:,0:fmt]
    if 2>1:
        time = np.copy(total_time)
        y_out = np.copy(total_y)
    else:
        y_out = total_y[:,0:fmt]
        time = total_time[0:fmt]
    start_time = ET_inputs[index][0].Start_time #in yrs i am pretty sure
    time = ((time/(365*24*60*60))-start_time)
    redox_state = ET_outputs[index].redox_state
    redox_state_solid = ET_outputs[index].redox_state_solid

    def safe_y(y, k, i, default=0.0):
        try:
            if k >= y.shape[0] or i >= y.shape[1]:
                return default
            val = y[k, i]
            if not np.isfinite(val):
                return default
            return val
        except (IndexError, TypeError):
            return default

    def safe_1d(arr, i, default=0.0):
        try:
            if i >= len(arr):
                return default
            val = arr[i]
            if not np.isfinite(val):
                return default
            return val
        except (IndexError, TypeError):
            return default
        
    def safe_1d_log10_to_lin(arr, i, default=0.0):
        """
        arr[i] is assumed to be log10(value_in_bar). Returns value_in_bar.
        Any missing / nan / inf returns default (0.0).
        """
        try:
            if i >= len(arr):
                return default
            logv = arr[i]
            if not np.isfinite(logv):
                return default
            # convert log10(bar) -> bar
            v = 10.0**logv
            # guard against overflow -> inf
            if not np.isfinite(v):
                return default
            return v
        except (IndexError, TypeError, OverflowError):
            return default
    
    for i in range(min(y_out.shape[1],len(time))):

        time_ar.append(safe_1d(time, i))
        tsurf_ar.append(safe_y(y_out, 8, i))
        tpot_ar.append(safe_y(y_out, 7, i))

        flux_surf_ar.append(safe_y(y_out, 11, i))
        flux_OLR_ar.append(safe_y(y_out, 9, i))
        flux_ASR_ar.append(safe_y(y_out, 10, i))

        melt_vol_frac_ar.append(safe_y(y_out, 32, i))   # OK if missing

        fO2_solid_ar.append(safe_1d_log10_to_lin(redox_state_solid, i))
        fO2_melt_ar.append(safe_1d_log10_to_lin(redox_state, i))

        thick_surf_bl_ar.append(safe_y(y_out, 34, i))

        massC_solid_ar.append(safe_y(y_out, 13, i))
        massC_melt_ar.append(safe_y(y_out, 18, i) * 0.012)
        massC_atm_ar.append(safe_y(y_out, 19, i) * 0.012)

        massH_solid_ar.append(safe_y(y_out, 0, i))
        massH_melt_ar.append(safe_y(y_out, 23, i) * 0.001)
        massH_atm_ar.append(safe_y(y_out, 21, i) * 0.001)

        massO_atm_ar.append(safe_y(y_out, 16, i) * 0.016)

        psurf_ar.append(
            safe_y(y_out, 25, i) +
            safe_y(y_out, 26, i) +
            safe_y(y_out, 27, i) +
            safe_y(y_out, 28, i) +
            safe_y(y_out, 29, i) +
            safe_y(y_out, 22, i) / 1e5
        )

        pH2O_ar.append(safe_y(y_out, 25, i))
        pCO2_ar.append(safe_y(y_out, 27, i))
        pCO_ar.append(safe_y(y_out, 28, i))
        pH2_ar.append(safe_y(y_out, 26, i))
        pCH4_ar.append(safe_y(y_out, 29, i))
        pO2_ar.append(safe_y(y_out, 22, i) / 1e5)

        mmw_ar.append(safe_y(y_out, 35, i))

        R_trans_ar.append(0.0) # Hard-coded, this is not something we can return
        R_solid_ar.append(safe_y(y_out, 2, i))

    # === Create output directory and filename ===
    output_folder = "CHILI_intercomparison_CSVs"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{run_label}.csv"
    filepath = os.path.join(output_folder, filename)

    # === Save to CSV with metadata ===
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t(yr)", "T_surf(K)", "T_pot(K)", "flux_surf(W/m2)", "flux_OLR(W/m2)", 
                         "flux_ASR(W/m2)", "phi(vol_frac)", "fO2_solid(bar)", "fO2_melt(bar)", "thick_surf_bl(m)", 
                         "massC_solid(kg)", "massC_melt(kg)", "massC_atm(kg)", "massH_solid(kg)", "massH_melt(kg)", 
                         "massH_atm(kg)", "massO_atm(kg)", "p_surf(bar)", "p_H2O(bar)", "p_CO2(bar)", 
                         "p_CO(bar)", "p_H2(bar)", "p_CH4(bar)", "p_O2(bar)", "mmw(kg/mol)", 
                         "R_trans(m)", "R_solid(m)" 
                         ])
        writer.writerows(zip(time_ar, tsurf_ar, tpot_ar, flux_surf_ar, flux_OLR_ar, 
                             flux_ASR_ar, melt_vol_frac_ar, fO2_solid_ar, fO2_melt_ar, thick_surf_bl_ar, 
                             massC_solid_ar, massC_melt_ar, massC_atm_ar, massH_solid_ar, massH_melt_ar, 
                             massH_atm_ar, massO_atm_ar, psurf_ar, pH2O_ar, pCO2_ar, 
                             pCO_ar, pH2_ar, pCH4_ar, pO2_ar, mmw_ar,
                             R_trans_ar, R_solid_ar))
    print(f"\n✅ Saved data to:\n{filepath}")

find_CHILI_intercomparison_data(59,0)
#find_data(60,0)
find_CHILI_intercomparison_data(58,0)
#find_data(61,0)
find_CHILI_intercomparison_data(57,0)
find_CHILI_intercomparison_data(56,0)
find_CHILI_intercomparison_data(55,0)
find_CHILI_intercomparison_data(54,0)
find_CHILI_intercomparison_data(53,0)
find_CHILI_intercomparison_data(52,0)
find_CHILI_intercomparison_data(51,0)
find_CHILI_intercomparison_data(50,0)
find_CHILI_intercomparison_data(40,0)
find_CHILI_intercomparison_data(39,0)
