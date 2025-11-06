PACMAN-P: Planetary Atmospheres, Crust, and MAntle evolution model for Primary atmospheres
A Coupled Magma Ocean + Atmosphere + Redox Evolution Model
Version 1.0

This set of python scripts runs the coupled magma ocean, climate, escape, and redox evolution model, as described in Krissansen-Totton et al. (2024) "The evolutionary transition from sub-Neptune to terrestrial planet does not preclude habitability". 

REQUIREMENTS: Python 3.0, including numpy, pylab, scipy, joblib, and numba modules.

HOW TO RUN CODE:
(1) Put all the python scripts in the same directory, and ensure python is working in this directory.
(2) Download climate grid from Zenodo, Ev_array_improved_corrected_CO_as_N2.npy, and place this in the same directory as python scripts. File available here: 10.5281/zenodo.13161895
(3) Check desired input settings in Input_settings.py
(4) Select planet in Main_NEW.py and check desired parameter ranges, number of iterations, and number of cores for parallelization. 
(5) IMPORTANT: If planet mass and radius is being modified from defaults, this must be updated in both the scripts Main_NEW.py and Melt_volatile_partitoning_EXP_v4.py.
(6) Run Main_NEW.py to execute Monte Carlo calculations over chosen parameter ranges.
(7) Use plotting scripts provided to inspect outputs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EXPLANATION OF CODE STRUCTURE:

%% Main_New.py
This python script runs the code. Within the script, the user has the option of selecting the planet to be modeled, as well as selecting number of model runs (1 for a single calculation, >>1 for Monte Carlo) and CPU cores. 

%% PACMAN_P.py
This script loads contains the forward model, including the system of ODEs for coupled time-evolution. The user should not need to modify anything in this script to run the code.

%% Melt_volatile_partitioning_EXP_v4.py
This script contains the functions used to calculate volatile partitioning between the magma ocean and the atmosphere. The user should only need to modify planet mass and radius in this file.

%% Input_settings.py
Here, the user may select the endmember assumption for metallic iron produced by H2 reduction of FeO (metal sinks to core or remains in mantle), the initial FeO mole fraction of the mantle, and whether graphite is permited to form in the melt.

%% MonteCarloPlotting.py and MC_multi_plot.py
For plotting Monte Carlo outputs - not an essential part of code.

%% Simple_Climate_test.py
Contains functions used to calculate surface temperature given volatile inventories, melt volume, and absorbed stellar radiation, ASR. Will call functions in the file radiative_functions_6dim_CO_as_N2 to obtain radiative transfer grid outputs.

%% radiative_functions_6dim_CO_as_N2.py
Set of functions for calling and interpolating between grid of outputs from radiative convective calculations. Functions exist for calling OLR, atmospheric water fraction, and upper atmosphere abundances for bulk constuents. Radiative-convective grid outputs are stored in the large file "Ev_array_improved_corrected_CO_as_N2" which is permanently archived on Zenodo:

%% stellar_funs.py
Loads stellar evolution parameterizations from Baraffe et al. ("Baraffe3.txt" for the sun or "Baraffe2015.txt" for different stellar masses), and returns total luminosity and XUV lumionsity as a function of time. See main text for expressions for XUV evolution relative to bolometric luminosity evolution. Trappist-1 XUV-evolution parameter from Birky et al. are loaded from the file "trappist_posterior_samples_updated.npy".

%% escape_functions.py
Includes parameterizations for XUV-limited and diffusion limited escape. The function "better_H_diffusion" calculates diffusion-limited escape of H through a fixed background gas. The function "Odert_three" calculates XUV-driven escape of H given a H-O-CO-N2 atmosphere, as described in Odert et al. (2018). Drag of O and CO are also computed. The function "find_epsilon" calculates escape efficiency, epsilon, as a function of the XUV flux.

%% escape_functions_corrected.py
Identical to escape_functions.py, but corrects a minor bug in the mixing ratio inputs which has a small effect on model outputs. Use this version for future calculations, use the original version to exactly reproduce calculations in the manuscript.

%% other_functions.py.
Contains a variety of functions, some of which are heritage from a previous model and are not used. Functions include radiogenic heat production ("qr"), mantle viscosity ("viscosity_fun"), partitioning of water between magma ocean and atmosphere ("H2O_partition_function"), partitioning of CO2 between magma ocean and atmosphere ("CO2_partition_function"), magma ocean mass calculation ("Mliq_fun"), analytic calculations for the solidification radius evolution ("rs_term_fun"), mantle adiabatic temperature profile ("adiabat"), solidus calculation ("sol_liq"), solidus radius calculation ("find_r"), and the mantle melt fraction integration ("temp_meltfrac"). 

%% all_classes_NPM.py
Defines classes used for input parameters.

%% numba_nelder_mead.py
Contains numba optimized Nelder_Mead optimization algorithm

END EXPLANATION OF CODE STRUCTURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
