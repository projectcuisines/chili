#!/usr/bin/env python3
"""
run_chili_magmavol.py

CHILI intercomparison runner for the MAGMAVOL static model.
Produces a single shared compo file (temperature grid) and
CHILI-formatted static output CSVs for TRAPPIST-1e, TRAPPIST-1b,
and TRAPPIST-1alpha at specified characteristic ages (tau3, tau4, tau6).

All planets share the same volatile inventories, planet parameters
(Earth-homogenised), and MAGMAVOL compo file per CHILI protocol.
Only the hot/cold evolutionary model assignments and output filenames
differ per planet.

Usage: run from anywhere — all paths are absolute.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import glob
import logging
from scipy.interpolate import UnivariateSpline

# -------------------------------------------------------------------------
# Add MAGMAVOL to path so we can import it
# -------------------------------------------------------------------------
MAGMAVOL_DIR = '/home/spandan/Desktop/magmavol/ThermoEngine/vaporock/notebooks/MAGMAVOL'
sys.path.insert(0, MAGMAVOL_DIR)
import MAGMAVOL_2 as mag

# -------------------------------------------------------------------------
# CHILI intercomparison paths
# -------------------------------------------------------------------------
CHILI_DIR    = '/home/spandan/Desktop/chili initiative/chili/intercomparison'
OUTPUTS_DIR  = os.path.join(CHILI_DIR, 'outputs', 'magmavol')
INPUTS_DIR   = os.path.join(CHILI_DIR, 'inputs', 'magmavol')
EVOL_DIR     = os.path.join(CHILI_DIR, 'outputs')

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR,  exist_ok=True)

# -------------------------------------------------------------------------
# MAGMAVOL reservoir — use absolute MAIN_DIR + relative reservoir name
# -------------------------------------------------------------------------
mag.MAIN_DIR = MAGMAVOL_DIR
mag.DIR_FOR_RESERVOIR_OF_ATMO_SOLUTIONS = 'RESERVOIR_SOLUTIONS_MO_ATMO_EQUIL'

# -------------------------------------------------------------------------
# Planet parameters — Earth values as per CHILI Table 4
# -------------------------------------------------------------------------
R_earth = 6.371e6    # m
M_earth = 5.972e24   # kg
G       = 6.674e-11  # m^3 kg^-1 s^-2

Rp = 1.0 * R_earth
Mp = 1.0 * M_earth
g  = G * Mp / Rp**2

# -------------------------------------------------------------------------
# Fixed initial volatile inventories — CHILI Table 4
# -------------------------------------------------------------------------
massH_kg = 4.7e20    # kg
massC_kg  = 2.73e20  # kg

def mass_to_pressure_bar(mass_kg, g, Rp):
	"""Convert atmospheric elemental mass (kg) to surface pressure (bar)."""
	return (mass_kg * g / (4 * np.pi * Rp**2)) / 1e5

pH_bar = mass_to_pressure_bar(massH_kg, g, Rp)
pC_bar = mass_to_pressure_bar(massC_kg, g, Rp)

print(f"Fixed volatile inventories (CHILI Table 4, Earth parameters):")
print(f"  massH = {massH_kg:.2e} kg  ->  pH = {pH_bar:.4f} bar")
print(f"  massC = {massC_kg:.2e} kg  ->  pC = {pC_bar:.4f} bar")
print()

# -------------------------------------------------------------------------
# MAGMAVOL run parameters
# -------------------------------------------------------------------------
H    = round(pH_bar, 2)   # 90.48 bar
C    = round(pC_bar, 2)   # 52.56 bar
O    = 0.
N    = 0.
He   = 0.
P    = 0.
S    = 0.

FO2_CALCULATION = 'TARGET_FO2'
FO2_BUFFER      = 'IW'
FO2_LOG_DEV     = 4.0   # IW+4 as per CHILI protocol

magma_ocean_compo  = 'BSE'
H2O_dissolution    = False
CO2_dissolution    = False
SO2_dissolution    = False

# -------------------------------------------------------------------------
# Temperature grid — 1500 to 3800 K in 25 K steps, run high to low
# -------------------------------------------------------------------------
T_MIN    = 1500.
T_MAX    = 3800.
T_STEP   = 25.
t_arr    = np.arange(T_MIN, T_MAX + T_STEP, T_STEP)   # low to high (for saving)
t_arr_run = np.flip(t_arr)                              # high to low (for running)

# -------------------------------------------------------------------------
# Characteristic times and hot/cold model assignments per planet.
# Hot/cold models are chosen from the evolutionary ensemble as the
# highest/lowest T_surf at each tau (from CHILI workshop slides).
# All planets share the same tau_yr values; only the models differ.
# For trappist1alpha tau6, no distinct cold model exists —
# both hot and cold are set to neongooey for consistency.
# -------------------------------------------------------------------------
TAU_CONFIG = {
	'trappist1e': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'lincs',     'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'proteus',   'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'moai'},
	},
	'trappist1b': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'neongooey', 'cold_model': 'pacman'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'gooey'},
	},
	'trappist1alpha': {
		'tau3': {'tau_yr': 1e3, 'hot_model': 'gooey',     'cold_model': 'proteus'},
		'tau4': {'tau_yr': 1e4, 'hot_model': 'gooey',     'cold_model': 'proteus'},
		'tau6': {'tau_yr': 1e6, 'hot_model': 'neongooey', 'cold_model': 'neongooey'},
	},
}

VALID_PLANETS = list(TAU_CONFIG.keys())

# -------------------------------------------------------------------------
# CHILI static output columns
# -------------------------------------------------------------------------
CHILI_COLUMNS = ['z(m)', 'p_tot(bar)', 'T(K)',
	'p_H2O(bar)', 'p_CO2(bar)', 'p_CO(bar)',
	'p_H2(bar)', 'p_CH4(bar)', 'p_O2(bar)']

# Species name mapping: MAGMAVOL/FastChem name -> CHILI name
SPECIES_MAP = {
	'H2O' : 'p_H2O(bar)',
	'H2O1': 'p_H2O(bar)',
	'CO2' : 'p_CO2(bar)',
	'CO2'  : 'p_CO2(bar)',
	'C1O2': 'p_CO2(bar)',
	'CO'  : 'p_CO(bar)',
	'C1O1': 'p_CO(bar)',
	'H2'  : 'p_H2(bar)',
	'H2'  : 'p_H2(bar)',
	'CH4' : 'p_CH4(bar)',
	'C1H4': 'p_CH4(bar)',
	'O2'  : 'p_O2(bar)',
	'O2'  : 'p_O2(bar)',
}

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
log_path = os.path.join(OUTPUTS_DIR, 'run_chili_magmavol.log')
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s  %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S',
	handlers=[
		logging.FileHandler(log_path, mode='a'),
		logging.StreamHandler()
	]
)
log = logging.getLogger()

# -------------------------------------------------------------------------
# Helper: read T_surf from evolution CSV at closest time to tau_yr
# -------------------------------------------------------------------------
def get_T_surf_at_tau(model_name, planet, tau_yr):
	"""Read T_surf from evolution CSV at closest time to tau_yr."""
	csv_file = os.path.join(EVOL_DIR, model_name,
		f'evolution-{model_name}-{planet}-data.csv')
	if not os.path.exists(csv_file):
		raise FileNotFoundError(f"Evolution CSV not found: {csv_file}")
	df  = pd.read_csv(csv_file)
	idx = np.argmin(np.abs(df['t(yr)'].values - tau_yr))
	row = df.iloc[idx]
	log.info(f"  {model_name}: closest t = {row['t(yr)']:.2f} yr, "
		f"T_surf = {row['T_surf(K)']:.2f} K")
	return float(row['T_surf(K)'])


# -------------------------------------------------------------------------
# Helper: build pkl filename (mirrors generatecompofile.py convention)
# -------------------------------------------------------------------------
def build_pkl_path(Tsurf):
	"""Build the expected pkl file path for a given Tsurf."""
	initial_atmo_dict = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S}
	target_atmo_dict  = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S}

	file = f'atmo_sol_T={Tsurf}K_target_atm='
	for k, v in target_atmo_dict.items():
		file += f'_{k}={v}'
	file += '_init_atm='
	for k, v in initial_atmo_dict.items():
		file += f'_{k}={v}'
	file += f'_magma_ocean_compo={magma_ocean_compo}'
	file += f'_H2O_dissolve={H2O_dissolution}'
	file += f'_CO2_dissolve={CO2_dissolution}'
	file += f'_{FO2_CALCULATION}_{FO2_BUFFER}_{FO2_LOG_DEV}'

	return os.path.join(
		mag.MAIN_DIR,
		mag.DIR_FOR_RESERVOIR_OF_ATMO_SOLUTIONS,
		file + '.pkl'
	)


# -------------------------------------------------------------------------
# Helper: extract CHILI partial pressures from MAGMAVOL output
# -------------------------------------------------------------------------
def extract_chili_pressures(mol_pressures):
	"""
	Extract CHILI-required partial pressures from MAGMAVOL
	molecular_composition_pressures dict.
	Returns dict with CHILI column names as keys.
	"""
	out = {col: np.nan for col in CHILI_COLUMNS}
	p_tot = 0.

	for sp, p in mol_pressures.items():
		chili_key = SPECIES_MAP.get(sp, None)
		if chili_key is not None:
			# take the max if there are duplicate mappings
			if np.isnan(out[chili_key]) or p > out[chili_key]:
				out[chili_key] = p
		p_tot += p

	out['p_tot(bar)'] = p_tot
	return out


# -------------------------------------------------------------------------
# STEP 1: Run MAGMAVOL at a single test temperature (hottest at tau3)
#         and print output to check molecular_composition_pressures keys
# -------------------------------------------------------------------------
def run_single_test(planet='trappist1e', tau_key='tau3', hot_or_cold='hot'):
	"""
	Run MAGMAVOL at the surface temperature of the hot or cold evolutionary
	model for the given planet and tau, print the molecular_composition_pressures
	keys, and save a single-row CHILI CSV at z=0.

	planet     : one of 'trappist1e', 'trappist1b', 'trappist1alpha'
	tau_key    : one of 'tau3', 'tau4', 'tau6'
	hot_or_cold: 'hot' or 'cold'
	"""
	if planet not in VALID_PLANETS:
		raise ValueError(f"Unknown planet '{planet}'. Choose from: {VALID_PLANETS}")

	cfg        = TAU_CONFIG[planet][tau_key]
	tau_yr     = cfg['tau_yr']
	model_name = cfg['hot_model'] if hot_or_cold == 'hot' else cfg['cold_model']

	log.info(f"=== CHILI test run: {planet} {tau_key} ({hot_or_cold}) "
		f"(tau = {tau_yr:.0e} yr) ===")
	log.info(f"  Volatile inputs: pH = {pH_bar:.4f} bar, pC = {pC_bar:.4f} bar")
	log.info(f"  FO2: {FO2_BUFFER}+{FO2_LOG_DEV}")

	# read T_surf for selected model
	log.info(f"Reading T_surf for {hot_or_cold} model ({model_name}) at tau = {tau_yr:.0e} yr ...")
	T_surf = get_T_surf_at_tau(model_name, planet, tau_yr)
	log.info(f"  T_surf ({hot_or_cold}) = {T_surf:.2f} K")

	# run MAGMAVOL at T_surf
	log.info(f"Running MAGMAVOL at T = {T_surf:.2f} K ...")
	data = mag.atmo_compo_magma_ocean_with_dissolution_AND_initial_atmosphere_ATMO_Target_atmosphere(
		Tsurf              = T_surf,
		initial_atmo_dict  = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
		target_atmo_dict   = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
		init_sol           = -1,
		H2O_dissolution    = H2O_dissolution,
		CO2_dissolution    = CO2_dissolution,
		SO2_dissolution    = SO2_dissolution,
		chemistry_module   = 'FASTCHEM',
		magma_ocean_compo  = magma_ocean_compo,
		magma_evaporation_model = 'VAPOROCK',
		save_data_in_solutions_reservoir = True,
		FO2_CALCULATION    = FO2_CALCULATION,
		FO2_BUFFER         = FO2_BUFFER,
		FO2_LOG_DEV        = FO2_LOG_DEV,
	)

	# print molecular_composition_pressures to check keys
	log.info("molecular_composition_pressures keys and values:")
	mol_pressures = data['molecular_composition_pressures']
	for sp, p in sorted(mol_pressures.items(), key=lambda x: -x[1]):
		log.info(f"  {sp:20s} = {p:.6e} bar")

	# extract CHILI pressures
	chili_vals         = extract_chili_pressures(mol_pressures)
	chili_vals['z(m)'] = 0.
	chili_vals['T(K)'] = T_surf

	# save single-row CHILI CSV — filename follows CHILI README convention
	out_csv = os.path.join(OUTPUTS_DIR,
		f'static-magmavol-{planet}-{tau_key}-{hot_or_cold}-data.csv')
	df_out  = pd.DataFrame([chili_vals])[CHILI_COLUMNS]
	df_out.to_csv(out_csv, index=False)
	log.info(f"Saved CHILI static CSV to: {out_csv}")
	print(df_out.to_string())

	# copy this runner script to inputs folder
	import shutil
	runner_dst = os.path.join(INPUTS_DIR,
		f'static-magmavol-{planet}-{tau_key}-{hot_or_cold}-config.py')
	shutil.copy(__file__, runner_dst)
	log.info(f"Copied runner script to inputs: {runner_dst}")

	return data


# -------------------------------------------------------------------------
# STEP 2a: Interpolate missing temperatures in an existing compo .dat file
#          Can be called standalone on a pre-generated file, or automatically
#          from run_full_grid() after writing the initial compo file.
# -------------------------------------------------------------------------
def interpolate_compo_file(compo_file=None):
	"""
	Interpolate missing temperature rows in a compo .dat file and overwrite it.

	Two modes:
	  - Called with compo_file path: load the file, detect temperatures missing
	    from the expected t_arr grid, interpolate from present rows using a
	    cubic UnivariateSpline, and overwrite.
	  - Called without argument (standalone): uses the default compo filename
	    derived from the module-level parameters.

	Skips interpolation if fewer than half the expected temperatures are present.
	"""
	header  = ['# T', 'Ptot', 'H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si',
		'Ar', 'Ti', 'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr',
		'Li', 'Cs', 'Rb', 'F', 'P']

	if compo_file is None:
		compo_file = os.path.join(MAGMAVOL_DIR,
			f'compo_chili_pH={pH_bar:.2f}_pC={pC_bar:.2f}_fug={FO2_LOG_DEV}_new.dat')

	if not os.path.exists(compo_file):
		log.info(f"interpolate_compo_file: file not found: {compo_file}")
		return

	# load existing file — skip the header row (comments='')
	existing = np.loadtxt(compo_file, comments='#')
	if existing.ndim == 1:
		existing = existing[np.newaxis, :]

	T_present = existing[:, 0]
	n_cols    = existing.shape[1]

	# find which temperatures from the expected grid are missing
	missing_temps = [T for T in t_arr if not np.any(np.isclose(T_present, T))]
	n_present     = len(T_present)
	n_expected    = len(t_arr)

	log.info(f"interpolate_compo_file: {n_present}/{n_expected} temperatures present "
		f"in {os.path.basename(compo_file)}")

	if not missing_temps:
		log.info("interpolate_compo_file: no missing temperatures — nothing to do.")
		return

	# threshold: need at least n/2 present to interpolate reliably
	if n_present < n_expected / 2:
		log.info(f"interpolate_compo_file: too few rows ({n_present}/{n_expected}) "
			f"to interpolate reliably. Skipping.")
		return

	log.info(f"interpolate_compo_file: interpolating {len(missing_temps)} missing "
		f"temperature(s) using cubic UnivariateSpline over {n_present} present points.")

	# fit a spline per column (Ptot + each species) over present temperatures
	T_sorted = np.array(sorted(T_present))
	splines  = {}
	for col_idx in range(1, n_cols):   # skip col 0 (T itself)
		y = existing[np.argsort(T_present), col_idx]
		splines[col_idx] = UnivariateSpline(
			T_sorted, y, k=min(3, n_present - 1), s=0
		)

	# build interpolated rows for missing temperatures
	interp_rows = []
	for T in missing_temps:
		row    = np.zeros(n_cols)
		row[0] = T
		for col_idx in range(1, n_cols):
			row[col_idx] = float(splines[col_idx](T))
		interp_rows.append(row)
		log.info(f"  Interpolated T={T:.0f} K")

	# merge existing and interpolated rows, sort by temperature
	all_rows = np.vstack([existing, np.array(interp_rows)])
	all_rows = all_rows[np.argsort(all_rows[:, 0])]

	np.savetxt(compo_file, all_rows, delimiter=' ',
		header=' '.join(header), comments='')
	log.info(f"interpolate_compo_file: overwrote {compo_file} "
		f"({len(all_rows)} rows total).")


# -------------------------------------------------------------------------
# STEP 2: Run full temperature grid and write compo file
#         (call after verifying single test output)
# -------------------------------------------------------------------------
def run_full_grid():
	"""
	Run MAGMAVOL across the full temperature grid and write a compo .dat file.
	"""
	log.info("=== Running full temperature grid ===")
	log.info(f"  T range: {T_MIN} to {T_MAX} K in {T_STEP} K steps "
		f"({len(t_arr)} points)")
	log.info(f"  Running high to low temperature")

	for Tsurf in t_arr_run:
		pkl_path = build_pkl_path(Tsurf)
		if os.path.exists(pkl_path):
			log.info(f"  T={Tsurf:.0f} K: pkl already exists, skipping.")
			continue
		log.info(f"  T={Tsurf:.0f} K: running MAGMAVOL ...")
		mag.atmo_compo_magma_ocean_with_dissolution_AND_initial_atmosphere_ATMO_Target_atmosphere(
			Tsurf              = Tsurf,
			initial_atmo_dict  = {'H': H, 'O': O, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			target_atmo_dict   = {'H': H, 'C': C, 'N': N, 'He': He, 'P': P, 'S': S},
			init_sol           = -1,
			H2O_dissolution    = H2O_dissolution,
			CO2_dissolution    = CO2_dissolution,
			SO2_dissolution    = SO2_dissolution,
			chemistry_module   = 'FASTCHEM',
			magma_ocean_compo  = magma_ocean_compo,
			magma_evaporation_model = 'VAPOROCK',
			save_data_in_solutions_reservoir = True,
			FO2_CALCULATION    = FO2_CALCULATION,
			FO2_BUFFER         = FO2_BUFFER,
			FO2_LOG_DEV        = FO2_LOG_DEV,
		)

	# --- write compo file ---
	log.info("Writing compo file ...")
	header  = ['# T', 'Ptot', 'H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si',
		'Ar', 'Ti', 'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr',
		'Li', 'Cs', 'Rb', 'F', 'P']
	species = ['H', 'He', 'C', 'N', 'O', 'Na', 'K', 'Si', 'Ar', 'Ti',
		'V', 'S', 'Cl', 'Mg', 'Al', 'Ca', 'Fe', 'Cr', 'Li', 'Cs',
		'Rb', 'F', 'P']

	arr_list     = []
	ptot         = []
	present_temps = []

	for Tsurf in t_arr:
		pkl_path = build_pkl_path(Tsurf)
		if not os.path.exists(pkl_path):
			log.info(f"  WARNING: pkl missing for T={Tsurf:.0f} K, skipping.")
			continue
		with open(pkl_path, 'rb') as f:
			data = pickle.load(f)
			arr_list.append(data['atoms_molar_fractions'])
			ptot.append(data['Ptot_gas_bar'])
			present_temps.append(Tsurf)

	if not arr_list:
		log.info("ERROR: no pkl files found. Run the grid first.")
		return

	index    = np.array(list(arr_list[0].keys()))
	sp_temp  = np.zeros(len(species))
	values   = np.zeros((len(present_temps), len(header)))

	for num, (T, molar, p) in enumerate(zip(present_temps, arr_list, ptot)):
		val_temp = np.array(list(molar.values()))
		for sp_num, sp in enumerate(species):
			check = np.where(index == sp)[0]
			if check.size == 1:
				sp_temp[sp_num] = val_temp[check[0]]
		values[num, 2:] = sp_temp
		values[num, 1]  = p
		values[num, 0]  = T

	compo_file = os.path.join(MAGMAVOL_DIR,
		f'compo_chili_pH={pH_bar:.2f}_pC={pC_bar:.2f}_fug={FO2_LOG_DEV}_new.dat')
	np.savetxt(compo_file, values, delimiter=' ',
		header=' '.join(header), comments='')
	log.info(f"Saved compo file to: {compo_file}")

	# interpolate any missing temperatures and overwrite
	interpolate_compo_file(compo_file)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
	# -------------------------------------------------------------------------
	# Step 1: run single-temperature z=0 cases.
	# Uncomment the planet/tau/hot_or_cold combination you want to run.
	# trappist1e (already done for tau3 hot; add others as needed)
	# -------------------------------------------------------------------------
	# --- trappist1e ---
	# data = run_single_test(planet='trappist1e', tau_key='tau3', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1e', tau_key='tau3', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1e', tau_key='tau4', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1e', tau_key='tau4', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1e', tau_key='tau6', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1e', tau_key='tau6', hot_or_cold='cold')

	# --- trappist1b ---
	# data = run_single_test(planet='trappist1b', tau_key='tau3', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1b', tau_key='tau3', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1b', tau_key='tau4', hot_or_cold='hot')
	data = run_single_test(planet='trappist1b', tau_key='tau4', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1b', tau_key='tau6', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1b', tau_key='tau6', hot_or_cold='cold')

	# --- trappist1alpha ---
	# data = run_single_test(planet='trappist1alpha', tau_key='tau3', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1alpha', tau_key='tau3', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1alpha', tau_key='tau4', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1alpha', tau_key='tau4', hot_or_cold='cold')
	# data = run_single_test(planet='trappist1alpha', tau_key='tau6', hot_or_cold='hot')
	# data = run_single_test(planet='trappist1alpha', tau_key='tau6', hot_or_cold='cold')

	# -------------------------------------------------------------------------
	# Step 2: run full temperature grid and write shared compo file.
	# Only needs to be done once — compo file is shared across all planets.
	# -------------------------------------------------------------------------
	# run_full_grid()

	# -------------------------------------------------------------------------
	# Step 2a: standalone interpolation on a pre-generated compo file.
	# Useful if the grid run finished with some missing temperatures.
	# Pass an explicit path, or leave empty to use the default filename.
	# -------------------------------------------------------------------------
	# interpolate_compo_file()
	# interpolate_compo_file('/path/to/compo_chili_....dat')
