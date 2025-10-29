# CHILI main intercomparison – input and output files

This directory will contain the main CHILI intercomparison project files. Each participating model deposits their data in a sub-folder in the repository. The data output CSV file(s) should be accompanied by a TXT file that lists (in itemized form) all major noteworthy characteristics of a code that deviate from published code descriptions. For example, changes to the code that were made to adhere to the CHILI protocol, as-of-yet unpublished updates to a code, or conversions of output data to comply with the required units, etc. 

In addition, all code folders need to contain the exact config files that were used to generate the code output and in the notes TXT file need to list how the code can be obtained. Optimally, this links to a permanent code Zenodo archive, or states a published code version on an open-source GitHub (or similar) repository. At minimum, a contact e-mail needs to be stated for how the respective code version can be obtained from one of the participating authors.

Optimally, please create a new branch with your files, create and modify folders and files as necessary in this branch, and then create a pull request to ``main``. We will then review the pull request on adherence to the requested data structures. You can find [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) instructions on how to create a pull request. If that does not work out, please simply email the organisers and we will add your data to the repository.

All code names are written in lowercase letters in this repository, irrespective of their particular capitalization. For example ``moachi`` instead of ``MOAChi``.

## Output format

> [!IMPORTANT]  
> Each individual output file should not exceed a file size of 1 MB, and the total output size for each code cannot exceed 10 MB.

## Directory structure

Each model has its own directory in ``inputs`` and ``outputs`` to store config files, notes, and outputs.
Each model should adhere to the following structure, with examples for one evolution model (``model1``) and one static model (``model2``):
```
inputs/
├── model1/
│   └── evolution-model1-earth-config.in
│   └── evolution-model1-venus-config.in
│   └── evolution-model1-earth-grid-Hlow-Clow-config.in
│   └── evolution-model1-earth-grid-Hlow-Cmid-config.in
│   └── evolution-model1-earth-grid-Hlow-Chigh-config.in
│   └── evolution-model1-earth-grid-Hmid-Clow-config.in
│   └── evolution-model1-earth-grid-Hmid-Cmid-config.in
│   └── evolution-model1-earth-grid-Hmid-Chigh-config.in
│   └── evolution-model1-earth-grid-Hhigh-Clow-config.in
│   └── evolution-model1-earth-grid-Hhigh-Cmid-config.in
│   └── evolution-model1-earth-grid-Hhigh-Chigh-config.in
│   └── evolution-model1-trappist1alpha-config.in
│   └── evolution-model1-trappist1b-config.in
│   └── evolution-model1-trappist1e-config.in
├── model2/
│   └── static-model2-earth-tau3-cold-config.in
│   └── static-model2-earth-tau3-hot-config.in
│   └── static-model2-earth-tau5-cold-config.in
│   └── static-model2-earth-tau5-hot-config.in
│   └── static-model2-earth-tau7-cold-config.in
│   └── static-model2-earth-tau7-hot-config.in
│   └── static-model2-venus-tau3-cold-config.in
│   └── static-model2-venus-tau3-hot-config.in
│   └── static-model2-venus-tau5-cold-config.in
│   └── static-model2-venus-tau5-hot-config.in
│   └── static-model2-venus-tau7-cold-config.in
│   └── static-model2-venus-tau7-hot-config.in
│   └── static-model2-trappist1alpha-tau3-cold-config.in
│   └── static-model2-trappist1alpha-tau3-hot-config.in
│   └── static-model2-trappist1alpha-tau5-cold-config.in
│   └── static-model2-trappist1alpha-tau5-hot-config.in
│   └── static-model2-trappist1alpha-tau7-cold-config.in
│   └── static-model2-trappist1alpha-tau7-hot-config.in
│   └── static-model2-trappist1b-tau3-cold-config.in
│   └── static-model2-trappist1b-tau3-hot-config.in
│   └── static-model2-trappist1b-tau5-cold-config.in
│   └── static-model2-trappist1b-tau5-hot-config.in
│   └── static-model2-trappist1b-tau7-cold-config.in
│   └── static-model2-trappist1b-tau7-hot-config.in
│   └── static-model2-trappist1e-tau3-cold-config.in
│   └── static-model2-trappist1e-tau3-hot-config.in
│   └── static-model2-trappist1e-tau5-cold-config.in
│   └── static-model2-trappist1e-tau5-hot-config.in
│   └── static-model2-trappist1e-tau7-cold-config.in
│   └── static-model2-trappist1e-tau7-hot-config.in
outputs/
├── model1/
│   └── evolution-model1-earth-data.csv
│   └── evolution-model1-venus-data.csv
│   └── evolution-model1-earth-grid-Hlow-Clow-data.csv
│   └── evolution-model1-earth-grid-Hlow-Cmid-data.csv
│   └── evolution-model1-earth-grid-Hlow-Chigh-data.csv
│   └── evolution-model1-earth-grid-Hmid-Clow-data.csv
│   └── evolution-model1-earth-grid-Hmid-Cmid-data.csv
│   └── evolution-model1-earth-grid-Hmid-Chigh-data.csv
│   └── evolution-model1-earth-grid-Hhigh-Clow-data.csv
│   └── evolution-model1-earth-grid-Hhigh-Cmid-data.csv
│   └── evolution-model1-earth-grid-Hhigh-Chigh-data.csv
│   └── evolution-model1-trappist1alpha-data.csv
│   └── evolution-model1-trappist1b-data.csv
│   └── evolution-model1-trappist1e-data.csv
│   └── evolution-model1-notes.txt
├── model2/
│   └── static-model2-earth-tau3-cold-data.csv
│   └── static-model2-earth-tau3-hot-data.csv
│   └── static-model2-earth-tau5-cold-data.csv
│   └── static-model2-earth-tau5-hot-data.csv
│   └── static-model2-earth-tau7-cold-data.csv
│   └── static-model2-earth-tau7-hot-data.csv
│   └── static-model2-venus-tau3-cold-data.csv
│   └── static-model2-venus-tau3-hot-data.csv
│   └── static-model2-venus-tau5-cold-data.csv
│   └── static-model2-venus-tau5-hot-data.csv
│   └── static-model2-venus-tau7-cold-data.csv
│   └── static-model2-venus-tau7-hot-data.csv
│   └── static-model2-trappist1alpha-tau3-cold-data.csv
│   └── static-model2-trappist1alpha-tau3-hot-data.csv
│   └── static-model2-trappist1alpha-tau5-cold-data.csv
│   └── static-model2-trappist1alpha-tau5-hot-data.csv
│   └── static-model2-trappist1alpha-tau7-cold-data.csv
│   └── static-model2-trappist1alpha-tau7-hot-data.csv
│   └── static-model2-trappist1b-tau3-cold-data.csv
│   └── static-model2-trappist1b-tau3-hot-data.csv
│   └── static-model2-trappist1b-tau5-cold-data.csv
│   └── static-model2-trappist1b-tau5-hot-data.csv
│   └── static-model2-trappist1b-tau7-cold-data.csv
│   └── static-model2-trappist1b-tau7-hot-data.csv
│   └── static-model2-trappist1e-tau3-cold-data.csv
│   └── static-model2-trappist1e-tau3-hot-data.csv
│   └── static-model2-trappist1e-tau5-cold-data.csv
│   └── static-model2-trappist1e-tau5-hot-data.csv
│   └── static-model2-trappist1e-tau7-cold-data.csv
│   └── static-model2-trappist1e-tau7-hot-data.csv
│   └── static-model2-notes.txt
│ ...
```

See further down for an explanation of naming conventions. All models should deposit all information necessary to recreate the protocol output data in the future in their respective ``inputs/model/`` folder. Additional subfolders can be create in both input and output folders if necessary, but together both folders need to stay witin the 10 MB upper file limit size. This limits in particular the upload of model-specific plots.

### Evolution models
Output data from time-evolved models is saved as CSV files (```outputs/<model_name>/evolution-<model_name>-<planet>-data.csv``` and ```outputs/<model_name>/evolution-<modelname>-earth-grid-H[low,mid,high]-C[low,mid,high]-data.csv```) with *commas as column separators*. The column headers should be:

- ```t(yr)```             Time in years
- ```T_surf(K)```         Surface temperature
- ```T_pot(K)```          Potential temperature
- ```flux_surf(W/m2)```   Net geothermal heat flux from top of mantle
- ```flux_OLR(W/m2)```    Top of atmosphere outgoing longwave radiation
- ```flux_ASR(W/m2)```    Top of atmosphere average absorbed stellar radiation
- ```phi(vol_frac)```     Mantle total volume fraction of melt
- ```fO2_solid(bar)```    Oxygen fugacity of solid mantle
- ```fO2_melt(bar)```     Oxygen fugacity of melt
- ```thick_surf_bl(m)```  Thickness of surface viscous boundary layer
- ```massC_solid(kg)```   Mass of carbon in the solid mantle
- ```massC_melt(kg)```    Mass of carbon in the melt
- ```massC_atm(kg)```     Mass of carbon in the atmosphere
- ```massH_solid(kg)```   Mass of hydrogen in the solid mantle
- ```massH_melt(kg)```    Mass of hydrogen in the melt
- ```massH_atm(kg)```     Mass of hydrogen in the atmosphere
- ```massO_atm(kg)```     Mass of oxygen in the atmosphere
- ```p_surf(bar)```       Total atmospheric surface pressure
- ```p_H2O(bar)```        Partial atmospheric pressure of H2O
- ```p_CO2(bar)```        Partial atmospheric pressure of CO2
- ```p_CO(bar)```         Partial atmospheric pressure of CO
- ```p_H2(bar)```         Partial atmospheric pressure of H2
- ```p_CH4(bar)```        Partial atmospheric pressure of CH4
- ```p_O2(bar)```         Partial atmospheric pressure of O2
- ```mmw(kg/mol)```       Mean molecular weight of the atmosphere
- ```R_trans(m)```        Transit radius of the planet in Earth radii
- ```R_solid(m)```        Radius of the rheological transition in the mantle

Code notes should be submitted as ```evolution-<model_name>-notes.txt``` in ```outputs/<model_name>/```. 

Code config files should (if possible) adhere to the naming convention ```evolution-<model_name>-<planet>-config```, with the file type model-specific (e.g., ```.toml```), in ```inputs/<model_name>/```.

### Static models

Output data from static models is also saved as CSV files (``static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv``) with *commas as column separators*. The column headers should be:

- ```z(m)```            Height in atmosphere, starting from 0 (= surface)
- ```p_tot(bar)```      Total atmospheric pressure at height z
- ```T(K)```            Temperature at height z
- ```p_H2O(bar)```      Partial atmospheric pressure of H2O at height z
- ```p_CO2(bar)```      Partial atmospheric pressure of CO2 at height z
- ```p_CO(bar)```       Partial atmospheric pressure of CO at height z
- ```p_H2(bar)```       Partial atmospheric pressure of H2 at height z
- ```p_CH4(bar)```      Partial atmospheric pressure of CH4 at height z
- ```p_O2(bar)```       Partial atmospheric pressure of O2 at height z

Submitted output files for the static models should be:
- ```inputs/<model_name>/static-<modelname>-<planet>-tau[3-9]-[hot,cold]-config```: any code config files necessary to recreate the output data, file type code-specific
- ```outputs/<model_name>/static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv```
- ```outputs/<model_name>/static-<modelname>-notes.txt```

The choice of the ages (&#120591;<sub>3</sub> - &#120591;<sub>9</sub>) will depend on the outcome of the evolutionary calculations, i.e., come later.
