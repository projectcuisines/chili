# CHILI protocol paper – inputs and example test results

This directory contains the results of the example test results presented in the CHILI protocol paper. Each participating model deposits their data in a sub-folder in the repository. The data output CSV file(s) should be accompanied by a TXT file that lists (in itemized form) all major noteworthy characteristics of a code that deviate from published code descriptions. For example, changes to the code that were made to adhere to the CHILI protocol, as-of-yet unpublished updates to a code, or conversions of output data to comply with the required units, etc. 

In addition, all code folders need to contain the exact config files that were used to generate the code output and in the notes TXT file need to list how the code can be obtained. Optimally, this links to a permanent code Zenodo archive, or states a published code version on an open-source GitHub (or similar) repository. At minimum, a contact e-mail needs to be stated for how the respective code version can be obtained from one of the participating authors. All submitted data needs to be <10 MB to not overburden the repository.

## Output format

> [!IMPORTANT]  
> Each individual output file should not exceed a file size of 1 MB, and the total output size for each code cannot exceed 10 MB.

### Evolution models
Output data is saved as CSV files (```evolution-<model_name>-<planet>.csv```) in this format, with *commas as separator*:

| t(yr)     | T_surf(K) | phi(vol_frac) | p_H2O(bar)     |
|-----------|-----------|---------------|----------------|
| ...       | ...       | ...           | ...            |
| ...       | ...       | ...           | ...            |

with

- ```t(yr)```:           Time in years
- ```T_surf(K)```:       Surface temperature
- ```phi(vol_frac)```:   Mantle total volume fraction of melt
- ```p_H2O(bar)```:      Partial pressure of water in the atmosphere

Code notes should be submitted as ```evolution-<model_name>-notes.txt```. Code input/configuration files should adhere to the naming convention ```evolution-<model_name>-<planet>-config```, with the file type model-specific (e.g., ```.toml```).

For example, a complete set of protocol input/output files for code GOOEY might look like:
- ```evolution-GOOEY-earth.csv```
- ```evolution-GOOEY-trappist1b.csv```
- ```evolution-GOOEY-notes.txt```
- ```evolution-GOOEY-earth-config.toml```
- ```evolution-GOOEY-trappist1b-config.toml```

### Static models

#### Surface chemistry models
Output data is saved as CSV files (``<model_name>-surface.csv``) in this format:

| mol_name  | p_i (bar) |
|-----------|-----------|
| H2O       | ...       |
| CO2       | ...       |
| ...       | ...       |

with

- ```mol_name```  : Molecule formula (e.g., H2O)
- ```p_i (bar)``` : Surface partial pressure in bar

#### Atmospheric structure models
Output data is saved as CSV files (``<model_name>-atm.csv``) in this format:

| z (m)       | P_tot (bar) | T (K)       | p_H2O (bar) | p_CO2 (bar) | p_i (bar)   |
|-------------|-------------|-------------|-------------|-------------|-------------|
| 0           | ...         | ...         | ...         | ...         | ...         |
| ...         | ...         | ...         | ...         | ...         | ...         |
| ...         | ...         | ...         | ...         | ...         | ...         |

with

- ```z (m)```        : Height in atmosphere in meters, starting from 0
- ```P_tot (bar)```  : Total pressure at height z in bar
- ```T (K)```        : Temperature at height z in Kelvin
-  ```p_H2O (bar)``` : Partial pressure of H2O at height z in bar
-  ```p_CO2 (bar)``` : Partial pressure of CO2 at height z in bar
-  ```p_i (bar)```   : Partial pressure of species i at height z in bar, can be many


## Directory structure

Each model has its own directory in ``inputs`` and ``outputs`` to store input files and outputs.
Each model should adhere to the following structure:
```
inputs/
├── model1/
│   └── <input file 1>
│   └── <input file 2>
│   └── ...
├── model2/
│   └── <input file 1>
│   └── ...
outputs/
├── model1/
│   └── model1-earth.csv
│   └── model1-trappist1b.csv
├── model2/
│   └── model2-earth.csv
│   └── model2-trappist1b.csv
```

The output files should all be named as ``<model_name>-earth.csv`` or ``<model_name>-trappist1b.csv`` etc. All models should deposit all information necessary to recreate the protocol output data in the future in their respective ``inputs/model/`` folder.
