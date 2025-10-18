# CHILI protocol paper – inputs and example test results

This directory contains the results of the example test results presented in the CHILI protocol paper. Each participating model deposits their data in a sub-folder in the repository. The data output CSV file(s) should be accompanied by a TXT file that lists (in itemized form) all major noteworthy characteristics of a code that deviate from published code descriptions. For example, changes to the code that were made to adhere to the CHILI protocol, as-of-yet unpublished updates to a code, or conversions of output data to comply with the required units, etc. 

In addition, all code folders need to contain the exact config files that were used to generate the code output and in the notes TXT file need to list how the code can be obtained. Optimally, this links to a permanent code Zenodo archive, or states a published code version on an open-source GitHub (or similar) repository. At minimum, a contact e-mail needs to be stated for how the respective code version can be obtained from one of the participating authors. All submitted data needs to be <10 MB to not overburden the repository.

## Output format

> [!IMPORTANT]  
> Each individual output file should not exceed a file size of 1 MB, and the total output size for each code cannot exceed 10 MB.


## Directory structure

Each model has its own directory in ``inputs`` and ``outputs`` to store config files, notes, and outputs.
Each model should adhere to the following structure (with examples):
```
inputs/
├── model1/
│   └── evolution-model1-earth-config.in
│   └── evolution-model1-trappist1b-config.in
├── model2/
│   └── static-model2-trappist1b-tau5-hot-config.in
outputs/
├── model1/
│   └── evolution-model1-earth-data.csv
│   └── evolution-model1-trappist1b-data.csv
│   └── evolution-model1-notes.txt
├── model2/
│   └── static--model2-trappist1b-tau5-hot-data.csv
│   └── static-model2-notes.txt
│ ...
```

See further down for naming conventions. All models should deposit all information necessary to recreate the protocol output data in the future in their respective ``inputs/model/`` folder.


### Evolution models
Output data is saved as CSV files (```evolution-<model_name>-<planet>-data.csv```) in this format, with *commas as separator*:

| t(yr)     | T_surf(K) | phi(vol_frac) | p_H2O(bar)     |
|-----------|-----------|---------------|----------------|
| ...       | ...       | ...           | ...            |
| ...       | ...       | ...           | ...            |

with

- ```t(yr)```        : Time in years
- ```T_surf(K)```    : Surface temperature
- ```phi(vol_frac)```: Mantle total volume fraction of melt
- ```p_H2O(bar)```   : Partial pressure of water in the atmosphere

Code notes should be submitted as ```evolution-<model_name>-notes.txt``` in ```outputs/model/```. Code config files should (if possible) adhere to the naming convention ```evolution-<model_name>-<planet>-config```, with the file type model-specific (e.g., ```.toml```).

For example, a complete set of protocol input/output files for the evolutionary code GOOEY might look like:
- ```inputs/gooey/evolution-GOOEY-earth-config.in```
- ```inputs/gooey/evolution-GOOEY-trappist1b-config.in```
- ```outputs/gooey/evolution-GOOEY-earth-data.csv```
- ```outputs/gooey/evolution-GOOEY-trappist1b-data.csv```
- ```outputs/gooey/evolution-GOOEY-notes.txt```

### Static models

Output data is saved as CSV files (``static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv``) in this format, with *commas as separator*:

| z(m)        | p_tot(bar)  | T(K)        | p_H2O(bar)  | 
|-------------|-------------|-------------|-------------|
| 0           | ...         | ...         | ...         |
| ...         | ...         | ...         | ...         |
| ...         | ...         | ...         | ...         |

with

- ```z (m)```        : Height in atmosphere in meters, starting from 0
- ```p_tot(bar)```   : Total pressure at height z in bar
- ```T(K)```         : Temperature at height z in Kelvin
- ```p_H2O(bar)```   : Partial pressure of H2O at height z in bar

Submitted output files for the static models should be:
- ```static-<modelname>-<planet>-tau[3-9]-[hot,cold]-config```: any code config files necessary to recreate the output data, file type code-specific
- ```static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv```
- ```static-<modelname>-notes.txt```

For example, a complete set of protocol input/output files for the static code MOAChi might look like:
- ```inputs/moachi/static-MOAChi-trappist1b-tau5-hot-config.in```
- ```outputs/moachi/static-MOAChi-trappist1b-tau5-hot-data.csv```
- ```outputs/moachi/static-MOAChi-notes.txt```
