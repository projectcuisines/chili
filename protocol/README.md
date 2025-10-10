# CHILI protocol paper – inputs and example test results

This directory contains the results of the first protocol results developed in the first CHILI protocol paper.

## Output format

### Evolution models
Output data is saved as CSV files in this format:

| time (yr) | Tsurf (K) | pH2O (bar) | phi (vol_frac) |
|-----------|-----------|------------|----------------|
| ...       | ...       | ...        | ...            |
| ...       | ...       | ...        | ...            |

with

- ```time (yr)```: Time in years
- ```Tsurf (K)```: Surface temperature in K
- ```pH2O (bar)```: Partial pressure of water in the atmosphere
- ```phi (vol_frac)```: Mantle total volume fraction of melt

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

> [!IMPORTANT]  
> Each individual output file should not exceed a file size of 1 MB!

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
