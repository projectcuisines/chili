# Static Model vs. Evolution Model Comparison

## Overview

This directory contains the data analysis comparing the **static models** against a grid of outputs from the **evolution models**. Static models are benchmarked to evolution-model snapshots, allowing a like-for-like comparison.

See the [protocol paper](https://iopscience.iop.org/article/10.3847/PSJ/ae593b) for the full methodology used to select the comparison grid.

## Methodology

For each planet and epoch (τ) combination, two evolution-model snapshots are selected as thermal end-members:

- **Hot** — the evolution model output whose surface temperature represents the upper bound for that (planet, τ) pair
- **Cold** — the evolution model output whose surface temperature represents the lower bound for that (planet, τ) pair

Each snapshot is identified by its source evolution model, the simulation timestep at which it occurs, and the corresponding surface temperature. These snapshots define the boundary conditions used as inputs when generating the static model grid.

## Evolution Model Timesteps

### Earth

| τ | Case | Model     | Timestep (yrs)  | Surface Temp (K)  |
|---|------|-----------|-----------------|-------------------|
| 3 | Hot  | planatmo  | 1.00e+03        | 3375.62           |
| 3 | Cold | pacman    | 9.81e+02        | 2825.68           |
| 4 | Hot  | proteus   | 9.89e+03        | 3070.32           |
| 4 | Cold | pacman    | 9.99e+03        | 2051.00           |
| 5 | Hot  | neongooey | 1.03e+05        | 2194.13           |
| 5 | Cold | pacman    | 9.94e+04        | 1690.44           |

### Venus

| τ | Case | Model    | Timestep (yrs)  | Surface Temp (K)  |
|---|------|----------|-----------------|-------------------|
| 3 | Hot  | planatmo | 1.00e+03        | 3390.61           |
| 3 | Cold | pacman   | 1.02e+03        | 2815.58           |
| 4 | Hot  | proteus  | 9.87e+03        | 3099.73           |
| 4 | Cold | pacman   | 1.00e+04        | 2082.01           |
| 5 | Hot  | neongooey| 9.77e+04        | 2297.62           |
| 5 | Cold | pacman   | 1.01e+05        | 1730.23           |

### TRAPPIST-1 e

| τ | Case | Model     | Timestep (yrs)  | Surface Temp (K)  |
|---|------|-----------|-----------------|-------------------|
| 3 | Hot  | lincs     | 975.59          | 3260.05           |
| 3 | Cold | pacman    | 981.19          | 2827.88           |
| 4 | Hot  | proteus   | 9913.00         | 3032.76           |
| 4 | Cold | pacman    | 9990.19         | 2081.67           |
| 6 | Hot  | neongooey | 1007724.39      | 2728.35           |
| 6 | Cold | moai      | 996805.67       | 1625.85           |

### TRAPPIST-1 b

| τ | Case | Model     | Timestep (yrs)  | Surface Temp (K)  |
|---|------|-----------|-----------------|-------------------|
| 3 | Hot  | neongooey | 989.67          | 3411.36           |
| 3 | Cold | pacman    | 1002.80         | 2835.30           |
| 4 | Hot  | neongooey | 9800.85         | 3363.80           |
| 4 | Cold | pacman    | 10011.80        | 2245.95           |
| 6 | Hot  | neongooey | 1016695.54      | 3276.41           |
| 6 | Cold | gooey     | 1000293.29      | 2090.01           |

### TRAPPIST-1 Alpha

| τ | Case | Model   | Timestep (yrs)  | Surface Temp (K)  |
|---|------|---------|-----------------|-------------------|
| 3 | Hot  | gooey   | 1010.89         | 3533.79           |
| 3 | Cold | proteus | 958.00          | 3242.33           |
| 4 | Hot  | gooey   | 10039.90        | 3520.54           |
| 4 | Cold | proteus | 9871.00         | 2735.84           |
| 6 | Hot  | gooey   | 1021323.35      | 3518.97           |
| 6 | Cold | N.A.                                          |

> There is only one simulation usbale at τ = 6 for TRAPPIST-1 Alpha.

