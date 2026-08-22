MOAChi inputs for individual runs. For the actual procedure, see the .ipynb file. 
See https://github.com/projectcuisines/chili/blob/main/intercomparison/analysis_static/static_model_outputs_csv.csv
for the input parameters. Overall, other than a select few runs that are incompatible with 
MOAChi chemical network (i.e. CO-CO2 rich atmosphere), MOAChi used the reported bulk 
elemental C, H and O mass of the atmosphere and enforced thermochemical equilibrium. 
For those with bulk molar C < 0.8 H,  MOAChi used the surface partial presures reported 
for atmospheric speciation. The latter includes

- Earth $\tau = 5$ hot
- Venus $\tau = 5$ hot
- Trappist-1 alpha $\tau = 3$ hot
- Trappist-1 alpha $\tau = 4$ hot
- Trappist-1 alpha $\tau = 4$ cold
- Trappist-1 alpha $\tau = 6$ hot

For input data from neongooey and moai, the elemental abundances are calculated from 
the surface partial pressures since there seem to be discrepancies in the reported bulk masses. 

