import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from lavatmos_goot_runner import container_lavatmos


def get_input(df,modelname):
    
    rows = df.loc[df["run_name"] == modelname]

    if not rows.empty:
        compvals = rows.iloc[0].to_dict()
    else:
        compvals = {}
    print(compvals)
    
    #print(compvals)
    Psurf=float(compvals['p_surf(bar)'])
    Tsurf=float(compvals['T_surf(K)'])
    Cmass=float(compvals['massC_atm(kg)']) * 1000 #g
    Hmass=float(compvals['massH_atm(kg)']) * 1000 #g
    Omass=float(compvals['massO_atm(kg)']) * 1000  #g

    #print(Cmass)



    matom=1.6605402* 10**-24 #g

    nC = Cmass/matom/12
    nO = Omass/matom/16
    nH = Hmass/matom
    n_tot= nC + nH + nO

    
    Cabun=nC/n_tot
    Habun=nH/n_tot
    Oabun=nO/n_tot



    abundances={'C' : Cabun,
                        'H' : Habun,
                        'O': Oabun,
                        'N' : 1e-20,
                        'S' : 1e-20,
                        'P' : 1e-20}
                        
    return Tsurf,Psurf,abundances
                        
    


if __name__ == "__main__":

    # the grid contains the input as given in the github input directory
    grid='input_staticpaper.csv'
    df=pd.read_csv(grid,sep=',')
    df.columns = df.columns.str.strip()

    df.columns = (
    df.columns
      .str.strip()      # remove leading/trailing spaces
      .str.replace(" ", "_", regex=False)  # replace internal spaces with underscores
)
    print(df.columns.tolist())

    df["run_name"] = (
    df["planet"].astype(str).str.strip() + "_" +
    df["τ"].astype(str).str.strip() + "_" +
    df["Case"].astype(str).str.strip() + "_" +
    df["Model"].astype(str).str.strip())
    
    
    run_names = df["run_name"].tolist()


    for modelname in run_names:
        Tsurf,Psurf,abundances=get_input(df,modelname)

        parameters = {

            # General parameters
        'run_name' : modelname,

        # Melt parameters
        'lava_comp' : 'BSE_palm',
        'silicate_abundances' : 'lavatmos3', # 'lavatmos1', 'lavatmos2', 'manual'

        # Volatile parameters
        'P_volatile' : Psurf, # bar
        'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
        'volatile_comp' :  abundances, # I used renormalised solar composition here
        'melt_fraction': 1.0
        }

        lavatmos_instance = container_lavatmos(parameters)
        lavatmos_instance.run_lavatmos(Tsurf)

        shutil.copy('/LavAtmos/FastChem/fastchem3/output/boa_chem.dat','/LavAtmos/output/'+modelname+'_chem.dat')
