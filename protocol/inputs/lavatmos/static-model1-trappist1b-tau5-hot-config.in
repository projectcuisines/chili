import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import subprocess
import shutil
import pickle
import glob
import argparse

sys.path.insert(1, input_path_LavAtmos)
from lavatmos_goot_runner import container_lavatmos



if __name__ == "__main__":

    # run at 10**5 years from PROTEUS output


    Tsurf=1870 #bar
    Psurf=315 #bar

    Cmass=4.37*10**23 #g
    Hmass=4.47*10**20 #g
    Omass=1.14*10**24 #g

    C_weight=12
    H_weight=1
    O_weight=16

    matom=1.6605402* 10**-24 #g

    nC = Cmass/matom/12
    nO = Omass/matom/16
    nH = Hmass/matom
    n_tot= nC + nH + nO

    print('carbon abundance:',nC/n_tot)
    print('hydrogen abundance:',nH/n_tot)
    print('oxygen abundance:',nO/n_tot)
    


    #lavatmos input surface pressure excludes oxygen
    p_input =  Psurf * (nC+nH)/n_tot

    print('Lavatmos input pressure:',p_input)

    abundances= {'C' : 1.0e-20,
                        'H' :Hmass/matom/n_tot,
                        'N' : 1.0e-20,
                        'S' : 1.0e-20,
                        'P' : 1.0e-20}



    parameters = {

    # General parameters
    'run_name' : 'run_CHILI_Hatmo',

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos2', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : p_input, # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
    'volatile_comp' :  abundances, # I used renormalised solar composition here
    }

   

    lavatmos_instance = container_lavatmos(parameters)
    lavatmos_instance.run_lavatmos(1870)