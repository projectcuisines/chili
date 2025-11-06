# fe_speciation.py
# Central selector for Fe speciation parameterizations.

from numba import jit
import numpy as np

XAl2O3 = 0.022423 
XCaO = 0.0335
XNa2O = 0.0024 
XK2O = 0.0001077 
XMgO = 0.478144  
XSiO2 =  0.4034

# --- Kress implementations ---
@jit(nopython=True)
def Iron_speciation_smooth_Kress(f_O2, P, T, Total_Fe):
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    #deltaIW_critical = np.log10(f_critical) - np.log10(fO2_IW) = 2 * np.log10(1.5*Total_Fe/0.8)
    #np.log10(f_critical) = 2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)
    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
    terms1 = 0.196*np.log(f_O2) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    terms = terms1+terms2+terms3+terms4  
    #final_log_ratio =  (- 1.828 * Total_Fe + terms)
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 
    if f_O2 >f_critical*10:
        return [np.exp(- 1.828 * Total_Fe + terms),Total_Fe]
    elif f_O2 > f_critical:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        #np.exp(- 1.828 * Total_Fe + terms)*fudge
        return [np.exp((- 1.828 * Total_Fe + terms)+fudge),Total_Fe]  
    else:
        return [np.exp(-np.inf),xFeO_sil]

@jit(nopython=True)
def Iron_speciation_smooth2_Kress(f_O2, P, T, Total_Fe):
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)
    #deltaIW_critical = np.log10(f_critical) - np.log10(fO2_IW) = 2 * np.log10(1.5*Total_Fe/0.8)
    #np.log10(f_critical) = 2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)
    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW))
    terms1 = 0.196*np.log(f_O2) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    terms = terms1+terms2+terms3+terms4  
    #final_log_ratio =  (- 1.828 * Total_Fe + terms)
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 
    if f_O2 >f_critical*10:
        return [np.exp(- 1.828 * Total_Fe + terms),Total_Fe]
    elif f_O2 > f_critical*1.00000000001:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        #np.exp(- 1.828 * Total_Fe + terms)*fudge
        return [np.exp((- 1.828 * Total_Fe + terms)+fudge),Total_Fe]  
    else:
        return [np.exp(-np.inf),xFeO_sil]

# --- Sossi implementations ---
@jit(nopython=True)
def Iron_speciation_smooth_Sossi(f_O2, P, T, Total_Fe):
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)) #this remains unchanged
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 #this should also stay the same i think

    # new parameterization from Sossi et al 2020
    log_XFeO1pt5_over_XFeo = 0.252*(np.log10(f_O2) - np.log10(fO2_IW)) - 1.53
    log_XFe2O3_over_XFeO = log_XFeO1pt5_over_XFeo - np.log10(2) 

    if f_O2 >f_critical*10:
        return [10**(log_XFe2O3_over_XFeO),Total_Fe]        ### returns log_XFe2O3_over_XFeO as log10
    elif f_O2 > f_critical:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        return [10**((log_XFe2O3_over_XFeO)+fudge),Total_Fe]  ### returns log_XFe2O3_over_XFeO as log10
    else:
        return [10**(-np.inf),xFeO_sil]

@jit(nopython=True)
def Iron_speciation_smooth2_Sossi(f_O2, P, T, Total_Fe):
    [A,B,C] = [27215 ,6.57 ,0.0552]
    fO2_IW= 10**(-A/T + B + C*(P/1e5-1)/T)

    f_critical = 10**(2 * np.log10(1.5*Total_Fe/0.8) + np.log10(fO2_IW)) #this remains unchanged
    xFeO_sil = 0.8*10**(0.5*(np.log10(f_O2) - np.log10(fO2_IW)))/1.5 #this should also stay the same i think

    # new parameterization from Sossi et al 2020
    log_XFeO1pt5_over_XFeo = 0.252*(np.log10(f_O2) - np.log10(fO2_IW)) - 1.53
    log_XFe2O3_over_XFeO = log_XFeO1pt5_over_XFeo - np.log10(2)

    if f_O2 >f_critical*10:
        return [10**(log_XFe2O3_over_XFeO),Total_Fe]        ### returns log_XFe2O3_over_XFeO as log10
    elif f_O2 > f_critical*1.00000000001:     #    modified mix
        fudge = 1-(1/(np.log10(f_O2) - np.log10(f_critical)) )
        return [10**((log_XFe2O3_over_XFeO)+fudge),Total_Fe]  ### returns log_XFe2O3_over_XFeO as log10
    else:
        return [10**(-np.inf),xFeO_sil]

# --- Dispatch table and public aliases ---
_SCHEMES = {
    "Kress": (Iron_speciation_smooth_Kress,  Iron_speciation_smooth2_Kress),
    "Sossi": (Iron_speciation_smooth_Sossi,  Iron_speciation_smooth2_Sossi),
}

# Public function objects that the rest of the code calls.
# They will be rebound by set_iron_param()
Iron_speciation_smooth  = Iron_speciation_smooth_Sossi
Iron_speciation_smooth2 = Iron_speciation_smooth2_Sossi

def set_iron_param(scheme: str):
    """Select which parameterization the public names refer to."""
    try:
        f1, f2 = _SCHEMES[scheme]
    except KeyError:
        raise ValueError(f"Unknown Fe speciation scheme '{scheme}'. "
                         f"Choose from {list(_SCHEMES)}")
    # Rebind the public names (single assignment cost, very fast)
    global Iron_speciation_smooth, Iron_speciation_smooth2
    Iron_speciation_smooth  = f1
    Iron_speciation_smooth2 = f2