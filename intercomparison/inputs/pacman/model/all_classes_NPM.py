class Switch_Inputs:
  def __init__(self ,Metal_sink_switch , do_solid_evo , heating_switch,Start_time, Max_time):
    self.Metal_sink_switch = Metal_sink_switch
    self.do_solid_evo = do_solid_evo
    self.heating_switch = heating_switch
    self.Start_time = Start_time
    self.Max_time = Max_time

class Planet_inputs:
  def __init__(self, RE,ME,rc,pm,Total_Fe_mol_fraction,Planet_sep,albedoC,albedoH):
    self.RE = RE
    self.ME = ME
    self.rc = rc
    self.pm = pm
    self.Total_Fe_mol_fraction = Total_Fe_mol_fraction
    self.Planet_sep = Planet_sep
    self.albedoC = albedoC
    self.albedoH = albedoH
    
class Init_conditions:
  def __init__(self, Init_solid_H2O,Init_fluid_H2O,Init_solid_O,Init_fluid_O,Init_solid_FeO1_5,Init_solid_FeO,Init_solid_CO2,Init_fluid_CO2):
    self.Init_solid_H2O = Init_solid_H2O
    self.Init_fluid_H2O = Init_fluid_H2O
    self.Init_solid_O = Init_solid_O
    self.Init_fluid_O = Init_fluid_O     
    self.Init_solid_FeO1_5 = Init_solid_FeO1_5
    self.Init_solid_FeO = Init_solid_FeO
    self.Init_solid_CO2 = Init_solid_CO2
    self.Init_fluid_CO2 = Init_fluid_CO2

class Numerics:
  def __init__(self, total_steps,step0,step1,step2,step3,step4,tfin0,tfin1,tfin2,tfin3,tfin4):
    self.total_steps = total_steps
    self.step0 = step0
    self.step1 = step1
    self.step2 = step2
    self.step3 = step3
    self.step4 = step4
    self.tfin0 = tfin0
    self.tfin1 = tfin1
    self.tfin2 = tfin2
    self.tfin3 = tfin3
    self.tfin4 = tfin4


class Stellar_inputs:
  def __init__(self, tsat_XUV, Stellar_Mass,fsat, beta0 , epsilon ):
    self.tsat_XUV = tsat_XUV
    self.Stellar_Mass = Stellar_Mass
    self.fsat = fsat
    self.beta0 = beta0
    self.epsilon  = epsilon 
    
class MC_inputs: 
  def __init__(self, esc_a, esc_b, esc_c, esc_d,esc_e, esc_f, interiora ,interiore,Tstrat,ThermoTemp):
    self.esc_a = esc_a
    self.esc_b = esc_b
    self.esc_c = esc_c
    self.esc_d = esc_d
    self.esc_e = esc_e
    self.esc_f = esc_f
    self.interiora = interiora
    self.interiore = interiore
    self.Tstrat = Tstrat
    self.ThermoTemp = ThermoTemp


class Model_outputs:
  def __init__(self,total_time,total_y,redox_state,redox_state_solid,f_O2_mantle,graph_check_arr,fmt,MoltenFe_in_FeO, MoltenFe_in_FeO1pt5,MoltenFe_in_Fe, F_FeO_ar,F_FeO1_5_ar ,F_Fe_ar):
    self.total_time = total_time
    self.total_y = total_y
    self.redox_state = redox_state
    self.redox_state_solid = redox_state_solid     
    self.f_O2_mantle = f_O2_mantle
    self.graph_check_arr = graph_check_arr
    self.fmt = fmt
    self.MoltenFe_in_FeO = MoltenFe_in_FeO
    self.MoltenFe_in_FeO1pt5 = MoltenFe_in_FeO1pt5
    self.MoltenFe_in_Fe = MoltenFe_in_Fe
    self.F_FeO_ar = F_FeO_ar
    self.F_FeO1_5_ar = F_FeO1_5_ar
    self.F_Fe_ar = F_Fe_ar
