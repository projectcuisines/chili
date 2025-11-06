import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import copy
import pandas as pd

def load_wong_colormap():
	'''
	Goal: load colors following the colorblind friendly scheme from Wong
	Inputs: 
		- None
	Outputs: 
		- Colors
	'''
	colors=[
			'#000000',
			'#e69f00',
			'#56b4e9',
			'#009e73',
			'#f0e442',
			'#0072b2',
			'#d55e00',
			'#cc79a7',
			]
	return colors

def combine_lists(a,b):
	'''
	Goal: merge two lists into a single one, ensuring graceful merging when one/both are strings/empty
	Inputs: 
		- a,b: lists to merge
	Outputs: 
		- combined: merged lists
	'''
	combined=[]
	if a is not None:
		if isinstance(a,list):
			combined.extend(a)
		else:
			combined.append(a)
	if b is not None:
		if isinstance(b,list):
			combined.extend(b)
		else:
			combined.append(b)
	return combined


def evolutionary_output_figure_Ts_phi_pH2O_vs_time(models_evolutionary,models_static=None,
													planet='Earth',plot_static=False,time_tau=5,
													save_plot=True):
	'''
	Goal: plot outputs in a 3-panel figure: A) surface temperature (K) as a function of time (yr)
											B) melt fraction as a function of time (yr)
											C) atmospheric water partial pressure (bar) as a function of time (yr)
	Inputs: 
		- models_evolutionary: list of evolutionary models to be plotted
		- models_static: list of static models to be plotted as dots
		- planet: name of planet to plot models for
		- plot_static: keyword to plot static model included in models_static as dots
		- time_tau: time for comparison with static models, such that time=10**time_tau
		- save_plot: keyword to save the plot
	Outputs: 
		- 3-panel figure
	'''

	### Create figure with 3 panels
	fig=plt.figure(figsize=(9,3))
	gs=gridspec.GridSpec(1,3,wspace=0.4,hspace=0.05,figure=fig)
	ax_temp=plt.subplot(gs[0,0])
	ax_phi=plt.subplot(gs[0,1],sharex=ax_temp)
	ax_h2o=plt.subplot(gs[0,2],sharex=ax_temp)
	

	### Set axes labels, scales and bounds
	ax_temp.set_ylabel(r'Surface Temperature (K)')
	ax_temp.set_xlabel(r'Time (yr)')
	ax_temp.set_xscale('log')
	ax_phi.set_ylabel(r'Melt Fraction')
	ax_phi.set_xlabel(r'Time (yr)')
	ax_h2o.set_ylabel(r'pH$_2$O (bar)')
	ax_h2o.set_xlabel(r'Time (yr)')
	ax_h2o.set_yscale('log')
	ax_temp.set_ylim([1400,3600])
	xlim_inf=1e1
	ax_temp.set_xlim([xlim_inf,1e9])
	ax_phi.set_ylim([0,1])
	ax_h2o.set_ylim([4e-1,1e3])
	ax_temp.text(0.98,0.98,'A',color='grey',va='top',ha='right',transform=ax_temp.transAxes,fontweight='bold',fontsize=26)
	ax_phi.text(0.98,0.98,'B',color='grey',va='top',ha='right',transform=ax_phi.transAxes,fontweight='bold',fontsize=26)
	ax_h2o.text(0.98,0.98,'C',color='grey',va='top',ha='right',transform=ax_h2o.transAxes,fontweight='bold',fontsize=26)
	
	### Upload Wong color scheme
	colors=load_wong_colormap()
	colors=[
			colors[1], # orange
			colors[0], # black
			colors[2], # skyblue
			colors[3], # green
			colors[6], # vermillion
			colors[5], # darkblue
			colors[4], # yellow
			colors[7], # pink
			]
	
	all_models=combine_lists(models_evolutionary,models_static) ### total number of models to plot
	planet=planet.lower() ### ensure lower case for planet name
	### Check for dashes in "TRAPPIST-1b"
	if planet=='trappist-1b':
		planet='trappist1b'

	### Cycle through the evolutionary files to plot their outputs
	for ifile,model_name in enumerate(models_evolutionary):
		model_name=model_name.lower() ### ensure lower case for planet name
		if model_name=='moachi': ### ensure correct formatting for MOAChi and LavAtmos
			model_name_label='MOAChi'
		elif model_name=='lavatmos':
			model_name_label=='LavAtmos'
		else:
			model_name_label=model_name.upper()
		filename='outputs_main/%s/evolution-%s-%s-data.csv'%(model_name,model_name,planet)
		try: ### ensure naming of file follows the protocol guidelines and extract values
			df=pd.read_csv(filename)
			times=df['t(yr)'].values
			T=df['T_surf(K)'].values
			phi=df['phi(vol_frac)'].values
			h2o=df['p_H2O(bar)'].values
		except FileNotFoundError as e: ### raise error if file name is not found
			raise FileNotFoundError(
				f"File not found: '{filename}'. Please check the file path and name.\n"
				"Please ensure your file name matches the required format.\n"
				"For evolutionary models: evolution-<model_name>-<planet>-data.csv"
			) from e
		except KeyError as e: ### raise error if file does not follow naming of columns
			### Read header row to inspect actual columns
			df=pd.read_csv(filename,header=0)
			actual_cols=list(df.columns)
			required_cols=['t(yr)','T_surf(K)','phi(vol_frac)','p_H2O(bar)']
			raise ValueError(
				f"CSV column format error: Expected columns {required_cols}.\n"
				f"Found columns: {actual_cols} in file: {filename}\n"
				"Please ensure your file matches the required column names and order."
			) from e
		
		### Plot quantities
		ax_temp.plot(times,T,color=colors[ifile])
		ax_phi.plot(times,phi,color=colors[ifile])
		ax_h2o.plot(times,h2o,color=colors[ifile])
		
		### Print names of models to put on plot
		ax_temp.text(0.4,(len(all_models)-ifile)*0.085+0.4,'%s'%(model_name_label),color=colors[ifile],va='bottom',ha='left',transform=ax_temp.transAxes,fontsize=12)
	
	last_printed_name_position=(len(all_models)-ifile-1)*0.085+0.4
	### Extract name of planet for figure title
	if planet.lower()=='earth':
		planet_name='Earth'
	else:
		planet_name='TRAPPIST-1b'

	### Plot datapoint for static model comparison for MOAChi and LavAtmos
	if plot_static and models_static!=None:
		for ifile,model_name in enumerate(models_static):
			model_name=model_name.lower() ### ensure lower case for planet name
			if model_name=='moachi': ### ensure correct formatting for MOAChi and LavAtmos
				model_name_label='MOAChi'
			elif model_name=='lavatmos':
				model_name_label='LavAtmos'
			else:
				model_name_label=model_name.upper()
			filename='outputs_main/%s/static-%s-%s-tau%i-surface.csv'%(model_name,model_name,planet,time_tau)
			try: ### ensure naming of file follows the protocol guidelines and extract values
				df=pd.read_csv(filename)
				phi=df['phi(vol_frac)'].values
				T=df['T_surf(K)'].values
				p_tot=df['p_tot(bar)'].values
				h2o=df['p_H2O(bar)'].values
			except FileNotFoundError as e: ### raise error if file name is not found
				raise FileNotFoundError(
					f"File not found: '{filename}'. Please check the file path and name.\n"
					"Please ensure your file name matches the required format.\n"
					"For static models: static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv."
				) from e
			except KeyError as e: ### raise error if file does not follow naming of columns
				### Read header row to inspect actual columns
				df=pd.read_csv(filename,header=0)
				actual_cols=list(df.columns)
				required_cols=['phi(vol_frac)','T(K)','p_tot(bar)','p_H2O(bar)']
				raise ValueError(
					f"CSV column format error: Expected columns {required_cols}.\n"
					f"Found columns: {actual_cols} in file: {filename}\n"
					"Please ensure your file matches the required column names and order."
				) from e
			ax_temp.scatter(10**time_tau,T,color=colors[len(models_evolutionary)+ifile],marker='o',s=20,zorder=4)
			ax_phi.scatter(10**time_tau,phi,color=colors[len(models_evolutionary)+ifile],marker='o',s=20,zorder=4)
			ax_h2o.scatter(10**time_tau,h2o,color=colors[len(models_evolutionary)+ifile],marker='o',s=20,zorder=4)
			ax_temp.text(0.4,last_printed_name_position-(ifile)*0.085,'%s'%(model_name_label),color=colors[len(models_evolutionary)+ifile],va='bottom',ha='left',transform=ax_temp.transAxes,fontsize=12)
	elif plot_static and models_static==None:
		raise ValueError(
						"No static model has been included in models_static.\n"
						"Please ensure you specified the static models to plot.\n"
						"If you do not want to plot static models, set plot_static=False and rerun.")
	### Put name of planet as figure title
	plt.suptitle('%s'%(planet_name),fontsize=14,bbox=dict(facecolor='white',edgecolor='black'))
	
	### Adjust margins of figure, tuned manually
	plt.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.86)
	
	if save_plot:
		### Save figure
		plt.savefig('outputs/output_%s_evolutionary.pdf'%(planet_name),dpi=720,bbox_inches='tight')


def static_models_output_figure_vertical_profiles(models_static=None,is_hot=True,
													planet='Earth',time_tau=5,
													save_plot=True):
	'''
	Goal: plot static models outputs in a 2-panel figure: A) atmospheric pressure (bar) as a function of temperature (K)
											B) altitude (m) as a function of temperature (K)
	Inputs: 
		- models_static: list of static models to be plotted
		- planet: name of planet to plot models for
		- is_hot: keyword to plot hot or cold case
		- time_tau: time for comparison with static models, such that time=10**time_tau
		- save_plot: keyword to save the plot
	Outputs: 
		- 2-panel figure
	'''

	### Create figure with 2 panels
	fig=plt.figure(figsize=(6,3))
	gs=gridspec.GridSpec(1,2,wspace=0.4,hspace=0.05,figure=fig)
	ax_pT=plt.subplot(gs[0,0])
	ax_zT=plt.subplot(gs[0,1],sharex=ax_pT)
	
	### Set axes labels, scales and bounds
	ax_pT.set_ylabel(r'Pressure (bar)')
	ax_pT.set_xlabel(r'Temperature (K)')
	# ax_pT.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
	ax_pT.set_yscale('log')
	ax_zT.set_ylabel(r'Altitude (km)')
	ax_zT.set_xlabel(r'Temperature (K)')
	# ax_zT.ticklabel_format(style='sci',axis='y',scilimits=(0,0),useMathText=True)
	# ax_zT.set_xscale('log')
	xlim_inf=1e1
	ax_pT.set_xlim([440,2200])
	ax_pT.set_ylim([1e-5,300])
	ax_pT.invert_yaxis()
	# ax_zT.set_xlim([xlim_inf,1e9])
	ax_zT.set_ylim([0,310])
	ax_pT.text(0.98,0.98,'A',color='grey',va='top',ha='right',transform=ax_pT.transAxes,fontweight='bold',fontsize=20)
	ax_zT.text(0.98,0.98,'B',color='grey',va='top',ha='right',transform=ax_zT.transAxes,fontweight='bold',fontsize=20)
	
	
	### Upload Wong color scheme
	colors=load_wong_colormap()
	colors=[colors[0],colors[6],colors[5]]
	
	planet=planet.lower() ### ensure lower case for planet name
	### Check for dashes in "TRAPPIST-1b"
	if planet=='trappist-1b':
		planet='trappist1b'

	### Check if hot or cold case
	if is_hot:
		hotcold='hot'
	else:
		hotcold='cold'

	### Cycle through the files to plot their outputs
	for ifile,model_name in enumerate(models_static):
		model_name=model_name.lower() ### ensure lower case for planet name
		if model_name=='moachi': ### ensure correct formatting for MOAChi and LavAtmos
			model_name_label='MOAChi'
		elif model_name=='lavatmos':
			model_name_label='LavAtmos'
		else:
			model_name_label=model_name.upper()
		filename='outputs_main/%s/static-%s-%s-tau%i-%s-data.csv'%(model_name,model_name,planet,time_tau,hotcold)
		try: ### ensure naming of file follows the protocol guidelines and extract values
			df=pd.read_csv(filename)
			z=df['z(m)'].values
			T=df['T(K)'].values
			p=df['p_tot(bar)'].values
			# h2o=df['p_H2O(bar)'].values
		except FileNotFoundError as e:
			filename='outputs_main/%s/evolution-%s-%s-tau%i-%s-data.csv'%(model_name,model_name,planet,time_tau,model_name)
			df=pd.read_csv(filename)
			z=df['z(m)'].values
			T=df['T(K)'].values
			p=df['p_tot(bar)'].values
			print('This is an evolutionary model')
		except FileNotFoundError as e: ### raise error if file name is not found
			raise FileNotFoundError(
				f"File not found: '{filename}'. Please check the file path and name.\n"
				"Please ensure your file name matches the required format.\n"
				"For static models: static-<modelname>-<planet>-tau[3-9]-[hot,cold]-data.csv."
			) from e
		except KeyError as e: ### raise error if file does not follow naming of columns
			### Read header row to inspect actual columns
			df=pd.read_csv(filename,header=0)
			actual_cols=list(df.columns)
			required_cols=['z(m)','T(K)','p_tot(bar)','p_H2O(bar)']
			raise ValueError(
				f"CSV column format error: Expected columns {required_cols}.\n"
				f"Found columns: {actual_cols} in file: {filename}\n"
				"Please ensure your file matches the required column names and order."
			) from e
		
		### Plot quantities
		ax_pT.plot(T,p,color=colors[ifile])
		ax_zT.plot(T,z/1e3,color=colors[ifile])

		### Plot single dots for LavAtmos
		if len(T)<2:
			ax_pT.scatter(T,p,color=colors[ifile],s=20,zorder=2)
			ax_zT.scatter(T,z/1e3,color=colors[ifile],s=20,zorder=2)
	
		### Print names of models to put on plot
		ax_pT.text(0.5,(len(models_static)-ifile)*0.085+0.3,'%s'%(model_name_label),color=colors[ifile],va='bottom',ha='left',transform=ax_pT.transAxes,fontsize=12)
		
	### Extract name of planet for figure title
	if planet.lower()=='earth':
		planet_name='Earth'
	else:
		planet_name='TRAPPIST-1b'
	
	### Put name of planet as figure title
	plt.suptitle('%s'%(planet_name),fontsize=14,bbox=dict(facecolor='white',edgecolor='black'))
	
	### Adjust margins of figure, tuned manually
	plt.subplots_adjust(left=0.13,bottom=0.15,right=0.96,top=0.87)
	
	if save_plot:
		### Save figure
		plt.savefig('outputs/output_%s_static.pdf'%(planet_name),dpi=720,bbox_inches='tight')



###########################################################################
###########################################################################
###############                                           #################
###############                    MAIN                   #################
###############                                           #################
###########################################################################
###########################################################################

if __name__ == "__main__":

	### Do you want to show plots of temperature, melt fraction and pH2O as a function of time for evolutionary models?
	plot_evolutionary_models=True

	### Do you want to show plots of temperature-altitude-pressure profiles from static models?
	plot_static_models=True
	
	###########################################################################
	###########################################################################
	###############                                           #################
	###############        Diagnostic plots for Earth         #################
	###############                                           #################
	###########################################################################
	###########################################################################

	### Make plots for evolutionary model comparison
	if plot_evolutionary_models:
		### List files to be plotted for Earth
		models_evolutionary_earth=[
										'gooey',
										'proteus',
										'pacman',
										'camo',
										# '<model_name>',
									]
		models_static_earth=[
								'moachi',
								'lavatmos',
								# '<model_name>',
							]

		### Plot 3-panel figure showing surface temperature, melt fraction and water partial pressure as a function of time
		evolutionary_output_figure_Ts_phi_pH2O_vs_time(models_evolutionary=models_evolutionary_earth, # list of evolutionary models
													models_static=models_static_earth, # list of static models
													planet='Earth', # which planet to plot?
													plot_static=False, # polt the static models as dots?
													time_tau=5, # time at which static models are shown, with time=10**time_tau years
													save_plot=False # save the plot in outputs/output_<planet>_evolutionary.pdf
													)

	
	###########################################################################
	###########################################################################
	###############                                           #################
	###############     Diagnostic plots for TRAPPIST1-b      #################
	###############                                           #################
	###########################################################################
	###########################################################################

	### Make plots for evolutionary model comparison
	if plot_evolutionary_models:
		### List files to be plotted for TRAPPIST1-b
		models_evolutionary_Tb1=[
									'gooey',
									'proteus',
									'pacman',
									'camo',
									# '<model_name>',
								]
		models_static_Tb1=[
								'moachi',
								'lavatmos',
								# '<model_name>',
							]
		
		### Plot 3-panel figure showing surface temperature, melt fraction and water partial pressure as a function of time
		evolutionary_output_figure_Ts_phi_pH2O_vs_time(models_evolutionary=models_evolutionary_Tb1, # list of evolutionary models
													models_static=models_static_Tb1, # list of static models
													planet='TRAPPIST-1b', # which planet to plot?
													plot_static=True, # polt the static models as dots?
													time_tau=5, # time at which static models are shown, with time=10**time_tau years
													save_plot=False # save the plot in outputs/output_<planet>_evolutionary.pdf
													)


	###########################################################################
	###########################################################################
	###############                                           #################
	###############            Static models plots            #################
	###############                                           #################
	###########################################################################
	###########################################################################

	if plot_static_models:
		### List files to be plotted for static models
		models_static_Tb1=[
								'proteus',
								'moachi',
								# 'lavatmos',
								# '<model_name>',
							]
		
		### Plot 2-panel figure showing pressure as a function of tmeperature and altitude as a function of temperature
		static_models_output_figure_vertical_profiles(models_static=models_static_Tb1, # list of static models
														planet='TRAPPIST-1b', # which planet to plot?
														is_hot=True, # is the hot case?
														time_tau=5, # time at which static models are shown, with time=10**time_tau years
														save_plot=False # save the plot in outputs/output_<planet>_evolutionary.pdf
														)

	plt.show()










