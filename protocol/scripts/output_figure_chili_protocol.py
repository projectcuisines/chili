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



def three_panel_output_figure_Ts_phi_pH2O_vs_time(files,plot_temp=False,plot_tau=False,save_plot=True):
	'''
	Goal: plot outputs in a 3-panel figure: A) surface temperature (K) as a function of time (yr)
											B) melt fraction as a function of time (yr)
											C) atmospheric water partial pressure (bar) as a function of time (yr)
	Inputs: 
		- files: list of CSV files for models to be plotted, including their path
		- plot_temp: keyword to plot specific temperatures as horizontal dashed lines
		- plot_tau: keyword to plot specific times as vertical dotted lines
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
	ax_h2o.set_ylabel(r'pH2O (bar)')
	ax_h2o.set_xlabel(r'Time (yr)')
	ax_h2o.set_yscale('log')
	ax_temp.set_ylim([1000,3600])
	xlim_inf=1e1
	ax_temp.set_xlim([xlim_inf,1e9])
	ax_phi.set_ylim([0,1])
	ax_h2o.set_ylim([1e-1,1e3])
	ax_temp.text(0.98,0.98,'A',color='grey',va='top',ha='right',transform=ax_temp.transAxes,fontweight='bold',fontsize=20)
	ax_phi.text(0.98,0.98,'B',color='grey',va='top',ha='right',transform=ax_phi.transAxes,fontweight='bold',fontsize=20)
	ax_h2o.text(0.98,0.98,'C',color='grey',va='top',ha='right',transform=ax_h2o.transAxes,fontweight='bold',fontsize=20)
	
	### Upload Wong color scheme
	colors=load_wong_colormap()
	
	### Cycle through the files to plot their outputs
	for ifile,filename in enumerate(files):
		try: ### try to open CSV file with Pandas, column naming follows Protocol paper output format
			df=pd.read_csv(filename)
			times=df['t(yr)'].values
			T=df['Tsurf(K)'].values
			phi=df['phi(vol_frac)'].values
			h2o=df['pH2O(bar)'].values
		except KeyError:
			### fallback if column names are unknown or inconsistent
			df=pd.read_csv(filename,header=0)
			times=df.iloc[:,0].values
			T=df.iloc[:,1].values
			h2o=df.iloc[:,2].values
			phi=df.iloc[:,3].values
		
		### Find name of model, which has to be formatted as /path/to/file/modelname-planet.csv
		model_name=filename.split('/')[-1].split('.csv')[0].split('-')[0]
		
		### Plot quantities
		ax_temp.plot(times,T,color=colors[ifile])
		ax_phi.plot(times,phi,color=colors[ifile])
		ax_h2o.plot(times,h2o,color=colors[ifile])
		
		### Print names of models to put on plot
		ax_temp.text(0.02,0.23-0.07*ifile,'%s'%(model_name),color=colors[ifile],va='bottom',ha='left',transform=ax_temp.transAxes)
		
	### Optional plot of characteristic temperatures as horizontal dashed lines
	if plot_temp:
		color_temp='black' ### Color of temperature lines
		temps=[300,1000] ### Values to be plotted
		for temp in temps:
			### Plot lines
			ax_temp.axhline(y=temp,xmin=0,xmax=1,color=color_temp,linestyle='dashed',linewidth=1)
			### Print numerical values plotted on top of dashed line
			ax_temp.text(xlim_inf*2,temp,r'$T_s=$'+'%.0f K'%(temp),color=color_temp,va='bottom',ha='left')

	### Optional plot of characteristic temperatures as horizontal dashed lines
	if plot_tau:
		color_taus='silver' ### Color of times lines
		taus=[1e8,4e9]  ### Values to be plotted
		for itau,tau in enumerate(taus):
			### Plot lines
			ax_temp.axvline(x=tau,ymin=0,ymax=1,color=color_taus,linestyle='dotted',linewidth=1)
			### Print numerical values plotted on left of dotted line
			### Time values to be printed have to be changed manually
			if itau==0:
				ax_temp.text(tau,2400,r'$\tau_1=10^{8}$ yr',color=color_taus,rotation=90,va='center',ha='right')
			else:
				ax_temp.text(tau,2400,r'$\tau_2=4 \times 10^{9}$ yr',color=color_taus,rotation=90,va='center',ha='right')

	### Extract name of planet, which has to be formatted as /path/to/file/modelname-planet.csv
	if filename.split('/')[-1].split('.csv')[0].split('-')[-1]=='Earth':
		planet='Earth'
	else:
		planet='TRAPPIST-1b'
	
	### Put name of planet as figure title
	plt.suptitle('%s'%(planet),fontsize=14,bbox=dict(facecolor='white',edgecolor='black'))
	
	### Adjust margins of figure, tuned manually
	plt.subplots_adjust(left=0.08,bottom=0.15,right=0.98,top=0.87)
	
	if save_plot:
		### Save figure
		plt.savefig('Protocol_runs/output_%s_3panel.pdf'%(planet),dpi=720,bbox_inches='tight')



def two_panel_output_figure_pH2O_vs_meltfrac_and_meltfrac_vs_time(files,plot_temp=False,plot_tau=False,save_plot=True):
	'''
	Goal: plot outputs in a 2-panel figure: A) atmospheric partial pressure of water (bar) as a function of melt fraction (yr)
											B) melt fraction as a function of time (yr)
		  This plot makes comparison with MORE possible: MORE finds temperature as function of melt fraction,
		  with no explicit dependence on time
	Inputs: 
		- files: list of CSV files for models to be plotted, including their path
		- plot_temp: keyword to plot specific temperatures as horizontal dashed lines
		- plot_tau: keyword to plot specific times as vertical dotted lines
		- save_plot: keyword to save the plot
	Outputs: 
		- 2-panel figure
	'''
	

	### Create figure with 2 panels
	fig=plt.figure(figsize=(6,3))
	gs=gridspec.GridSpec(1,2,wspace=0.4,hspace=0.05,figure=fig)
	ax_h2o=plt.subplot(gs[0,0])
	ax_phi=plt.subplot(gs[0,1])
	
	### Set axes labels, scales and bounds
	ax_h2o.set_ylabel(r'pH$_2$O (bar)')
	ax_h2o.set_xlabel(r'Melt Fraction')
	ax_h2o.set_yscale('log')
	ax_phi.set_ylabel(r'Melt Fraction')
	ax_phi.set_xlabel(r'Time (yr)')
	ax_phi.set_xscale('log')
	xlim_inf=1e1
	ax_h2o.set_xlim([0,1])
	ax_h2o.set_ylim([1e-1,1e3])
	ax_h2o.invert_xaxis()
	ax_phi.set_xlim([xlim_inf,1e9])
	ax_phi.set_ylim([0,1])
	
	### Upload Wong color scheme
	colors=load_wong_colormap()
	
	### Cycle through the files to plot their outputs
	for ifile,filename in enumerate(files):
		try: ### try to open CSV file with Pandas, column naming follows Protocol paper output format
			df=pd.read_csv(filename)
			times=df['t(yr)'].values
			T=df['Tsurf(K)'].values
			phi=df['phi(vol_frac)'].values
			h2o=df['pH2O(bar)'].values
		except KeyError:
			### fallback if column names are unknown or inconsistent
			df=pd.read_csv(filename,header=0)
			times=df.iloc[:,0].values
			T=df.iloc[:,1].values
			h2o=df.iloc[:,2].values
			phi=df.iloc[:,3].values
		
		### Find name of model, which has to be formatted as /path/to/file/modelname-planet.csv
		model_name=filename.split('/')[-1].split('.csv')[0].split('-')[0]
		
		### Plot quantities
		ax_h2o.plot(phi,h2o,color=colors[ifile])
		ax_phi.plot(times,phi,color=colors[ifile])
	
		### Print names of models to put on plot
		ax_h2o.text(0.0,0.99-0.07*ifile,'%s'%(model_name),color=colors[ifile],va='top',ha='left',transform=ax_h2o.transAxes)
		
	### Extract name of planet, which has to be formatted as /path/to/file/modelname-planet.csv
	if filename.split('/')[-1].split('.csv')[0].split('-')[-1]=='Earth':
		planet='Earth'
	else:
		planet='TRAPPIST-1b'
	
	### Put name of planet as figure title
	plt.suptitle('%s'%(planet),fontsize=14,bbox=dict(facecolor='white',edgecolor='black'))
	
	### Adjust margins of figure, tuned manually
	plt.subplots_adjust(left=0.13,bottom=0.15,right=0.96,top=0.87)
	
	if save_plot:
		### Save figure
		plt.savefig('Protocol_runs/output_%s_meltfrac.pdf'%(planet),dpi=720)



###########################################################################
###########################################################################
###############                                           #################
###############                    MAIN                   #################
###############                                           #################
###########################################################################
###########################################################################

if __name__ == "__main__":
	
	###########################################################################
	###########################################################################
	###############                                           #################
	###############        Diagnostic plots for Earth         #################
	###############                                           #################
	###########################################################################
	###########################################################################

	### List files to be plotted for Earth
	files_earth=[
			'Protocol_runs/GOOEY/GOOEY-Earth.csv',
			'Protocol_runs/PROTEUS/PROTEUS-Earth.csv',
			'Protocol_runs/PACMAN/PACMAN-Earth.csv',
			'Protocol_runs/CAMO/CAMO-Earth.csv',
			# 'othermodel.txt',
			]

	### Plot 3-panel figure showing surface temperature, melt fraction and water partial pressure as a function of time
	three_panel_output_figure_Ts_phi_pH2O_vs_time(files_earth, ### files to plot
													plot_temp=False, ### plot characteristic temperatures?
													plot_tau=False, ### plot characteristic times?
													save_plot=True ### save the figure?
													)

	### List files to be plotted for Earth
	files_earth_meltfrac=[
			'Protocol_runs/GOOEY/GOOEY-Earth.csv',
			'Protocol_runs/PROTEUS/PROTEUS-Earth.csv',
			'Protocol_runs/PACMAN/PACMAN-Earth.csv',
			'Protocol_runs/CAMO/CAMO-Earth.csv',
			'Protocol_runs/MORE/MORE-Earth.csv',
			# 'othermodel.txt',
			]
	
	### Plot 2-panel figure showing water partial pressure as a function of melt fraction and melt fraction as a function of time
	two_panel_output_figure_pH2O_vs_meltfrac_and_meltfrac_vs_time(files_earth_meltfrac, ### files to plot
													plot_temp=False, ### plot characteristic temperatures?
													plot_tau=False, ### plot characteristic times?
													save_plot=True ### save the figure?
													)


	###########################################################################
	###########################################################################
	###############                                           #################
	###############     Diagnostic plots for TRAPPIST1-b      #################
	###############                                           #################
	###########################################################################
	###########################################################################

	### List files to be plotted for TRAPPIST1-b
	files_T1b=[
			'Protocol_runs/GOOEY/GOOEY-T1b.csv',
			'Protocol_runs/PROTEUS/PROTEUS-T1b.csv',
			'Protocol_runs/PACMAN/PACMAN-T1b.csv',
			'Protocol_runs/CAMO/CAMO-T1b.csv',
			# 'othermodel.txt',
			]
	
	### Plot 3-panel figure showing surface temperature, melt fraction and water partial pressure as a function of time
	three_panel_output_figure_Ts_phi_pH2O_vs_time(files_T1b, ### files to plot
													plot_temp=False, ### plot characteristic temperatures?
													plot_tau=False, ### plot characteristic times?
													save_plot=True ### save the figure?
													)

	
	### List files to be plotted for TRAPPIST1-b
	files_T1b_meltfrac=[
			'Protocol_runs/GOOEY/GOOEY-T1b.csv',
			'Protocol_runs/PROTEUS/PROTEUS-T1b.csv',
			'Protocol_runs/PACMAN/PACMAN-T1b.csv',
			'Protocol_runs/CAMO/CAMO-T1b.csv',
			'Protocol_runs/MORE/MORE-T1b.csv',
			# 'othermodel.txt',
			]
	
	### Plot 2-panel figure showing water partial pressure as a function of melt fraction and melt fraction as a function of time
	two_panel_output_figure_pH2O_vs_meltfrac_and_meltfrac_vs_time(files_T1b_meltfrac, ### files to plot
													plot_temp=False, ### plot characteristic temperatures?
													plot_tau=False, ### plot characteristic times?
													save_plot=True ### save the figure?
													)

	plt.show()










