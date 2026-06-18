This folder contains melting curve data for the available models. These are 
tabulated as whitespace-separated ASCII files, with file names prefixed by the 
corresponding model name. 

The dependent variables are solidus and liquidus temperatures. For models which
dynamically calculate melting properties (e.g. dependent on H2O content in the 
mantle), representative dry melting curves for Earth are used.

The independent variables are either pressure, depth, or radius. These are 
mapped using the data in `earth_adiabat.dat`. 

The columns of each file are then:
`pressure[Pa] solidus[K] liquidus[K]`

Or, in the case of depth being an independent variable
`depth[m] solidus[K] liquidus[K]`

Or, in the case of radius being an independent variable
`radius[m] solidus[K] liquidus[K]`

The data in `earth_adiabat.dat` were obtained from Katsura (2022).
```bibtex
@article{katsura_2022,
	author = {Katsura, Tomoo},
	title = {{A Revised Adiabatic Temperature Profile for the Mantle}},
	journal = {J. Geophys. Res. Solid Earth},
	volume = {127},
	number = {2},
	pages = {e2021JB023562},
	year = {2022},
	month = feb,
	issn = {2169-9313},
	publisher = {John Wiley {\&} Sons, Ltd},
	doi = {10.1029/2021JB023562}
}
```
