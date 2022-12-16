# n_screen_scan

## Run BTV_run.py

`BTV_run.py` will take beam parameters from `BTV_config.txt` 
and use `BTV_calculations.py` to take data from the specified number
of BTV detectors to calculate the emittance of the beam. 

This code can be configured to run on saved data or to take real-time
data for analysis.

Tools to plot the 2D projections of the phase space can be found in 
`plotting.py`.

Plots produced as below:
![BTV data and beam parameter calculations](data/BTV42_image.png)
![Beam envelope calculated from data, propagated through beamline](data/Propagated_beta_functions.png)