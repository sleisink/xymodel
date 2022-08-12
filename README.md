# XY model
Simulates an XY model and generate plots. With custom optimal angles, dynamical lattice.  

xy_model.py is the main file which contains the `back-end' of everything. The following files can also be executed as a script. calc.py runs a simulation and writes the results to a folder. plotprop.py makes y(x)-plots of variables. plotsnaps.py makes plots of the network and of additional parameters depending on the network topology (i.e. graph spectra). plottogether.py can overlay multiple plots for comparison. 

The script-executable files can be executed as usual as `python3 <filename.py> <args> <kwargs>`. For a brief manual: use `python3 <filename.py> -h`.
