EH2745: Computer Applications in Power Systems- Assignment 2. By Sarika Vaiyapuri Gunassekaran

Objective: This code employs Python programming and machine learning strategies to model and analyze a power system. For modeling and simulation of the power system, it makes use of the pandapower library.

Prerequisites: The code requires the following dependencies:
Python 3. x (Spyder Version 5)
Pandapower

Functionality:
1. Power Grid Model
Buses, generators, loads, and transmission lines are all added to an empty power network that is created using the powergrid_net() function. Specific parameters, such as voltage, power output, and line length, are used to define each element. Full code is written in A2_main file.
2. Time Series Power Implementation: 
Power consumption time series data are produced by the create_data_source() function for each load bus. Depending on the system condition, either "High Load" or "Low Load," it generates profiles for active power (P) and reactive power (Q). A DFData object and the created profiles are returned by the function.
Based on the profiles produced by the create_data_source() function, the create_controllers() function adds control objects to the network to change the active and reactive power of the loads.
The time series analysis results are saved using an OutputWriter object created by the create_output_writer() function. The output path and the variables that should be tracked are both specified.
3. Time series analysis:
The programme executes a number of time series analysis under various system conditions:
For the "High Load" and "Low Load" modes, high_load() and low_load() replicate the power flow, respectively.
Under "High Load" and "Low Load" stages, respectively, the functions high_load_gen3_discon() and low_load_gen3_discon() simulate the power flow with Generator 3 unplugged.
Under "High Load" and "Low Load" modes, respectively, the functions high_load_line8_discon() and low_load_line8_discon() simulate the power flow with Lines 5 to 6 disconnected.
The data is stored in excel sheets in the folders created namley, High Load, Low Load, G3 high load, G3 low load, Line8 High load and Line8 Low Load.
4. Result Analysis: 
The function reads the simulation results for each system condition from the generated Excel files. Each bus's voltage magnitude (Vpu) and angle (degrees) are read, and they are then combined into data frames for machine learning algorithms (k-means and KNN).
All the data is stored in dataset excel file and normalized one in dataset_norm_labeled excel file.

Reference: Referred to codes from the GitHub Repository.



