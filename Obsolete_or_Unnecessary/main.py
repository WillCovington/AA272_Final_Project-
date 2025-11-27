import os
import pandas as pd
import gnss_lib_py as glp
from data_organization import organize_raw_data
from plot_gnss_data import plot_all
from visualize_interaction import visualize_satellites
# import cno_data
directory = "2025_11_24_13_03_46" # CHANGE ME FOR DIFFERENT DATA FILE
organized_data = organize_raw_data(directory)

# the outputs from organize_raw_data are a modified form of the data taken in from our text file as well as a dataframe with the information for each satellite observed
txt_full = organized_data["txt_full"]
sat_dfs = organized_data["satellite_dataframes"]

# call plot_all(sat_dfs) if you want to just see the plots one after another
# plot_all(sat_dfs)

# call visualize_satellites(sat_dfs) if you want to be able to fiddle with the available data
visualize_satellites(sat_dfs, joint_exclusion=True)