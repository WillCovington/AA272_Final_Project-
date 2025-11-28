import os
import pandas as pd
import gnss_lib_py as glp
from data_organization import organize_raw_data

# read in directory data
directory = "2025_11_24_13_03_46" # CHANGE ME FOR DIFFERENT DATA FILE
organized_data = organize_raw_data(directory)

# the outputs from organize_raw_data are a modified form of the data taken in from our text file as well as a dataframe with the information for each satellite observed
txt_full = organized_data["txt_full"]
sat_dfs = organized_data["satellite_dataframes"]