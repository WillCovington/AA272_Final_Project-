import os
import pandas as pd
import gnss_lib_py as glp
from data_organization import organize_raw_data
from plot_gnss_data import plot_all
# import cno_data
directory = "2025_11_16_15_57_00" # CHANGE ME FOR DIFFERENT DATA FILE
organized_data = organize_raw_data(directory)

plot_all(directory)