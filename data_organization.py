import numpy as np
import pandas as pd
import gnss_lib_py as glp

# the idea here is to read in the file directory for a sample
# the file organization schema is to have a different panda dataframe for each satellite
# within this dataframe, there are columns for the following values:
# [time, constellation, svid, cn0, pseudorange, prrate, elevation, azimuth]
# these dataframes will all be put together in a dictionary to be easily called upon

def organize_raw_data(directory_input):
    # extracting the raw data
    # Note: the file directory formatting should be as such: ./Year_Month_Day_Hour_Minute_Second/gnss_log_Year_Month_Day_Hour_Minute_Second.txt/nmea/25o
    # this is how the files are automatically generated, so we just have to make the correct corresponding directory
    base = f"./{directory_input}/gnss_log_{directory_input}"
    txt_file   = base + ".txt"
    nmea_file  = base + ".nmea"
    rinex_file = base + ".25o"

    txt_raw = glp.AndroidRawGnss(txt_file)
    nmea_raw = glp.Nmea(nmea_file)
    rinex_raw = glp.RinexObs(rinex_file) # current issue with rinex files is that the output from gnssloggerpro is the wrong version; skipping for now, not necessary

    # print("txt type: " + str(type(txt_raw)))
    print("nmea_type: " + str(type(nmea_raw)))
    # print("rinex type: " + type(rinex_raw)) # same issue as above

    txt_data = txt_raw.preprocess(txt_file)
    print(nmea_raw)

def build_satellite_dataframes(txt_preprocess_data, nmea_data):
    # this function goes through for each satellite and finds the following pieces of data associated with that satellite's id
    # [time, CN0, PR, PRR, Elevation, Azimuth]

    return True
