import numpy as np
import pandas as pd
import gnss_lib_py as glp
import datetime
import os
import warnings

# This was the original version of the script that was going to be used to extract the data we wanted
# however, 

# this pops up with one of the gnss warnings; it's annoying and I just want to get rid of it
warnings.filterwarnings("ignore", category=FutureWarning)


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
    rinex_obs_file = base + ".25o"
    
    # rather than rename the .25n file, we just pull it directly given the regular file directory
    year, month, day, doy = get_day_of_year(directory_input)
    rinex_nav_file = f"./{directory_input}/brdc{doy}0.25n"

    txt_raw = glp.AndroidRawGnss(txt_file)
    nmea_raw = glp.Nmea(nmea_file)
    
    # the rinex navigation file output from gnss logger pro is apparently like version 4.01 and the dumbass gnss_lib rinex parser is made for version 3.05
    # so, this converter just updates it (really, it Downdates it, dumb) so it can be passed in    
    updated_rinex = convert_rinex_for_glp(rinex_obs_file)
    rinex_obs_raw = glp.RinexObs(updated_rinex)

    # NOTE in order to get the rinex navigation file (.25n), you have to go and manually download it for the day you tested
    # here are the steps for making it usable
    # 1. Figure out the date you want the .25n file for (ex. Nov 11, 2025). 
    # 2. Figure out what day of the year it is (use this: https://www.epochconverter.com/daynumbers)
    # 3. Go to the following page: cddis.nasa.gov/archive/gnss/data/daily/<YEAR>/<DAY OF YEAR>/<LAST TWO NUMBERS OF YEAR>n (don't forget the n)
    # 4. Download the file named "brdc<DAY OF YEAR>0.25n.gz"
    # 5. Unzip it and put it in the file directory it matches
    # 6. The lines above will go through and extract the actual nav file from this directory
    rinex_nav_raw = glp.RinexNav(rinex_nav_file)

    # putting all of our data into pandas dataframes (except txt_data)so they're easier to work with
    txt_data = txt_raw.preprocess(txt_file)
    nmea_data = nmea_raw.pandas_df()
    rinex_obs_data = rinex_obs_raw.pandas_df()
    rinex_nav_data = rinex_nav_raw.pandas_df()
    
    # list of satellites we observe in our measurement
    sat_list = sorted(rinex_obs_data['sv_id'].unique())
    
    # merging receiver positions
    rx_pos = nmea_data[['time', 'lat_rx_deg', 'lon_rx_deg', 'alt_rx_m']].dropna()
    txt_data = txt_data.merge(rx_pos, on="time", how="left")


def build_satellite_dataframes(txt_preprocess_data, nmea_data):
    # this function goes through for each satellite and finds the following pieces of data associated with that satellite's id
    # [time, CN0, PR, PRR, Elevation, Azimuth]

    return True

def convert_rinex_for_glp(input_path):
    # the gnss_lib_py library sucks in One regard, and that's that it isn't updated to the latest version of rinex observation files
    # so this does that
    with open(input_path, "r") as f_in:
        lines = f_in.readlines()

    base, ext = os.path.splitext(input_path)
    output_path = base + "_updated" + ext

    if not lines:
        raise ValueError("Empty RINEX file")

    first = lines[0]
    if "RINEX VERSION / TYPE" in first:
        version_field = first[:20]
        rest = first[20:]
        new_version_field = "     3.05           "
        first = new_version_field + rest

    with open(output_path, "w") as f_out:
        f_out.write(first)
        f_out.writelines(lines[1:])

    return output_path

def get_day_of_year(directory_input):
    # directory_input format: YYYY_MM_DD_HH_MM_SS
    parts = directory_input.split("_")
    year, month, day = map(int, parts[:3])

    date = datetime.date(year, month, day)
    doy = date.timetuple().tm_yday

    return year, month, day, doy
