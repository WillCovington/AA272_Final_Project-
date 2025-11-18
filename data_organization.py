import numpy as np
import pandas as pd
import gnss_lib_py as glp
import datetime
import os
import warnings
import georinex as gr

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
    rinex_file = base + ".25o"

    txt_raw = glp.AndroidRawGnss(txt_file)
    nmea_raw = glp.Nmea(nmea_file)
    
    # the rinex navigation file output from gnss logger pro is apparently like version 4.01 and the dumbass gnss_lib rinex parser is made for version 3.05
    # so, this converter just updates it (really, it Downdates it, dumb) so it can be passed in    
    updated_rinex = convert_rinex_for_glp(rinex_file)
    rinex_raw = glp.RinexObs(updated_rinex) # current issue with rinex files is that the output from gnssloggerpro is the wrong version; skipping for now, not necessary

    # in order to get azimuth and elevation, we need the rinex navigation file, which can be found online
    nav_file = get_nav_file(directory_input)
    
    # print("txt type: " + str(type(txt_raw)))
    # print("nmea_type: " + str(type(nmea_raw)))
    # print("rinex type: " + type(rinex_raw)) # same issue as above

    txt_data = txt_raw.preprocess(txt_file)
    print(nmea_raw)

def build_satellite_dataframes(txt_preprocess_data, nmea_data):
    # this function goes through for each satellite and finds the following pieces of data associated with that satellite's id
    # [time, CN0, PR, PRR, Elevation, Azimuth]

    return True

def convert_rinex_for_glp(input_path):
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

def get_nav_file(directory_input):
    year, month, day = map(int, directory_input.split("_")[:3])
    date = datetime.date(year, month, day)
    doy = date.timetuple().tm_yday

    yy = str(year)[-2:]
    doy3 = f"{doy:03d}"

    # Expected filename and remote path
    nav_name = f"brdc{doy3}0.{yy}n"
    nav_gz = nav_name + ".gz"
    url = f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy3}/{yy}n/{nav_gz}"

    local_gz = os.path.join(directory_input, nav_gz)
    local_nav = os.path.join(directory_input, nav_name)

    # If already downloaded, just return it
    if os.path.exists(local_nav):
        return local_nav
    if os.path.exists(local_gz):
        # Unzip
        import gzip, shutil
        with gzip.open(local_gz, "rb") as f_in, open(local_nav, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return local_nav

    # Authenticated download via georinex (uses .netrc)
    gr.download(url, local_gz)

    # Unzip
    import gzip, shutil
    with gzip.open(local_gz, "rb") as f_in, open(local_nav, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return local_nav

