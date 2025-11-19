import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_organization_test_3 import organize_raw_data
import math


#############################################
#  A. SKY PLOT UTILITY
#############################################
def plot_skyplot(sat_dfs):
    """
    Create a skyplot using any satellite that has elevation/azimuth values.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    for svid, df in sat_dfs.items():
        # Drop NaNs
        df = df.dropna(subset=["elevation_deg", "azimuth_deg"])
        if df.empty:
            continue

        theta = np.deg2rad(df["azimuth_deg"])
        r = 90 - df["elevation_deg"]   # Skyplot uses zenith = 0°

        ax.plot(theta, r, '.', label=str(svid))

    ax.set_title("GNSS Skyplot", fontsize=16)
    ax.set_rlim(90, 0)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
    plt.show()



#############################################
#  B. CN0 vs TIME FOR ALL SATELLITES
#############################################
def plot_cn0_vs_time(sat_dfs):
    plt.figure(figsize=(10, 6))

    for svid, df in sat_dfs.items():
        if "Cn0DbHz" not in df.columns:
            continue

        plt.plot(df["gps_millis"], df["Cn0DbHz"], label=f"SV {svid}")

    plt.title("CN0 vs Time")
    plt.xlabel("GPS Time (ms)")
    plt.ylabel("C/N0 (dB-Hz)")
    plt.legend()
    plt.grid()
    plt.show()



#############################################
#  C. CN0 HISTOGRAM
#############################################
def plot_cn0_histogram(sat_dfs):
    plt.figure(figsize=(8, 5))

    all_cn0 = []
    for df in sat_dfs.values():
        all_cn0.extend(df["Cn0DbHz"].dropna().values)

    plt.hist(all_cn0, bins=30)
    plt.title("Distribution of CN0 Values")
    plt.xlabel("C/N0 (dB-Hz)")
    plt.ylabel("Count")
    plt.grid()
    plt.show()



#############################################
#  D. MAIN PLOTTING FUNCTION
#############################################
def plot_all(directory):
    print("Processing GNSS data...")
    organized = organize_raw_data(directory)

    txt_full = organized["txt_full"]
    sat_dfs  = organized["satellite_dataframes"]

    print("Plotting skyplot...")
    plot_skyplot(sat_dfs)

    print("Plotting CN0 vs time...")
    plot_cn0_vs_time(sat_dfs)

    print("Plotting CN0 histogram...")
    plot_cn0_histogram(sat_dfs)

    print("Done.")



#############################################
#  E. ENTRY POINT
#############################################
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python plot_gnss_data.py <YYYY_MM_DD_HH_MM_SS>")
        exit()

    directory_input = sys.argv[1]
    plot_all(directory_input)
