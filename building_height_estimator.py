import numpy as np
import gnss_lib_py as glp
import matplotlib.pyplot as plt
from data_organization import organize_raw_data
from compute_distance_to_walls import distance_to_polygon_in_heading

def wls_receiver_location(directory):
    """
    Compute receiver location using weighted-least-squares.

    Determines WLS position of receiver using data from .txt file,
    as well as .clk and .sp3 data files pulled from the internet.
    """
    # Load in .txt file data.
    filepath = f"{directory}/gnss_log_{directory}.txt"
    raw_data = glp.AndroidRawGnss(input_path=filepath,
                                filter_measurements=True,
                                measurement_filters={"sv_time_uncertainty" : 500.},
                                verbose=False)
    
    # Augment with additional data from .clk and .sp3 files.
    full_states = glp.add_sv_states(raw_data, source="precise", verbose=False)

    # Correct for clock biases of satellites.
    full_states["corr_pr_m"] = full_states["raw_pr_m"] + full_states['b_sv_m']

    # Use only GPS satellites for the computation.
    full_states = full_states.where("gnss_id",("gps"))

    wls_estimate = glp.solve_wls(full_states, max_count=40, tol = 1)
    avg_rx_lat = wls_estimate["lat_rx_wls_deg"].mean()
    avg_rx_lon = wls_estimate["lon_rx_wls_deg"].mean()
    # Note: the altitude is not important here because we are assuming the 
    # receiver and quad have the same altitude (see below).
    avg_rx_lla = (avg_rx_lat, avg_rx_lon, 40.0)
    return avg_rx_lla

def compute_height_estimate(sat_dfs, avg_rx_lla, el_max, cn0_max):
    """
    Compute average height of surrounding buildings in an urban canyon (i.e. Stanford quad).
    
    This function filters the dataframes for all satellites based on a maximum elevation angle
    and C/N0 (determined heuristically), and uses this noisy data to estimate the heights of
    surrounding buildings. This is done by placing the receiver at the local origin of an ENU
    frame in the urban canyon. The canyon is defined as a 2D polygon that surrounds the receiver.
    The algorithm then uses the azimuths in the filtered satellite data to define the heading angle
    from the reiver to the satellite and calculate the horizontal distance to the first obstruction 
    in that direction. The height of the obstruction can then be estimated using the tangent of the
    satellite's elevation angle.
    """
    # Iterate over all satellites, using only elevations below the threshold elevation
    # and C/N0 values below the threshold C/N0.
    height_est_arr = []
    az_arr = []

    for sat in sat_dfs.values():
        filtered = sat[(sat["elevation_deg"] < el_max) & (sat["Cn0DbHz"] < cn0_max)]
        if filtered.empty:
            continue
        else:
            for az, el in zip(filtered["azimuth_deg"].values, filtered["elevation_deg"].values):
                # Compute a height estimate for every az-el data point actoss all filtered dataframes.
                distance = distance_to_polygon_in_heading(polygon, avg_rx_lla, az)
                h_est = distance*np.tan(np.deg2rad(el))
                height_est_arr.append(h_est)
                az_arr.append(az)
    avg_h_est = np.mean(np.array(height_est_arr))
    return (avg_h_est, height_est_arr, az_arr)

if __name__ == "__main__":
    # Read in directory data and generate dataframes for all satellites.
    directory = "2025_11_24_12_59_11" # <-- CHANGE ME FOR DIFFERENT DATA FILE
    organized_data = organize_raw_data(directory)
    sat_dfs = organized_data["satellite_dataframes"]

    ## Initializations ##

    # Stanford main quad area dimensions in lat/lon (source: Google maps).
    N_corner = (37.42800462101945, -122.17110561760701, 40.0)
    W_corner = (37.427368467502745, -122.17132167055206, 40.0)
    E_corner = (37.427590917103906, -122.16917077508404, 40.0)
    S_corner = (37.42695582911489, -122.16938586988063, 40.0)

    # Define a polygon with vertices at these coordinates.
    polygon = [
        N_corner, E_corner, S_corner, W_corner
    ]

    # Upper thresholds for satellite C/N0 (dBHz) and elevation (deg).
    el_max = 25.0
    cn0_max = 20.0

    avg_rx_lla = wls_receiver_location(directory)
    avg_h_est, height_est_arr, az_arr = compute_height_estimate(sat_dfs, avg_rx_lla, el_max, cn0_max)
    print(f"Average height estimate for surrounding buildings: {round(avg_h_est,2)} meters")

    fig = plt.figure() 
    ax = fig.add_subplot(111)
    ax.plot(az_arr, height_est_arr, "ob")
    ax.plot(az_arr, avg_h_est*np.ones(len(az_arr)), "-r")
    ax.set_xlabel("Satellite Azimuth [deg]")
    ax.set_ylabel("Estimated building height [m]")
    ax.set_title("Estimated building height w.r.t. heading")
    plt.show()
