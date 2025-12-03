import matplotlib.pyplot as plt
import numpy as np
import gnss_lib_py as glp

from compute_distance_to_walls import distance_to_polygon_in_heading
from data_organization import organize_raw_data
from overlay_map import create_interactive_map

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

    wls_estimate = glp.solve_wls(full_states, max_count=40, tol = 0.5)
    avg_rx_lat = wls_estimate["lat_rx_wls_deg"].mean()
    avg_rx_lon = wls_estimate["lon_rx_wls_deg"].mean()
    avg_rx_alt = wls_estimate["alt_rx_wls_m"].mean()
    avg_rx_lla = (avg_rx_lat, avg_rx_lon, avg_rx_alt)
    return avg_rx_lla

def compute_height_estimate_static(sat_dfs, avg_rx_lla, el_min, el_max, cn0_min, cn0_max):
    """
    Compute average height of surrounding buildings in an urban canyon (i.e. Stanford quad).
    
    This function filters the dataframes for all satellites based on a minimum and maximum elevation
    angle and C/N0 (determined heuristically). This filtered data is used to estimate the heights of
    any surrounding buildings. This is done by defining the origin of a local East-North-Up (ENU)
    frame at the receiver's location in the urban canyon. The canyon "floor" is defined as a 2D
    polygon that surrounds the receiver. The algorithm then uses the azimuths in the filtered 
    satellite data to define the heading angle from the receiver to the satellite and calculates
    the horizontal distance to the first obstruction in that direction. The height of the 
    obstruction can then be estimated using the tangent of the satellite's elevation angle.
    """
    # Iterate over all satellites, using only elevations below the threshold elevation
    # and C/N0 values below the threshold C/N0.
    height_est_arr = []
    az_arr = []

    for sat in sat_dfs.values():
        filtered = sat[(sat["elevation_deg"] > el_min) & (sat["elevation_deg"] < el_max) & 
                       (sat["Cn0DbHz"] > cn0_min) & (sat["Cn0DbHz"] < cn0_max)]
        if filtered.empty:
            continue
        else:
            for az, el in zip(filtered["azimuth_deg"].values, filtered["elevation_deg"].values):
                # Compute a height estimate for every az-el data point actoss all filtered dataframes.
                distance = distance_to_polygon_in_heading(polygon_lla, avg_rx_lla, az)
                h_est = distance*np.tan(np.deg2rad(el))
                # Append each height estimate and azimuth angle to arrays for plotting.
                height_est_arr.append(h_est)
                az_arr.append(az)
    # Average over all height estimates.
    avg_h_est = np.mean(np.array(height_est_arr))
    return (avg_h_est, height_est_arr, az_arr)

def compute_height_estimate_dynamic(sat_dfs, avg_rx_lla, cn0_min, cn0_max, polygon_lla, h0=8.0, tol=1e-3, max_iter=20):
    """
    Iteratively compute heights of surrounding buildings in an urban canyon (i.e. Stanford quad).
    
    This algorithm begins by filtering sat_dfs by a minimum and maximum allowable C/N0. 
    The filtered data is used to estimate the heights of any surrounding buildings. 
    
    This is done by defining the origin of a local East-North-Up (ENU)
    frame at the receiver's location in the urban canyon. The canyon "floor" is defined as a 2D
    polygon that surrounds the receiver. The algorithm then uses the azimuths in the filtered 
    satellite data to define the heading angle from the receiver to the satellite and calculates
    the horizontal distance to the first building in that direction. An initial guess for the
    average height of all surrounding buildings is used to dynamically generate an elevation mask
    for the building based on the calculated distance. If the satellite's elevation is inside the 
    mask, the height of the obstruction is estimated using the tangent of the satellite's
    elevation angle. After all height estimates have been calculated, the global average height is
    taken as the new guess for the next iteration.

    The process repeats until the global average height converges to the specified tolerance or
    a maximum number of iterations is reached.
    """

    # Initialize parameters
    res = np.inf
    i = 0
    h_guess = h0
    while (res > tol and i < max_iter):
        height_est_arr = []
        az_arr = []
        el_max_arr = []
        el_min_arr = []
        d_arr = []
        for sat in sat_dfs.values():
            filtered = sat[(sat["Cn0DbHz"] > cn0_min) & (sat["Cn0DbHz"] < cn0_max)]
            if filtered.empty:
                continue
            else:
                for az, el in zip(filtered["azimuth_deg"].values, filtered["elevation_deg"].values):
                    # Compute a height estimate for every az-el data point actoss all filtered dataframes.
                    distance = distance_to_polygon_in_heading(polygon_lla, avg_rx_lla, az)
                    if distance is None:
                        continue
                    # determine elevation mask dynamically assuming constant building height
                    el_expected = (np.rad2deg(np.atan(h_guess/distance)) if distance != 0 else 90)
                    el_max = el_expected + 2.5
                    el_min = max(0, el_expected - 2.5)
                    el_max_arr.append(el_max)
                    el_min_arr.append(el_min)
                    d_arr.append(distance)

                    # Only estimate height for datapoints within the dynamic elevation mask.
                    if (el <= el_max and el >= el_min):
                        h_est = distance*np.tan(np.deg2rad(el))
                        # Append each height estimate and azimuth angle to arrays for plotting.
                        height_est_arr.append(h_est)
                        az_arr.append(az)
        
        # Average over all height estimates.
        avg_h_est = np.mean(np.array(height_est_arr))

        # Compute residual and update guess.
        res = abs(avg_h_est - h_guess)
        h_guess = avg_h_est

        i += 1
        print(f"iteration {i}, new height guess: {h_guess}")
    return (avg_h_est, height_est_arr, az_arr, d_arr, el_min_arr, el_max_arr)

def height_average_by_azimuth_intervals(data):
    """
    Sorts np.array of height estimates and azimuths and computes the 
    average height estimate for each 45-degree interval in azimuth.
    """
    # Sort by angle.
    data_sorted = data[data[:,0].argsort()]
    
    # Define 72° bin edges.
    bins = np.arange(0, 361, 72)
    bin_labels = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]

    # Extract angle and value arrays.
    angles = data_sorted[:,0]
    heights = data_sorted[:,1]

    # Compute averages per bin.
    averages = {}
    for (low, high) in bin_labels:
        # Boolean mask for angles in the bin [low, high).
        mask = (angles >= low) & (angles < high)
        if np.any(mask):
            averages[(low, high)] = np.mean(heights[mask])
        else:
            averages[(low, high)] = np.nan  # empty interval
    return data_sorted, averages

if __name__ == "__main__":
    # Read in directory data and generate dataframes for all satellites.
    directory = "2025_11_24_12_59_11" # <-- CHANGE ME FOR DIFFERENT DATA FILE
    organized_data = organize_raw_data(directory)
    sat_dfs = organized_data["satellite_dataframes"]

    # Get receiver location using WLS.
    avg_rx_lla = wls_receiver_location(directory)

    # Stanford main quad area dimensions in lat/lon (source: Google maps).
    # Note: We are assuming the receiver and quad have the same altitude.
    N_corner = (37.42800462101945, -122.17110561760701, avg_rx_lla[2])
    W_corner = (37.427368467502745, -122.17132167055206, avg_rx_lla[2])
    E_corner = (37.427590917103906, -122.16917077508404, avg_rx_lla[2])
    S_corner = (37.42695582911489, -122.16938586988063, avg_rx_lla[2])

    # Define a polygon with vertices at these coordinates.
    polygon_lla = [
        N_corner, E_corner, S_corner, W_corner
    ]

    quad_map = create_interactive_map(avg_rx_lla[0], avg_rx_lla[1], polygon_lla)
    file_path = "quad_map.html"
    quad_map.save(file_path)

    # Upper thresholds for satellite C/N0 (dBHz) and elevation (deg).
    cn0_min = 10.0
    cn0_max = 25.0

    # Get height estimate of surrounding buildings.
    avg_h_est, height_est_arr, az_arr, d_arr, el_min_arr, el_max_arr = compute_height_estimate_dynamic(sat_dfs, avg_rx_lla, cn0_min, cn0_max, polygon_lla)
    print(f"Converged to global average height estimate of: {round(avg_h_est,2)} meters\n")
    
    # Sort elevation mask data by distance (ascending)
    d_el_min = np.array([(d, el) for d, el in zip(d_arr, el_min_arr)])
    d_el_min_sorted = d_el_min[d_el_min[:,0].argsort()]
    d_el_max = np.array([(d, el) for d, el in zip(d_arr, el_max_arr)])
    d_el_max_sorted = d_el_max[d_el_max[:,0].argsort()]
    
    # Calculate average height for all azimuth intervals
    az_h = np.array([(az, h) for az, h in zip(az_arr, height_est_arr)])
    az_h_sorted, h_avg_by_az = height_average_by_azimuth_intervals(az_h)
    
    # Plot elevation mask bounds vs. distance to buildings
    fig1 = plt.figure() 
    ax1 = fig1.add_subplot(111)
    ax1.plot(d_el_max_sorted[:,0], d_el_max_sorted[:,1], "-r", markersize=3, label="Elevation upper bound")
    ax1.plot(d_el_min_sorted[:,0], d_el_min_sorted[:,1], "-b", markersize=3, label="Elevation lower bound")
    ax1.set_xlabel("Distance [m]", fontsize=15)
    ax1.set_ylabel("Elevation [m]", fontsize=15)
    ax1.set_title("Elevation Mask Bounds vs. Distance to Obstruction", fontsize=20)
    ax1.tick_params(axis='x', which='major', labelsize=15)
    ax1.tick_params(axis='y', which='major', labelsize=15)
    plt.legend(loc='upper right', fontsize=15) 
    plt.grid()

    # Plot height estimates as a function of satellite azimuth
    fig2 = plt.figure() 
    ax2 = fig2.add_subplot(111)
    ax2.plot(az_h_sorted[:,0], az_h_sorted[:,1], "ob", markersize=3, label="Approximate heights")
    for az_interval in h_avg_by_az:
        ax2.plot(az_interval, h_avg_by_az[az_interval]*np.ones(2), "--r", markersize=10)
        print(f"For {float(az_interval[0])}-{float(az_interval[1])} deg, averaged height is {round(h_avg_by_az[az_interval],2)} meters\n")
    ax2.plot(360, h_avg_by_az[(288, 360)], "--r", markersize=10, label="Averaged heights by heading")
    ax2.plot([0, 360], avg_h_est*np.ones(2), "--g", markersize=10, label="Global average height")
    ax2.set_xlabel("Satellite Azimuth [deg]", fontsize=15)
    ax2.set_ylabel("Estimated Building Height [m]", fontsize=15)
    ax2.set_title("Estimated Building Height vs. Heading Angle", fontsize=20)
    ax2.set_xticks(np.arange(0, 361, 72))
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=15)
    plt.xlim(0, 360)
    plt.legend(loc='upper left', fontsize=15) 
    plt.grid()
    plt.show()
    