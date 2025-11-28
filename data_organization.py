import numpy as np
import pandas as pd
import gnss_lib_py as glp
import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------
# Utility: Convert RINEX v4 → v3.05
# ------------------------------
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


# ------------------------------
# Utility: Day of year
# ------------------------------
def get_day_of_year(directory_input):
    year, month, day = map(int, directory_input.split("_")[:3])
    date = datetime.date(year, month, day)
    doy = date.timetuple().tm_yday
    return year, month, day, doy


# ------------------------------
# Add synthetic time axis to NMEA
# ------------------------------
def add_time_to_nmea(nmea_df, txt_df):
    if nmea_df.empty:
        return nmea_df

    nmea = nmea_df.copy()

    if "gps_millis" not in txt_df.columns:
        raise ValueError("txt_data must contain gps_millis!")

    t0 = txt_df["gps_millis"].min()
    t1 = txt_df["gps_millis"].max()
    n = len(nmea)

    nmea["gps_millis"] = np.linspace(t0, t1, n)
    return nmea

def add_gps_millis_to_txt(txt_df):
    d = txt_df.copy()

    # Must have these fields
    required = ["TimeNanos", "FullBiasNanos", "BiasNanos"]
    for r in required:
        if r not in d.columns:
            raise ValueError(f"TXT data missing required time field: {r}")

    # Compute GPS time in nanoseconds
    gpsTimeNanos = d["TimeNanos"] - (d["FullBiasNanos"] + d["BiasNanos"])

    # Convert to milliseconds
    d["gps_millis"] = gpsTimeNanos * 1e-6

    return d


# ------------------------------
# Convert receiver LLA → ECEF
# ------------------------------
def lla_to_ecef(lat_deg, lon_deg, alt_m):
    # a = 6378137.0
    # e2 = 6.69437999014e-3

    # lat = np.radians(lat_deg)
    # lon = np.radians(lon_deg)
    # N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    # X = (N + alt_m) * np.cos(lat) * np.cos(lon)
    # Y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    # Z = (N * (1 - e2) + alt_m) * np.sin(lat)

    # return X, Y, Z
    X = []
    Y = []
    Z = []
    for latitude, longitude, altitude in zip(lat_deg, lon_deg, alt_m):
        ecef_coords = glp.geodetic_to_ecef(np.array([[latitude], [longitude], [altitude]]))
        X.append(ecef_coords[0])
        Y.append(ecef_coords[1])
        Z.append(ecef_coords[2])
    return np.array(X), np.array(Y), np.array(Z)


# ------------------------------
# Merge receiver position into TXT
# ------------------------------
def build_nmea_receiver_positions(nmea_data, txt_data):
    # Merge using nearest timestamps
    df = pd.merge_asof(
        txt_data.sort_values("gps_millis"),
        nmea_data.sort_values("gps_millis"),
        on="gps_millis",
        direction="nearest",
    )

    # Fill missing NMEA fields
    for col in ["lat_rx_deg", "lon_rx_deg", "alt_rx_m"]:
        df[col] = df[col].interpolate().bfill().ffill()

    # Compute ECEF
    df["rx_ecef_x"], df["rx_ecef_y"], df["rx_ecef_z"] = lla_to_ecef(
        df["lat_rx_deg"].to_numpy(),
        df["lon_rx_deg"].to_numpy(),
        df["alt_rx_m"].to_numpy(),
    )

    return df



# ------------------------------
# Compute elevation / azimuth
# ------------------------------
def add_elevation_azimuth(df):
    # out = df.copy()

    # dx = out["SvPositionEcefXMeters"] - out["rx_ecef_x"]
    # dy = out["SvPositionEcefYMeters"] - out["rx_ecef_y"]
    # dz = out["SvPositionEcefZMeters"] - out["rx_ecef_z"]

    # lat = np.radians(out["lat_rx_deg"])
    # lon = np.radians(out["lon_rx_deg"])

    # t = np.cos(lat) * np.cos(lon) * dx + np.cos(lat) * np.sin(lon) * dy + np.sin(lat) * dz
    # e = -np.sin(lon) * dx + np.cos(lon) * dy
    # n = -np.sin(lat) * np.cos(lon) * dx - np.sin(lat) * np.sin(lon) * dy + np.cos(lat) * dz

    # out["azimuth_deg"] = np.degrees(np.arctan2(e, n))
    # out["elevation_deg"] = np.degrees(np.arctan2(t, np.sqrt(e**2 + n**2)))

    # out.loc[out["azimuth_deg"] < 0, "azimuth_deg"] += 360
    # return out
    out = df.copy()
    az = []
    el = []
    
    for x_rx_m, y_rx_m, z_rx_m, x_sv_m, y_sv_m, z_sv_m in zip(
        out["rx_ecef_x"].values, out["rx_ecef_y"].values, out["rx_ecef_z"].values,
        out["SvPositionEcefXMeters"].values, out["SvPositionEcefYMeters"].values,
        out["SvPositionEcefZMeters"].values):

        el_az = glp.ecef_to_el_az(np.array([x_rx_m, y_rx_m, z_rx_m]).reshape(3, 1),
                                  np.array([x_sv_m, y_sv_m, z_sv_m]).reshape(3, 1))
        el.append(el_az[0][0])
        az.append(el_az[1][0])
        
    out["azimuth_deg"] = np.array(az)
    out["elevation_deg"] = np.array(el)
    return out


# ------------------------------
# Build per-satellite DataFrames
# ------------------------------
def build_satellite_dataframes(txt_full):
    sat_dfs = {}

    SPEED_OF_LIGHT = 299792458.0  # m/s

    # --- Compute pseudorange if possible ---
    if all(col in txt_full.columns for col in [
        "TimeNanos", "FullBiasNanos", "BiasNanos", "ReceivedSvTimeNanos"
    ]):
        txt_full["raw_pr_m"] = (
            (txt_full["TimeNanos"]
             - txt_full["FullBiasNanos"]
             - txt_full["BiasNanos"]
             - txt_full["ReceivedSvTimeNanos"])
            * 1e-9 * SPEED_OF_LIGHT
        )
    else:
        txt_full["raw_pr_m"] = np.nan

    # Pseudorange uncertainty → not available directly
    txt_full["raw_pr_sigma_m"] = np.nan

    # --- Now build per-satellite dataframes ---
    for svid in txt_full["Svid"].unique():
        df_sat = txt_full[txt_full["Svid"] == svid].copy()

        sat_dfs[svid] = df_sat[
            [
                "gps_millis",
                "Cn0DbHz",
                "raw_pr_m",
                "raw_pr_sigma_m",
                "PseudorangeRateMetersPerSecond",
                "elevation_deg",
                "azimuth_deg",
            ]
        ]

    return sat_dfs





# ============================================================
#  MAIN PIPELINE
# ============================================================
def organize_raw_data(directory_input):

    base = f"./{directory_input}/gnss_log_{directory_input}"
    txt_file = base + ".txt"
    nmea_file = base + ".nmea"
    rinex_obs_file = base + ".25o"

    year, month, day, doy = get_day_of_year(directory_input)
    # rinex_nav_file = f"./{directory_input}/brdc{doy}0.25n"

    txt_raw = glp.AndroidRawGnss(txt_file)
    nmea_raw = glp.Nmea(nmea_file)

    updated_rinex = convert_rinex_for_glp(rinex_obs_file)
    rinex_obs_raw = glp.RinexObs(updated_rinex)
    #rinex_nav_raw = glp.RinexNav(rinex_nav_file)

    txt_data = txt_raw.preprocess(txt_file)
    txt_data = add_gps_millis_to_txt(txt_data)
    nmea_data = nmea_raw.pandas_df()
    
    # Quick Check:
    print("gps_millis in TXT:", "gps_millis" in txt_data.columns)
    print("Any NaN in gps_millis:", txt_data["gps_millis"].isna().sum())


    # ---- FIX: give NMEA a time axis so merge works ----
    nmea_data = add_time_to_nmea(nmea_data, txt_data)

    # ---- Merge NMEA receiver position into txt_data ----
    txt_with_rx = build_nmea_receiver_positions(nmea_data, txt_data)

    # ---- Compute elevation & azimuth ----
    txt_full = add_elevation_azimuth(txt_with_rx)

    # ---- Group into per-satellite DataFrames ----
    sat_dfs = build_satellite_dataframes(txt_full)

    return {
        "txt_full": txt_full,
        "satellite_dataframes": sat_dfs,
    }
